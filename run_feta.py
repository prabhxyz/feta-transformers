import argparse, os, json, time, random
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from inspect import signature

# ---- PyTorch 2.9 TF32 knobs (no-op on older PT) ----
try:
    # valid: 'ieee' | 'tf32' | 'none'
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
except Exception:
    pass

from datasets import load_dataset
from transformers import (
    GPT2TokenizerFast, GPT2ForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from transformers.utils import is_flash_attn_2_available
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType

# ----------------------------
# Helpers & GPU setup
# ----------------------------
def _can_torch_compile() -> bool:
    """Return True only if Triton is importable; required by torch.compile/inductor."""
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False

def set_seed(s=42, deterministic=False):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def device_guard(device_str: str) -> str:
    if device_str == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu or provision a GPU.")
    return device_str

def ensure_pad(model, tok):
    """Ensure tokenizer has a PAD token and propagate to model config/generation config."""
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id or tok.eos_token_id
    model.config.pad_token_id = pad_id
    try:
        model.generation_config.pad_token_id = pad_id
    except Exception:
        pass
    return model

# ----------------------------
# Data
# ----------------------------
def prepare_data(tokenizer, max_len=128, batch_size=16, workers=4):
    ds = load_dataset("glue", "sst2")

    # dynamic padding is fine; collect_hidden will align lengths
    def tok(ex):
        return tokenizer(ex["sentence"], truncation=True, max_length=max_len)

    ds = ds.map(tok, batched=True, num_proc=None)
    ds = ds.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    ds.set_format(type="torch", columns=cols)

    dc = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    train = DataLoader(
        ds["train"], batch_size=batch_size, shuffle=True,
        collate_fn=dc, pin_memory=True, num_workers=workers, persistent_workers=(workers > 0)
    )
    val = DataLoader(
        ds["validation"], batch_size=batch_size, shuffle=False,
        collate_fn=dc, pin_memory=True, num_workers=workers, persistent_workers=(workers > 0)
    )
    return train, val, ds

# ----------------------------
# Model builders
# ----------------------------
def build_base(num_labels=2, flash_ok=True, device="cuda"):
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=num_labels)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = ensure_pad(model, tok)
    if flash_ok and is_flash_attn_2_available():
        try:
            model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass
    return model.to(device), tok

def wrap_lora(model, r=4, alpha=16, dropout=0.05):
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["c_attn", "c_proj"]
    )
    return get_peft_model(model, cfg)

def wrap_prefix(model, m=16):
    cfg = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=m)
    return get_peft_model(model, cfg)

# ----------------------------
# Training / eval helpers (version-compatible)
# ----------------------------
def train_with_trainer(peft_model, tok, train_loader, val_loader, outdir, epochs=3, lr=2e-4):
    import inspect

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = (not bf16) and torch.cuda.is_available()

    base_kwargs = dict(
        output_dir=outdir,
        per_device_train_batch_size=train_loader.batch_size,
        per_device_eval_batch_size=val_loader.batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        report_to="none",
        dataloader_pin_memory=True,
        remove_unused_columns=False,   # critical with PEFT-wrapped forward
    )

    # Build TrainingArguments compatibly across versions
    try:
        args = TrainingArguments(
            **base_kwargs,
            evaluation_strategy="epoch",
            save_strategy="no",
            fp16=fp16, bf16=bf16,
            gradient_checkpointing=False,
        )
    except TypeError:
        sig = inspect.signature(TrainingArguments.__init__)
        kwargs = dict(base_kwargs)
        if "eval_strategy" in sig.parameters: kwargs["eval_strategy"] = "epoch"
        if "evaluation_strategy" in sig.parameters: kwargs["evaluation_strategy"] = "epoch"
        if "save_strategy" in sig.parameters: kwargs["save_strategy"] = "no"
        if "report_to" not in sig.parameters: kwargs.pop("report_to", None)
        if "fp16" in sig.parameters: kwargs["fp16"] = fp16
        if "bf16" in sig.parameters: kwargs["bf16"] = bf16
        if "gradient_checkpointing" in sig.parameters: kwargs["gradient_checkpointing"] = False
        if "remove_unused_columns" not in sig.parameters: kwargs.pop("remove_unused_columns", None)
        args = TrainingArguments(**kwargs)

    # Conditionally enable compile (skip on Windows/no Triton)
    if _can_torch_compile():
        try:
            peft_model = torch.compile(peft_model, mode="max-autotune")
        except Exception:
            pass

    collator = DataCollatorWithPadding(tok, pad_to_multiple_of=8)

    trainer_kwargs = dict(
        model=peft_model,
        args=args,
        train_dataset=train_loader.dataset,
        eval_dataset=val_loader.dataset,
        tokenizer=tok,  # future rename to processing_class in v5
        data_collator=collator,
    )
    if "label_names" in signature(Trainer.__init__).parameters:
        trainer_kwargs["label_names"] = ["labels"]

    tr = Trainer(**trainer_kwargs)
    tr.train()
    metrics = tr.evaluate()
    return metrics, peft_model

@torch.no_grad()
def accuracy(model, loader, device="cuda"):
    model.eval()
    correct = 0; total = 0
    autocast_dtype = None
    if torch.cuda.is_available():
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.no_grad()
        with ctx:
            logits = model(**batch).logits
        preds = logits.argmax(-1)
        correct += (preds == batch["labels"]).sum().item()
        total += preds.numel()
    return correct / total

@torch.no_grad()
def ms_per_token(model, loader, device="cuda", warmup=20, iters=80):
    """CUDA events for precise GPU time. Returns ms/token."""
    if device != "cuda":
        model.eval()
        times, toks = [], []
        it = 0
        for b in loader:
            if it >= iters: break
            b = {k: v.to(device) for k, v in b.items()}
            t0 = time.time(); _ = model(**b); t1 = time.time()
            times.append((t1 - t0) * 1000.0)
            toks.append(int(b["attention_mask"].sum().item()))
            it += 1
        return float(np.sum(times) / max(1, np.sum(toks)))

    model.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # warmup
    for i, b in enumerate(loader):
        if i >= warmup: break
        _ = model(**{k: v.to(device, non_blocking=True) for k, v in b.items()})
    # measure
    ms_total = 0.0; toks_total = 0; it = 0
    for b in loader:
        if it >= iters: break
        batch = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        starter.record()
        _ = model(**batch)
        ender.record()
        torch.cuda.synchronize()
        ms_total += starter.elapsed_time(ender)
        toks_total += int(batch["attention_mask"].sum().item())
        it += 1
    return ms_total / max(1, toks_total)

@torch.no_grad()
def collect_hidden(model, loader, device="cuda", max_batches=16):
    """Average hidden states across batches with possibly different sequence lengths.
       Returns [L+1, T_max, H] on CPU."""
    import torch.nn.functional as F

    model.eval()
    out_sum = None; count = 0
    autocast_dtype = None
    if torch.cuda.is_available():
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def pad_to_T(x, T_target):
        # x: [L+1, T, H] -> [L+1, T_target, H]
        T = x.shape[1]
        if T == T_target:
            return x
        if T < T_target:
            return F.pad(x, (0, 0, 0, T_target - T))  # pad along sequence dim
        else:
            return x[:, :T_target, :]

    for i, b in enumerate(loader):
        if i >= max_batches: break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.no_grad()
        with ctx:
            o = model(**b, output_hidden_states=True)
        H = torch.stack([h.float() for h in o.hidden_states], dim=0)  # [L+1,B,T,H]
        H = H.mean(dim=1)  # -> [L+1,T,H] (mean over batch)

        if out_sum is None:
            out_sum = H
        else:
            # align sequence length before summation
            T_target = max(out_sum.shape[1], H.shape[1])
            out_sum = pad_to_T(out_sum, T_target) + pad_to_T(H, T_target)
        count += 1

    return (out_sum / count).cpu()

def heatmap_equivalence(base, pfx, lra, val_loader, device="cuda"):
    H0 = collect_hidden(base, val_loader, device)
    Hp = collect_hidden(pfx,  val_loader, device)
    Hl = collect_hidden(lra,  val_loader, device)
    dZp = Hp - H0
    dZl = Hl - H0
    layers = dZp.shape[0]
    res = []
    for L in range(1, layers):  # skip embedding layer 0
        num = torch.norm(dZp[L] - dZl[L], p="fro")
        den = torch.norm(dZp[L], p="fro") + 1e-9
        res.append(float((num / den).item()))
    return res  # len = num_layers

def compile_prefix_to_lora(prefix_model, base_ckpt="gpt2", r=4, calib_loader=None,
                           iters=1000, lr=1e-3, device="cuda"):
    """Gradient-based regression on logits for compile step."""
    base = GPT2ForSequenceClassification.from_pretrained(base_ckpt, num_labels=2).to(device)
    # align pad id with the (already ensured) prefix model
    base.config.pad_token_id = getattr(prefix_model.config, "pad_token_id", None) or base.config.pad_token_id
    if is_flash_attn_2_available():
        try: base.config.attn_implementation = "flash_attention_2"
        except Exception: pass

    # wrap with LoRA
    lora = wrap_lora(base, r=r).to(device)
    # ensure pad on peft-wrapped model too
    lora.config.pad_token_id = base.config.pad_token_id

    # >>> Freeze everything EXCEPT LoRA adapter params
    for n, p in lora.named_parameters():
        p.requires_grad = ("lora_" in n)

    # guard torch.compile (Windows without Triton)
    if _can_torch_compile():
        try:
            lora = torch.compile(lora, mode="max-autotune")
        except Exception:
            pass

    # optimize only trainable params (the LoRA ones)
    trainable = [p for p in lora.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)

    lora.train()
    prefix_model.eval()
    loss_hist = []

    autocast_dtype = None
    if torch.cuda.is_available():
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    it = 0
    while it < iters:
        for b in calib_loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.no_grad():
                tgt = prefix_model(**b).logits.detach()  # target is prefix logits (no grad)
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype else torch.no_grad()
            with ctx:
                pred = lora(**b).logits
                loss = F.mse_loss(pred, tgt)  # dtype cast not necessary; keeps grad graph
            loss.backward()
            opt.step()
            loss_hist.append(float(loss.item()))
            it += 1
            if it >= iters:
                break

    tail = loss_hist[-min(50, len(loss_hist)):] if loss_hist else [0.0]
    return lora, float(np.mean(tail))

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", type=str)
    ap.add_argument("--epochs", default=3, type=int)
    ap.add_argument("--batch", default=16, type=int)
    ap.add_argument("--lr", default=2e-4, type=float)
    ap.add_argument("--prefix_list", default="8,16,32", type=str)
    ap.add_argument("--lora_list",   default="1,2,4", type=str)
    ap.add_argument("--compile_r",   default=4, type=int)
    ap.add_argument("--calib_size",  default=1000, type=int)
    ap.add_argument("--iters_compile", default=1000, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--max_len", default=128, type=int)
    args = ap.parse_args()

    args.device = device_guard(args.device)
    os.makedirs(args.out, exist_ok=True)
    set_seed(42, deterministic=args.deterministic)

    # Base model (unadapted) + tokenizer
    base, tok = build_base(device=args.device)
    train_loader, val_loader, ds = prepare_data(tok, max_len=args.max_len, batch_size=args.batch, workers=args.workers)

    pref_rows, lora_rows, comp_rows = [], [], []

    # ---- Prefix runs ----
    for m in [int(x) for x in args.prefix_list.split(",") if x]:
        model_p = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(args.device)
        model_p = ensure_pad(model_p, tok)
        if is_flash_attn_2_available():
            try: model_p.config.attn_implementation = "flash_attention_2"
            except Exception: pass
        pfx = wrap_prefix(model_p, m=m)
        pfx = ensure_pad(pfx, tok)
        _metrics, pfx = train_with_trainer(
            pfx, tok, train_loader, val_loader,
            outdir=os.path.join(args.out, f"prefix_m{m}"),
            epochs=args.epochs, lr=args.lr
        )
        acc = accuracy(pfx, val_loader, device=args.device)
        ms  = ms_per_token(pfx, val_loader, device=args.device)
        pref_rows.append((ms, acc))

    # ---- LoRA runs ----
    last_lora_for_heatmap = None
    for r in [int(x) for x in args.lora_list.split(",") if x]:
        model_l = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(args.device)
        model_l = ensure_pad(model_l, tok)
        if is_flash_attn_2_available():
            try: model_l.config.attn_implementation = "flash_attention_2"
            except Exception: pass
        lra = wrap_lora(model_l, r=r)
        lra = ensure_pad(lra, tok)
        _metrics, lra = train_with_trainer(
            lra, tok, train_loader, val_loader,
            outdir=os.path.join(args.out, f"lora_r{r}"),
            epochs=args.epochs, lr=args.lr
        )
        acc = accuracy(lra, val_loader, device=args.device)
        ms  = ms_per_token(lra, val_loader, device=args.device)
        lora_rows.append((ms, acc))
        last_lora_for_heatmap = lra

    # ---- Heatmap (Prefix m=last vs LoRA r=last) ----
    m_heat = int(args.prefix_list.split(",")[-1])
    model_p_heat = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(args.device)
    model_p_heat = ensure_pad(model_p_heat, tok)
    if is_flash_attn_2_available():
        try: model_p_heat.config.attn_implementation = "flash_attention_2"
        except Exception: pass
    pfx_heat = wrap_prefix(model_p_heat, m=m_heat)
    pfx_heat = ensure_pad(pfx_heat, tok)
    _metrics, pfx_heat = train_with_trainer(
        pfx_heat, tok, train_loader, val_loader,
        outdir=os.path.join(args.out, f"prefix_m{m_heat}_heat"),
        epochs=args.epochs, lr=args.lr
    )

    base.eval()
    hm_vals = heatmap_equivalence(base, pfx_heat, last_lora_for_heatmap, val_loader, device=args.device)
    with open(os.path.join(args.out, "heatmap.csv"), "w") as f:
        f.write(",".join([f"{x:.6f}" for x in hm_vals]) + "\n")  # GPT-2 small -> 12 columns

    # ---- Compile prefix (m=max) → LoRA (rank=compile_r) ----
    calib = ds["train"].select(range(min(args.calib_size, ds["train"].num_rows)))
    dc = DataCollatorWithPadding(tok, pad_to_multiple_of=8)
    calib_loader = DataLoader(
        calib, batch_size=args.batch, shuffle=True,
        collate_fn=dc, pin_memory=True, num_workers=args.workers, persistent_workers=(args.workers > 0)
    )
    comp_lora, comp_mse = compile_prefix_to_lora(
        pfx_heat, r=args.compile_r, calib_loader=calib_loader,
        iters=args.iters_compile, device=args.device
    )
    # ensure pad on compiled LoRA too (it inherits from base, but belt & suspenders)
    comp_lora.config.pad_token_id = getattr(pfx_heat.config, "pad_token_id", comp_lora.config.pad_token_id)

    comp_acc = accuracy(comp_lora, val_loader, device=args.device)
    comp_ms  = ms_per_token(comp_lora, val_loader, device=args.device)
    comp_rows.append((comp_ms, comp_acc))

    # ---- Write CSVs ----
    def dump_csv(path, rows):
        with open(path, "w") as f:
            f.write("ms_per_tok,acc\n")
            for ms, acc in rows:
                f.write(f"{ms:.6f},{acc*100.0:.4f}\n")  # acc in %

    os.makedirs(args.out, exist_ok=True)
    dump_csv(os.path.join(args.out, "acc_latency_prefix.csv"),   pref_rows)
    dump_csv(os.path.join(args.out, "acc_latency_lora.csv"),     lora_rows)
    dump_csv(os.path.join(args.out, "acc_latency_compiled.csv"), comp_rows)

    # ---- Metrics JSON ----
    metrics = {
        "prefix": [{"m": int(m), "ms_per_tok": float(ms), "acc_pct": float(acc*100.0)}
                   for (m, (ms, acc)) in zip([int(x) for x in args.prefix_list.split(",") if x], pref_rows)],
        "lora":   [{"r": int(r), "ms_per_tok": float(ms), "acc_pct": float(acc*100.0)}
                   for (r, (ms, acc)) in zip([int(x) for x in args.lora_list.split(",") if x], lora_rows)],
        "compiled_lora": {"from_m": m_heat, "r": args.compile_r,
                          "ms_per_tok": float(comp_ms), "acc_pct": float(comp_acc*100.0),
                          "compile_mse": float(comp_mse)},
        "notes": "SST-2 validation; latency = CUDA-event ms/token in eval.",
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote CSVs in {args.out}")

if __name__ == "__main__":
    main()
