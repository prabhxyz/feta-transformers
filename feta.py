# feta losses + compile step
import time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import toy_block

def nuclear_surrogate(mats, eps=1e-5):
    # quick smooth approx to nuclear norm: trace sqrt(M M^T)
    reg = 0.0
    for m in mats:
        if m is None: continue
        s = torch.linalg.svdvals(m)
        reg = reg + (s + eps).sum()
    return reg

@torch.no_grad()
def forward_collect(model, x, use_prefix, use_lora):
    # turn off lora by zeroing ranks if needed
    # implemented by toggling use_prefix and assuming lora always present in weights
    z, h = model(x, use_prefix=use_prefix)
    return z, h

def eq_loss(model, x):
    z0, h0 = forward_collect(model, x, use_prefix=False, use_lora=True)
    zP, hP = forward_collect(model, x, use_prefix=True, use_lora=False)
    zL, hL = forward_collect(model, x, use_prefix=False, use_lora=True)
    # compare block outputs (post-attn pre-residual stored as h['y'])
    return (hP['y'] - hL['y']).pow(2).mean()

def compile_prompt_to_lora(model, data, iters=100, lr=1e-2):
    # fit lora a,b to match prompt effect, then zero prefixes
    opt = torch.optim.Adam([
        p for n,p in model.named_parameters() if ('a' in n or 'b' in n)
    ], lr=lr)
    for p in [model.kp, model.vp]:
        if p is not None: p.requires_grad_(False)

    for _ in range(iters):
        x = data.sample(64).to(next(model.parameters()).device)
        zP,_ = model(x, use_prefix=True)
        z0,_ = model(x, use_prefix=False)
        loss = (zP - z0).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # drop prefixes
    if model.kp is not None: model.kp.data.zero_()
    if model.vp is not None: model.vp.data.zero_()
    return
