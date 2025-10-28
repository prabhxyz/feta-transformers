
# trainer on synthetic task + logging + simple latency proxy
import os, time, math, csv, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import toy_block
from .feta import eq_loss, nuclear_surrogate, compile_prompt_to_lora

class toy_data:
    # simple sequence -> parity of sums task; nontrivial but learnable
    def __init__(self, d=64, n=64, classes=4, device='cpu'):
        self.d, self.n, self.c = d, n, classes
        self.device = device
        torch.manual_seed(0)

        # make a random linear readout target with small nonlinearity
        self.W = torch.randn(d, classes)/math.sqrt(d)
        self.u = torch.randn(classes)

    def sample(self, b):
        x = torch.randn(b, self.n, self.d)
        s = x.sum(dim=1) @ self.W + self.u
        y = s.argmax(dim=-1)
        return x, y

def latency_ms(fn, repeat=5):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000)
    ts.sort()
    return ts[len(ts)//2]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['prompt','lora','hybrid'], default='hybrid')
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--compile', type=int, default=0)
    ap.add_argument('--m', type=int, default=8)        # prefix tokens
    ap.add_argument('--r', type=int, default=2)        # lora rank
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--h', type=int, default=4)
    ap.add_argument('--seq', type=int, default=64)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--out', type=str, default='/mnt/data/feta_impl/results/metrics.csv')
    args = ap.parse_args()

    device = 'cpu' # TODO: change to cuda later
    data = toy_data(d=args.d, n=args.seq, device=device)
    m = args.m if args.mode in ['prompt','hybrid'] else 0
    rq = rk = rv = ro = (args.r if args.mode in ['lora','hybrid'] else 0)

    model = toy_block(d=args.d, h=args.h, r_q=rq, r_k=rk, r_v=rv, r_o=ro, m_prefix=m).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    header = ['step','mode','loss','acc','eq','latency_ms','use_prefix','use_lora','compiled']
    new_file = not os.path.exists(args.out)
    f = open(args.out, 'a', newline='')
    w = csv.writer(f)
    if new_file: w.writerow(header)

    compiled = 0
    for step in range(1, args.steps+1):
        x,y = data.sample(args.bs)
        # choose branch
        use_prefix = (args.mode in ['prompt','hybrid'])
        use_lora = (args.mode in ['lora','hybrid'])

        # forward for task loss
        z,_ = model(x, use_prefix=use_prefix)
        # simple pooled head
        logits = z.mean(dim=1) @ torch.randn(args.d, data.c)
        # keep head fixed random to force block to adapt
        logits = logits.to(device)
        loss = F.cross_entropy(logits, y)

        # add eq loss if hybrid
        eql = 0.0
        if args.mode == 'hybrid':
            eql = eq_loss(model, x)
            loss = loss + 0.1 * eql

        # rank pressure if lora present
        reg = 0.0
        if use_lora:
            mats = []
            for n,p in model.named_parameters():
                if n.endswith('.a') or n.endswith('.b'):
                    mats.append(p)
            reg = nuclear_surrogate(mats) * 1e-4
            loss = loss + reg

        opt.zero_grad(); loss.backward(); opt.step()

        # accuracy estimate
        acc = (logits.argmax(-1) == y).float().mean().item()

        # latency proxy: measure forward with and without prefixes to see cost
        lat = latency_ms(lambda: model(x, use_prefix=use_prefix))

        w.writerow([step, args.mode, float(loss.item()), float(acc), float(eql if isinstance(eql,float)==False else 0.0), float(lat), int(use_prefix), int(use_lora), compiled])
        f.flush()

        # quick compile mid-run to show effect
        if args.compile==1 and step == args.steps//2 and args.mode=='hybrid' and compiled==0:
            compile_prompt_to_lora(model, data, iters=60, lr=5e-3)
            compiled = 1

    f.close()

if __name__ == '__main__':
    main()
