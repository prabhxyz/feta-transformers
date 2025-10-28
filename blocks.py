# minimal toy transformer block with prefixes + lora

import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x): return x * torch.sigmoid(x)

class lora_linear(nn.Module):
    def __init__(self, in_f, out_f, r=0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_f, out_f) / math.sqrt(in_f))
        self.r = r
        if r > 0:
            self.a = nn.Parameter(torch.zeros(in_f, r))
            self.b = nn.Parameter(torch.zeros(r, out_f))
        else:
            self.register_parameter('a', None)
            self.register_parameter('b', None)

    def forward(self, x):
        base = x @ self.w
        if self.r > 0:
            base = base + x @ (self.a @ self.b)
        return base

class toy_block(nn.Module):
    def __init__(self, d=64, h=4, r_q=0, r_k=0, r_v=0, r_o=0, m_prefix=0):
        super().__init__()
        self.d = d; self.h = h; self.dk = d // h
        # qkv + o with lora
        self.wq = lora_linear(d, d, r_q)
        self.wk = lora_linear(d, d, r_k)
        self.wv = lora_linear(d, d, r_v)
        self.wo = lora_linear(d, d, r_o)
        # prefixes (per layer kv)
        self.m_prefix = m_prefix
        if m_prefix > 0:
            self.kp = nn.Parameter(torch.zeros(m_prefix, self.d))
            self.vp = nn.Parameter(torch.zeros(m_prefix, self.d))
        else:
            self.register_parameter('kp', None); self.register_parameter('vp', None)
        self.ln = nn.LayerNorm(d)

    def split_heads(self, x):
        b, n, d = x.shape
        x = x.view(b, n, self.h, self.dk).transpose(1,2)  # b,h,n,dk
        return x

    def combine(self, x):
        b,h,n,dk = x.shape
        return x.transpose(1,2).contiguous().view(b, n, h*dk)

    def attn(self, q, k, v):
        s = (q @ k.transpose(-2,-1)) / math.sqrt(q.size(-1))
        a = torch.softmax(s, dim=-1)
        return a @ v, a

    def forward(self, x, use_prefix=True):
        b,n,d = x.shape
        x = self.ln(x)
        q = self.split_heads(self.wq(x))
        k = self.split_heads(self.wk(x))
        v = self.split_heads(self.wv(x))

        if use_prefix and self.m_prefix > 0:
            # expand prefixes across batch
            kp = self.kp.unsqueeze(0).expand(b, -1, -1)
            vp = self.vp.unsqueeze(0).expand(b, -1, -1)
            kp = self.split_heads(kp)
            vp = self.split_heads(vp)
            k = torch.cat([k, kp], dim=2)
            v = torch.cat([v, vp], dim=2)

        y, a = self.attn(q, k, v)
        y = self.combine(y)
        z = self.wo(y)
        return x + z, {'q':q.detach(), 'k':k.detach(), 'v':v.detach(), 'a':a.detach(), 'y':y.detach()}
