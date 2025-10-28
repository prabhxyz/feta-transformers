# make simple plots from csv logs
# one chart per figure; matplotlib only
import csv, os
import numpy as np
import matplotlib.pyplot as plt

IN = "metrics.csv"
OUTD = "/results/plots"

def load():
    rows = []
    with open(IN,'r') as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ['step','loss','acc','eq','latency_ms','use_prefix','use_lora','compiled']:
                row[k] = float(row[k])
            rows.append(row)
    return rows

def plot_metric(rows, metric, fname, title):
    plt.figure()
    for mode in sorted(set([r['mode'] for r in rows])):
        xs = [r['step'] for r in rows if r['mode']==mode]
        ys = [r[metric] for r in rows if r['mode']==mode]
        plt.plot(xs, ys, label=mode)
    plt.xlabel('step'); plt.ylabel(metric); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTD, fname)); plt.close()

def main():
    os.makedirs(OUTD, exist_ok=True)
    rows = load()
    plot_metric(rows, 'loss', 'loss.png', 'task loss')
    plot_metric(rows, 'acc',  'acc.png',  'accuracy')
    plot_metric(rows, 'eq',   'eq.png',   'equivalence loss (hybrid only)')
    plot_metric(rows, 'latency_ms', 'latency.png', 'latency ms (prefix adds cost; compile removes)')

if __name__ == '__main__':
    main()
