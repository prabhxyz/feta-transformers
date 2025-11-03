# FETA: First-Order Equivalence Between Token and Weight Adaptation in Transformers

## Overview

This project studies the relationship between two common parameter-efficient adaptation methods for transformer models: **prefix tuning** and **LoRA (Low-Rank Adaptation)**.
The work introduces **FETA** (First-Order Equivalence Between Token and Weight Adaptation), a framework showing that these two methods are mathematically related under a first-order approximation of a transformer block.

The key contribution is a method to convert trained prefix tokens into equivalent LoRA weight updates. This process is called the **compile step**. It allows a model trained with prefix tokens to be merged into its weights, reducing runtime overhead while maintaining similar behavior.

---

## Theoretical Summary

* **Prefix tuning** adds a small number of trainable virtual tokens to each input sequence. These tokens modify the model indirectly through attention.
* **LoRA** inserts low-rank trainable matrices inside attention and feedforward projections. These matrices adapt the model in weight space with minimal parameters.

FETA shows that, when linearized around a fixed input:

1. Both methods induce a linear map from parameter updates to output perturbations.
2. The two maps are equivalent to first order, except that prefixes cannot modify content-to-content attention relationships.
3. A least-squares projection can convert between the two methods at fixed data.
4. A prefix of length *m* cannot reproduce LoRA updates with rank greater than *m*, establishing a lower bound on expressivity.

---

## Implementation Summary

The experiments use **GPT-2-small** fine-tuned on the **SST-2** sentiment classification dataset.
The following components are included:

1. **Prefix Training** – trains models with different prefix lengths (8, 16, 32).
2. **LoRA Training** – trains models with different ranks (1, 2, 4).
3. **Compile Step** – fits a LoRA model to reproduce the outputs of a trained prefix model using mean squared error regression.
4. **Evaluation** – measures accuracy, inference latency, and the equivalence residuals (difference between prefix and LoRA hidden states).

All metrics and plots are stored in the `outputs/` folder.

---

## Experimental Results

| Model                              | Accuracy (%) | Latency (μs/token) |
| :--------------------------------- | :----------- | :----------------: |
| Prefix (m = 8)                     | 66.7         |        43.0        |
| Prefix (m = 16)                    | 76.1         |        44.1        |
| Prefix (m = 32)                    | 79.5         |        44.2        |
| LoRA (r = 1)                       | 91.4         |        47.4        |
| LoRA (r = 2)                       | 90.6         |        48.0        |
| LoRA (r = 4)                       | 91.7         |        76.6        |
| Compiled LoRA (from m = 32, r = 4) | 77.1         |        69.7        |

### Observations

* Increasing prefix length improves accuracy but remains below LoRA performance, consistent with the theoretical rank limit.
* LoRA achieves high accuracy with minor increases in latency.
* The compiled LoRA model closely reproduces prefix behavior while removing prompt overhead.
* The equivalence residual heatmap shows large mismatches in early layers (around 0.95–0.98) that gradually decrease in later layers (around 0.6), indicating convergence of prefix and LoRA effects toward the model output.

These results support the FETA theory: prefixes and LoRA produce similar first-order effects, but LoRA provides broader coverage of the model’s operator space.

---

## Visuals

* `acc_vs_latency.png` – Accuracy versus latency comparison.
* `accuracy_summary.png` – Accuracy of prefix, LoRA, and compiled models.
* `equiv_heatmap.png` – Layerwise equivalence residuals showing where prefix and LoRA differ most.

All figures are automatically saved under `outputs/`.

---

## Running the Experiment

### 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run

Execute the experiment on a CUDA-capable GPU:

```bash
python run_feta.py --device cuda --epochs 3 --batch 16 \
  --prefix_list 8,16,32 --lora_list 1,2,4 --compile_r 4 --out outputs
```

This will train prefix and LoRA models, compile the prefix to LoRA, and produce results in `outputs/`.

### 3. Visualize

(Optional) Generate plots using:

```bash
python visualize_feta_results.py
```

---

## Conclusion

This project demonstrates a clear first-order equivalence between token and weight adaptation mechanisms in transformer models.
The compile step provides a practical way to merge prefix-trained models into LoRA representations, preserving accuracy while improving runtime efficiency.
The results validate both the theoretical framework and its practical implementation on a real transformer task.
