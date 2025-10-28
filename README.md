## FETA: functionally-equivalent token adaptation (WIP)

- currently minimal code to demo idea
- runs on synthetic task; shows loss curves, eq loss, compile keeps quality, and runtime cost drops

### run

```bash
python -m src.train --mode hybrid --steps 150
python -m src.train --mode prompt --steps 150
python -m src.train --mode lora --steps 150
python -m src.train --mode hybrid --steps 150 --compile 1
python -m src.plot_metrics
```
