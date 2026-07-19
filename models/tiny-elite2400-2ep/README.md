# tiny-elite2400-2ep — saved best model

The best WDL model produced so far (REPORT.md **Run 12**), committed here because
it is relatively expensive to reproduce (~2.8 days on a 4-core CPU). This is a
self-describing checkpoint: `model.safetensors` (weights, 3.6 MB), `model.toml`
(architecture), and `meta.json` (fitted temperature + baselines). `eval`,
`predict`, and `replay` load the architecture from `model.toml`, so **no arch
flags are needed**.

## What it is

- **Arch:** `tiny` (0.9M params) — d_model 128, 4 layers, 4 heads, FFN 512,
  GAB (avg-pool), board-only / Elo-free.
- **Trained on:** the elite Elo≥2400 all-TWIC dataset (~7.77M positions,
  `--require-elo --min-elo 2400 --min-ply 20 --max-ply 100 --positions-per-game 10`,
  train = TWIC issues 210–1629). CPU + Intel MKL, batch 1024, lr 5e-4,
  2-epoch cosine (15,176 steps), ~2.8 days.
- **Selection/calibration:** best-val checkpoint (val log-loss **1.0004** at step
  10,000, converged), temperature **T = 1.295** fitted on the twic1630–1639 test set.

## Held-out results (twic1640–1649, 94,567 positions)

| Metric | value | material baseline | base-rate |
|---|---|---|---|
| Accuracy | **48.3%** | 38.4% | 31.9% |
| Log-loss ↓ | **1.011** | 1.089 | 1.106 |
| Brier ↓ | **0.607** | 0.660 | 0.672 |
| ECE ↓ | **1.5%** | 2.1% | 7.2% |

Best model in the project to date; beats `nano` (Run 10, 44.5%) and 1-epoch
`tiny` (Run 11, 47.1%). Draw-biased but least of the three (predicts draw ~47%;
recall win 51 / draw 62 / loss 31). Full analysis in `../../REPORT.md` §Run 12.

## Use it

```bash
cargo build --release   # CPU-only off macOS; Metal+Accelerate on M1
./target/release/chess-wdl-predict --checkpoint models/tiny-elite2400-2ep \
  --fen "<FEN>" --device cpu
./target/release/chess-wdl-eval    --checkpoint models/tiny-elite2400-2ep \
  --data data/shards/eval --device cpu --baseline
```

WDL is from the **side-to-move's** perspective. Outputs are calibrated with the
stored T; pass nothing extra.
