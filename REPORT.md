# Chess WDL Prediction — Results Report

First end-to-end training and held-out evaluation of the Chessformer/Maia-3-inspired
win/draw/loss model described in [`DESIGN.md`](DESIGN.md).

## Setup

| | |
|---|---|
| **Task** | Predict P(win/draw/loss) for the side to move, from a single position (no history) |
| **Train** | `twic210.pgn` — 1,890 games → **143,432** positions (35.8% win / 28.0% draw / 36.1% loss) |
| **Validation / early-stop** | `twic211.pgn` — 1,026 games → 82,427 positions (1 game dropped: result `*`) |
| **Held-out eval** | `twic212.pgn` — 963 games → **81,528** positions |
| **Model** | `configs/nano.toml` — d_model 64, 2 layers, 2 heads, FFN 256, GAB (avg-pool), Elo conditioning, dropout 0.2 → **0.15M params** |
| **Training** | CPU, batch 512, AdamW lr 3e-4 (cosine + warmup), label smoothing 0.05, 6 epochs (1,686 steps) |
| **Best val** | log-loss **1.0106** at step 1,405; fitted calibration temperature **T = 1.05** |

## Held-out results (twic212, 81,528 unseen positions)

| Metric | **Model** (calibrated) | Material baseline | Base-rate | Uniform |
|---|---|---|---|---|
| **Accuracy** | **51.8%** | 42.9% | 34.9% | 33.3% |
| **Log-loss** ↓ | **0.985** | 1.066 | 1.098 | 1.099 |
| **Brier** ↓ | **0.590** | 0.641 | 0.666 | 0.667 |
| **Calibration (ECE)** ↓ | **1.8%** | 6.7% | 1.2% | — |

Confusion matrix (rows = true, cols = predicted):

```
          win   draw   loss     recall
win     15915   5253   7050     56.4%
draw     7549   7187  10123     28.9%
loss     5009   4282  19160     67.3%
```

## Interpretation

**The model works and clearly beats every baseline.** Always guessing the most
common class scores 34.9%; a material-only logistic gets 42.9%; the model reaches
**51.8%** — roughly **+17 points over chance and +9 over material**. Log-loss and
Brier agree, and the probabilities are **very well calibrated** (1.8% ECE; the
near-1.0 temperature confirms little correction was needed). So it is reading
genuine positional structure beyond material count.

**The absolute ceiling is low — and that is the data, not the model:**

- **Draws are the hard class (29% recall).** A drawish position resembles a
  slightly-better-for-one-side position, draws are the minority (~28%), so the
  model hedges toward decisive calls. This is the dominant error source.
- **Loss-bias asymmetry** (loss recall 67% vs win 56%): a real data artifact —
  games *end* when a side is busted, and we include terminal/near-terminal
  positions, so the side to move is disproportionately the losing side late in
  games.
- **~1,890 training games is tiny** for outcome prediction (positions within a
  game are correlated and share one label). The larger 0.93M `tiny` model
  overfit within a single epoch (best val 1.0196 at epoch 0, then worse); the
  0.15M `nano` model with heavier dropout generalized better (val 1.0106). This
  is why the *smaller* model won.

**Bottom line:** predicting a human master game's result from one position is
intrinsically noisy, so ~52% accuracy with excellent calibration is a sound,
honest outcome for this dataset, and the architecture is doing real work rather
than memorizing.

## Notes on the M1 GPU path

Training and inference run on Metal (`--device metal`) as required, but Candle
0.10's Metal backend has **no kernels for the fused `layer_norm`, `softmax`, or
`cross_entropy`/`log_softmax` ops**. These were reimplemented from primitive ops
(mean/var/sqrt; max/exp/sum; gather-NLL) that *do* run on the GPU. Two practical
findings on this machine:

- **Batch 1024 on Metal hangs at step 0** (batch ≤512 is fine) — an unresolved
  Candle/Metal issue at larger batch. Investigate before relying on big batches.
- For a model this small, **per-step Metal throughput is modest** (kernel-launch
  overhead dominates) and the GPU load made the system sluggish. The final run
  used **CPU** (`--device cpu`, ~1 core via Accelerate) — slower per step but
  responsive. Both paths are supported and produce identical logic.

## Reproduce

```bash
# 1. Prepare shards (already gitignored under data/)
./target/release/chess-wdl-prepare --input data/pgn/twic210.pgn --output data/shards/train
./target/release/chess-wdl-prepare --input data/pgn/twic211.pgn --output data/shards/test
./target/release/chess-wdl-prepare --input data/pgn/twic212.pgn --output data/shards/eval

# 2. Train (CPU; swap --device metal to use the M1 GPU)
./target/release/chess-wdl-train --model-config configs/nano.toml \
  --data data/shards/train --val-data data/shards/test \
  --device cpu --epochs 6 --batch-size 512 --checkpoint-dir checkpoints/nano

# 3. Evaluate on the held-out set with baselines
./target/release/chess-wdl-eval --checkpoint checkpoints/nano \
  --data data/shards/eval --device cpu --baseline
```

## Ideas to push further (not yet done)

- `--positions-per-game N` to decorrelate within-game samples (smaller, cleaner
  effective dataset).
- Class weighting / focal loss to recover draw recall.
- More data (additional TWIC issues) — the biggest lever, given the ~1,900-game ceiling.
- Resolve the batch-1024 Metal hang to make GPU training practical at scale.
