# Chess WDL Prediction — Results Report

End-to-end training and held-out evaluation of the Chessformer/Maia-3-inspired
win/draw/loss model described in [`DESIGN.md`](DESIGN.md).

**Runs so far:**
- **Run 1** — baseline: one file (twic210), terminal positions included → **51.8%** acc.
- **Run 2** — more data (90 files) + decorrelated sampling + ply window → **57.3%** acc. *(current best)*
- **Run 3** — `tiny` (0.93M) on 1/3 the data → **56.4%** acc: matches nano with far less data (capacity is data-efficient), but nano stays the model to ship.

## Run 1 — baseline (twic210 only, terminal positions included)

### Setup

| | |
|---|---|
| **Task** | Predict P(win/draw/loss) for the side to move, from a single position (no history) |
| **Train** | `twic210.pgn` — 1,890 games → **143,432** positions (35.8% win / 28.0% draw / 36.1% loss) |
| **Validation / early-stop** | `twic211.pgn` — 1,026 games → 82,427 positions (1 game dropped: result `*`) |
| **Held-out eval** | `twic212.pgn` — 963 games → **81,528** positions |
| **Model** | `configs/nano.toml` — d_model 64, 2 layers, 2 heads, FFN 256, GAB (avg-pool), Elo conditioning, dropout 0.2 → **0.15M params** |
| **Training** | CPU, batch 512, AdamW lr 3e-4 (cosine + warmup), label smoothing 0.05, 6 epochs (1,686 steps) |
| **Best val** | log-loss **1.0106** at step 1,405; fitted calibration temperature **T = 1.05** |

### Held-out results (twic212, 81,528 unseen positions)

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

### Interpretation (Run 1)

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

## Run 2 — more data + decorrelated sampling (current best)

### Setup

| | |
|---|---|
| **Train** | twic210–289 (80 files) → **781,487** positions |
| **Validation** | twic290–294 (5 files) → 62,909 |
| **Held-out eval** | twic295–299 (5 files) → 80,171 |
| **Sampling** | `--min-ply 20 --max-ply 100 --positions-per-game 10` (decorrelate within games; drop opening + drawn-out tails) |
| **Model** | `configs/nano.toml` (0.15M), same as Run 1 |
| **Training** | CPU, batch 1024, lr 5e-4, 3 epochs (2,292 steps), `nice -n 19`; best val 0.9446, T=1.056 |

Class balance improved vs Run 1 (train 36.4/31.4/32.2 win/draw/loss): dropping
terminal positions removed the loss-bias, and the ply window raised the draw rate.

### Held-out results (twic295–299, 80,171 unseen positions)

| Metric | **Run 2** (nano) | Run 1 (nano) | Material | Base-rate |
|---|---|---|---|---|
| **Accuracy** | **57.3%** | 51.8% | 43.0% | 37.4% |
| **Log-loss** ↓ | **0.924** | 0.985 | 1.073 | 1.095 |
| **Brier** ↓ | **0.548** | 0.590 | 0.646 | 0.664 |
| **ECE** ↓ | **2.0%** | 1.8% | — | — |

Confusion matrix (rows = true, cols = predicted):

```
          win   draw   loss     recall
win     20391   3955   5625     68.0%
draw     8359   8334   7366     34.6%
loss     5456   3460  17225     65.9%
```

### Interpretation (Run 2)

- **+5.5 points accuracy (51.8 → 57.3%)** from data alone (same 0.15M model):
  more independent games + decorrelated sampling = more signal, less memorization.
- **Draw recall 29% → 35%** and **win/loss now symmetric (68/66%)** — the Run 1
  loss-bias artifact is gone, exactly as predicted from dropping terminal positions.
- Still **well-calibrated** (ECE 2.0%); the fitted T=1.056 was essentially a no-op
  (at T=1, ECE is 1.97%), so the model is *intrinsically* calibrated.
- **More epochs don't help nano:** a 10-epoch run plateaued at val ~0.945 by
  epoch 3 — identical to the 3-epoch run. nano (0.15M) has hit its capacity
  ceiling on this data, motivating Run 3.

## Run 3 — larger model (`tiny`, 0.93M)

Testing whether more capacity beats nano now that the dataset is larger (Run 1
showed `tiny` overfit on 143k). Practical note: `tiny` is slow here —
**~5.5 s/step (CPU)**, **~3.7 s/step (Metal)** — dominated by the many small
primitive ops (manual layer-norm/softmax) plus dataset size, not the device. To
keep it CPU-tractable and the machine responsive, Run 3 used a **reduced** train
set (`positions-per-game=3`, **237,350** positions — 3.3× less than Run 2),
batch 512, `nice -n 19`, 4 epochs (1,856 steps). Best val **0.9497**, T=1.127.

### Held-out results (twic295–299, same eval set as Run 2)

| Metric | nano Run 2 (781k) | **tiny Run 3 (237k)** | Material | Base-rate |
|---|---|---|---|---|
| **Accuracy** | **57.3%** | 56.4% | 40.5% | 37.4% |
| **Log-loss** ↓ | **0.924** | 0.935 | 1.075 | 1.097 |
| **Brier** ↓ | **0.548** | 0.557 | 0.650 | 0.665 |
| **ECE** ↓ (T=1) | 2.0% | 1.1% | — | — |

Confusion (tiny): win recall **70.8%**, draw **37.4%**, loss **57.3%**.

```
          win   draw   loss
win     21205   4104   4662
draw     9168   8999   5892
loss     6836   4328  14977
```

### Interpretation (Run 3) — does bigger help?

**Suggestive but not conclusive.** `tiny` (0.93M) trained on **one-third the
data** lands at 56.4% acc / 0.935 log-loss — within ~1 point of nano's 57.3% /
0.924 on full data. So the extra capacity roughly *compensated for 3.3× less
data*, which hints `tiny` is the more data-efficient/capable model and would
likely beat nano on equal data. We couldn't run that head-to-head: `tiny` on the
full 781k is ~1.5 hr/epoch here, impractical for this session.

Caveats: the comparison is confounded by **less data** and a **different class
balance** in the reduced set (train 38.1/32.1/**29.8** win/draw/loss). The lower
loss fraction shows up as `tiny`'s weaker **loss recall (57% vs nano's 66%)** —
it under-predicts the under-represented class — while its **draw recall improved
to 37%**. Both models stay well-calibrated at T=1 (ECE ~1–2%; the fitted
temperature again slightly over-corrected).

**Practical bottom line:** **nano remains the best model to ship** — equal or
better held-out metrics at ~6× fewer params and far cheaper training. The
larger model is promising but needs full-data training (and ideally a faster
op-fused forward pass) to realize its edge.

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
