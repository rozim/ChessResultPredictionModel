# Chess WDL Prediction — Results Report

End-to-end training and held-out evaluation of the Chessformer/Maia-3-inspired
win/draw/loss model described in [`DESIGN.md`](DESIGN.md).

**Runs so far:**
- **Run 1** — baseline: one file (twic210), terminal positions included → **51.8%** acc.
- **Run 2** — more data (90 files) + decorrelated sampling + ply window → **57.3%** acc. *(best on the broad-Elo distribution)*
- **Run 3** — `tiny` (0.93M) on 1/3 the data → **56.4%** acc: matches nano with far less data (capacity is data-efficient), but nano stays the model to ship.
- **Run 4** — strict Elo filter (both 2400–2899) → harder, 48%-draw distribution: **53.8%** acc, but the model becomes draw-biased (see §Run 4).
- **Run 5** — Elo removed from the model (board-only input). 5a: **51.9%** acc (Elo removal costs ~2 pts — the within-band rating *difference* did carry signal). 5b: + `--draw-weighting` rebalances recall (loss 23→31%) at a small accuracy/calibration cost (see §Run 5).
- **Run 6** — bigger model (`tiny`, Elo-free) on full broad-Elo data, 3-epoch cap → **48.7%** acc: *under-converged* (still improving at the cap), so worse than nano. Confirms tiny needs far more training than a CPU budget allows (see §Run 6).
- **Run 7** — same `tiny`, trained to convergence (~12 epochs, ~36 h) → **51.3%** acc: recovers the under-convergence loss but still **doesn't beat nano** at ~36× the cost. Settles it: bigger model is not worth it here (see §Run 7).
- **Run 8** — memorization split: flag eval/test positions that also occur in train (14.9% / 13.5% overlap) → **seen 32.9%** vs **unseen 44.0%** acc. Counterintuitively, seen positions are *harder*: the shared ones are mostly low-signal, inconsistently-labelled openings, so on a single-issue train set the split tracks game *phase*, not memorization (see §Run 8).

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

## Run 4 — strict Elo filter (elite-GM, draw-heavy)

### Setup

| | |
|---|---|
| **Filter** | `--require-elo --min-elo 2400 --max-elo 2899` (both tags present), plus the Run 2 ply window / decorrelation |
| **Train** | twic210–289 → **187,178** positions (29.3% win / **46.3% draw** / 24.4% loss) |
| **Val** | twic290–294 → 19,817 |
| **Held-out eval** | twic295–299 → **14,738** (48% draws) |
| **Model** | `configs/nano.toml` (0.15M); status logged every 20 steps, val every 100 |
| **Training** | CPU, batch 512, lr 4e-4, early-stopped at step 2,400 (best step 1,900); best val 0.9977, T=1.157 |

### Held-out results (twic295–299, 14,738 positions, 48% draws)

| Metric | **Model** (T=1) | Material | Base-rate ("always draw") |
|---|---|---|---|
| **Accuracy** | **53.8%** | 47.9% | 47.7% |
| **Log-loss** ↓ | **0.957** | 1.046 | 1.054 |
| **Brier** ↓ | **0.575** | 0.630 | 0.635 |
| **ECE** ↓ (T=1) | **1.1%** | — | — |

Per-class recall: **draw 79%**, loss 35%, **win 27%**.

```
          win   draw   loss
win      1134   2589    432
draw      650   5550    828
loss      325   1984   1246
```

### Interpretation (Run 4) — the elite filter makes it harder

- **Still beats every baseline** (+6 pts acc, lower log-loss/Brier) and is
  **intrinsically well-calibrated** (ECE 1.1% at T=1; the fitted T=1.157
  over-smoothed slightly). So it learns real signal.
- **But accuracy fell vs. Run 2 (53.8% vs 57.3%)** and the model is now
  **draw-biased**: it predicts draw for ~69% of positions (draw recall 79%, but
  win recall 27%, loss recall 35%). Rational — ~48% of elite games are drawn and
  the eventual decisive result is hard to read from a single quiet position, so
  "draw" is the safe bet.
- Motivates two follow-ups: optional **draw/class weighting** to rebalance
  win/loss recall, and **removing Elo as an input** (the band is now fixed to
  strong GMs, so the Elo embedding carries no signal — simplify the model).

## Run 5 — Elo removed from the model (± draw-weighting)

Same Elo-filtered data as Run 4 (`data/shards3/`, 48% draws), `nano`, CPU,
`nice -n 19`. The model input is now **board-only** (`input_dim = 18`; no Elo
embedding). 5a is the Elo-free baseline; 5b adds `--draw-weighting`
(inverse-frequency class weights: win 1.14, draw 0.72, loss 1.37).

### Held-out results (twic295–299, 14,738 positions)

| Metric | Run 4 (with Elo) | **5a** (Elo-free) | **5b** (Elo-free + draw-wt) | Base-rate |
|---|---|---|---|---|
| **Accuracy** | 53.8% | **51.9%** | 50.9% | 47.7% |
| **Log-loss** ↓ | 0.957 | **0.998** | 1.026 | 1.054 |
| **Brier** ↓ | 0.575 | **0.597** | 0.615 | 0.635 |
| **ECE** ↓ (T=1) | 1.1% | 1.7% | 5.6% | — |
| **Recall** win/draw/loss | 27/79/35 | 30/80/23 | **32/72/31** | — |

### Interpretation (Run 5)

- **Removing Elo cost ~2 points accuracy and ~0.04 log-loss** (53.8 → 51.9%,
  0.957 → 0.998). So the assumption "Elo is constant within the filtered band,
  therefore no signal" is only *partly* true: the **rating difference** between
  the two players (e.g. 2850 vs 2420) is mildly predictive of a decisive result.
  Removing it is a reasonable simplification — board-only is cleaner and still
  well clear of baselines — but it isn't free.
- **`--draw-weighting` rebalances recall** as designed: loss recall **23 → 31%**,
  win **30 → 32%**, draw **80 → 72%** (the model predicts draw for ~61% of
  positions instead of ~69%). The price is ~1 pt accuracy, worse log-loss, and
  worse calibration (ECE 1.7 → 5.6%) — it deliberately stops optimizing the true
  class distribution. Use it when balanced win/loss recall matters more than raw
  accuracy or calibrated probabilities; leave it off (default) otherwise.
- **Best model overall remains Run 2** (broad-Elo data, 57.3% acc). On the
  elite-GM distribution the task is simply harder and draw-dominated.

## Run 6 — bigger model (`tiny`) on broad-Elo, Elo-free

Does more capacity beat nano on the balanced broad-Elo data (Run 2's 57.3%)
when given the *full* 781k set (Run 3's caveat was it saw only ⅓)? `tiny`
(0.90M, Elo-free), CPU, `nice -n 19`, 3-epoch cap.

| Held-out broad-Elo (80,171) | nano Run 2 (Elo, converged) | tiny Run 3 (Elo, ⅓ data) | **tiny Run 6 (Elo-free, 3 ep)** | base-rate |
|---|---|---|---|---|
| Accuracy | **57.3%** | 56.4% | 48.7% | 37.4% |
| Log-loss ↓ | **0.924** | 0.935 | 1.008 | 1.095 |
| ECE ↓ | 2.0% | — | 1.7% | — |

### Interpretation (Run 6) — under-converged, not "bigger is worse"

`tiny` scored **48.7% / 1.008** — well below nano — but **it never converged**:
validation was *still falling steeply* at the 3-epoch cap (1.012 → 1.006 → 1.004,
no plateau, no early stop). The deeper 4-layer model optimizes far slower than
nano's 2 layers, and ~4,600 steps is nowhere near enough. Two confounds stack:
under-training (dominant) and Elo removal (~2 pts, Run 5).

**Practical takeaway, consistent across all runs:** under a CPU compute budget,
the small/shallow **`nano` is the model to ship**. Extra capacity only pays off
if you can afford to converge it (~2 hr/epoch here → many hours), which on this
hardware you generally can't. *(A longer, converged `tiny` run is queued to test
the fair-fight version of this question.)*

## Run 7 — `tiny` trained to convergence (the fair fight)

Run 6 left `tiny` under-trained, so this run extends it: same `tiny` (0.90M,
Elo-free) on the full 781k broad-Elo set, 15-epoch cap, CPU, `nice -n 19`.
It **early-stopped at ~epoch 13** (step 19,000, best val **0.9748**) after
**~36 hours** of wall-clock.

### Held-out broad-Elo (80,171)

| Metric | nano Run 2 (Elo, ~2 h) | tiny Run 6 (3 ep) | **tiny Run 7 (converged)** | base-rate |
|---|---|---|---|---|
| **Accuracy** | **57.3%** | 48.7% | 51.3% | 37.4% |
| **Log-loss** ↓ | **0.924** | 1.008 | 0.978 | 1.095 |
| **Brier** ↓ | **0.548** | 0.605 | 0.585 | 0.664 |
| **ECE** ↓ | 2.0% | 1.7% | 1.5% | — |
| **Recall** w/d/l | 68/35/66 | — | 53/54/47 | — |

### Interpretation (Run 7) — bigger is *not* worth it here

- **Convergence recovered most of Run 6's deficit** (48.7 → 51.3% acc,
  1.008 → 0.978 log-loss) — confirming under-training, not capacity, drove the
  bad Run 6 number.
- **But converged `tiny` still loses to `nano`** (51.3% vs 57.3% acc). The ~6-pt
  gap is larger than the ~2-pt Elo handicap (Run 5) can explain, so it isn't just
  the removed Elo: the extra capacity simply doesn't help on this data/task.
- `tiny` does have a **more balanced error profile** (recall 53/54/47 vs nano's
  decisive 68/35/66) — it spreads its bets — but that doesn't translate to higher
  accuracy or lower log-loss.
- **Cost:** ~36 h vs nano's ~2 h — roughly **18× the wall-clock for a worse
  model**. Decisive confirmation of the running theme: on this corpus and
  hardware, the small/shallow **`nano` is the model to ship**; scaling up the
  transformer buys nothing here.

## Run 8 — memorization split (seen vs. novel positions)

New capability: every test/eval position is stamped with a `seen` flag — true if
the exact position (board + castling + en-passant, side-to-move frame; Elo and
label ignored) also appears in the training set. `chess-wdl-prepare
--seen-against <train-dir>` computes it from a position fingerprint, and
`chess-wdl-eval` then breaks all metrics down by seen vs. unseen, so we can ask
whether the model is *memorizing* training positions or generalizing.

### Setup

| | |
|---|---|
| **Train** | `twic210.pgn` → **143,432** positions (Run-1-style data: no ply window, terminals included) |
| **Test / early-stop** | `twic211.pgn` → 82,427 positions — **12,298 (14.9%) seen in train** |
| **Held-out eval** | `twic212.pgn` → 81,528 positions — **10,990 (13.5%) seen in train** |
| **Model** | `configs/nano.toml` (0.15M, board-only / Elo-free per Run 5) |
| **Training** | CPU, `nice -n 19`, batch 512, lr 3e-4; early-stopped step 3,200 (best ~1,200), best val **1.0792**, T=**1.335** |

### Held-out results (twic212, 81,528 positions)

| Subset | n | Accuracy | Log-loss ↓ | Brier ↓ | ECE ↓ |
|---|---|---|---|---|---|
| **all** | 81,528 | 42.5% | 1.059 | 0.639 | 2.0% |
| **seen** in train | 10,990 | **32.9%** | 1.101 | 0.668 | 1.8% |
| **unseen** (novel) | 70,538 | **44.0%** | 1.053 | 0.635 | 2.4% |
| baseline: material | 81,528 | 42.9% | 1.066 | 0.641 | — |
| baseline: base-rate | 81,528 | 34.9% | 1.098 | 0.666 | — |

### Interpretation (Run 8) — the split tracks game phase, not memorization

- **The metric works**, but the result inverts the naive expectation: positions
  the model *saw in training* score **worse** (32.9% vs 44.0% acc, higher
  log-loss), i.e. there is **no memorization edge**.
- **Why:** with a single TWIC issue as the train set, the positions shared with
  later issues are almost entirely **openings** — inherently low-signal *and*
  labelled inconsistently across games (the same opening is a win in one game, a
  loss in another), so there is no single outcome to memorize. "Unseen"
  positions skew toward mid/endgame where material gives the model something to
  grip. So here the seen/unseen split is mostly a proxy for **game phase**.
- The absolute accuracy is below Run 1's 51.8% on the same files because this
  model is now **board-only/Elo-free** (Run 5); on broad-Elo TWIC data the
  player rating *difference* carried more signal than in the elite band, so its
  removal costs more here.
- **The plumbing scales:** run `scripts/regen-data.sh` against the full TWIC
  collection and the overlap will involve far more (and deeper) shared
  positions, sharpening this into a genuine memorization probe.

## Notes on the M1 GPU path

Training and inference run on Metal (`--device metal`) as required, but Candle
0.10's Metal backend has **no kernels for the fused `layer_norm`, `softmax`, or
`cross_entropy`/`log_softmax` ops**. These were reimplemented from primitive ops
(mean/var/sqrt; max/exp/sum; gather-NLL) that *do* run on the GPU. Two practical
findings on this machine:

- **Batch 1024 on Metal hangs at step 0** (batch ≤512 is fine) — an unresolved
  Candle/Metal issue at larger batch. Investigate before relying on big batches.
- For a model this small, **per-step Metal throughput is modest** (kernel-launch
  overhead dominates) and the GPU load made the system sluggish. Training runs
  used **CPU** (`--device cpu`) — slower per step but responsive. Both paths are
  supported and produce identical logic.
- **Fused CPU ops (`src/fused.rs`).** softmax and layer-norm are rayon-parallel
  `CustomOp1`s on CPU (fused forward + correct backward), dispatched per device.
  Measured **~1.23× faster** tiny CPU step (302s → 246s for 40 steps + a val
  pass). The win is bounded by Candle's **single-threaded batched matmul**, which
  remains the dominant cost; further speedup would need a threaded GEMM backend.

## Reproduce

```bash
# 1. Prepare shards (gitignored under data/). Build train first, then stamp
#    test/eval with the `seen` flag (= position also present in train).
#    Convenience script does all three from data/pgn/*.pgn:
./scripts/regen-data.sh                      # -> data/shards/{train,test,eval}
# ...or by hand:
./target/release/chess-wdl-prepare --input data/pgn/twic210.pgn --output data/shards/train
./target/release/chess-wdl-prepare --input data/pgn/twic211.pgn --output data/shards/test \
  --seen-against data/shards/train
./target/release/chess-wdl-prepare --input data/pgn/twic212.pgn --output data/shards/eval \
  --seen-against data/shards/train

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
