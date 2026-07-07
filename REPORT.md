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
- **Run 9** — scale-up + ply bands: **22.6M** train positions (twic900–999), nano capped at 13k steps (~0.3 epoch, ~3.8 h) → **48.5%** acc / 0.988 log-loss. New per-ply-band metrics show accuracy climbing **37% (openings) → ~66% (deep middlegame)**, which *explains* the Run-8 inversion: "seen" ≈ early-ply ≈ intrinsically hard, not memorized (see §Run 9).
- **Run 10** — elite Elo≥2400 across **all 1440 TWIC issues**, nano trained **to convergence** (1-epoch cosine, batch 1024, ~8.4 h on a 4-core Linux/MKL box) → **44.5%** acc / 1.046 log-loss on a fresh, recent held-out set (twic1640–1649). Beats material (38.4%) and base-rate (31.9%) and is well-calibrated (ECE 2.4%), but draw-biased; the lower headline vs Run 4 is a less-draw-heavy eval (32% vs 48%) + Elo-free, not a regression (see §Run 10).
- **Run 11** — **`tiny` (0.9M) to convergence on the same 7.77M elite set** (same recipe as Run 10 bar the config; ~2 days CPU) → **47.1%** acc / **1.022** log-loss, **beating nano (Run 10) on every held-out metric** (+2.6 pts acc, −0.024 log-loss, ECE 1.6% vs 2.4%) and less draw-biased. **First time in the run history a bigger model beats nano** — prior tests (Runs 3/6/7) were ≤0.78M positions; at 7.77M the capacity finally pays off, confirming Run 3 (see §Run 11).

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

## Run 9 — scale-up (22.6M positions) + per-ply-band metrics

Two additions: (1) every position now stores its **ply** (half-move index in the
game), and `chess-wdl-eval` breaks metrics into **ply bands of 20**; (2) a much
larger corpus — TWIC issues **twic900–999** (100 files, 226 MB PGN) — to give the
memorization probe (Run 8) real shared-position depth.

### Setup

| | |
|---|---|
| **Train** | twic900–997 (98 files) → **22,608,380** positions (35.1% win / 29.4% draw / 35.4% loss); 16.9M unique |
| **Test / early-stop** | twic998 → 198,278 — **31.9%** also in train |
| **Held-out eval** | twic999 → 317,331 — **23.6%** also in train |
| **Model** | `configs/nano.toml` (0.15M, board-only / Elo-free) |
| **Training** | CPU, `nice -n 19`, batch 512, **`--max-steps 13000`** (~0.3 of one 44k-step epoch — deliberately capped for a <6 h budget at ~1.1 s/step), val every 500. **~3 h 48 m**; best val **0.9767**, T=**0.860** |

Reproduce the data with `./scripts/regen-data.sh /path/to/Twic data/shards 'twic9??.pgn'`
(builds train first, then stamps test/eval `--seen-against` it).

### Held-out results (twic999, 317,331 positions)

| Metric | **Model** (cal.) | Material | Base-rate |
|---|---|---|---|
| **Accuracy** | **48.5%** | 44.7% | 36.5% |
| **Log-loss** ↓ | **0.988** | 1.055 | 1.091 |
| **Brier** ↓ | **0.594** | 0.632 | 0.662 |
| **ECE** ↓ | **1.0%** | 4.9% | 1.0% |

**Memorization split:**

| Subset | n | Accuracy | Log-loss | ECE |
|---|---|---|---|---|
| **seen** in train (23.6%) | 75,015 | **37.0%** | 1.089 | 0.4% |
| **unseen** (novel) | 242,316 | **52.1%** | 0.957 | 1.4% |

**Ply bands (game phase):**

| Ply band | n | Accuracy | Log-loss |
|---|---|---|---|
| 0–19 | 76,234 | 36.8% | 1.090 |
| 20–39 | 74,268 | 41.1% | 1.076 |
| 40–59 | 65,634 | 50.8% | 0.995 |
| 60–79 | 47,485 | 58.5% | 0.895 |
| 80–99 | 27,589 | 62.3% | 0.822 |
| 100–119 | 14,449 | 65.6% | 0.770 |
| 120–139 | 6,654 | 66.4% | 0.755 |
| 140–159 | 3,030 | 63.0% | 0.794 |
| 160+ | 1,988 | ~63% | ~0.72 |

### Interpretation (Run 9)

- **Scale helps a lot.** Same 0.15M model, board-only: 42.5% (single issue, Run 8)
  → **48.5%** on the 22.6M-position corpus, with strong calibration (ECE 1.0%,
  T=0.86). And this is only ~0.3 of one epoch — the curve was still gently
  improving at the 13k-step cap, so more compute would buy more.
- **The ply bands explain Run 8's "memorization inversion."** Accuracy rises
  almost monotonically with game phase: **openings (ply 0–19) sit at base-rate
  (~37%, log-loss ~1.09)** — the result is genuinely unreadable that early — while
  deep middlegames (ply 100–139) reach **~66%**. Because the positions shared
  between TWIC issues are overwhelmingly **openings**, "seen" ≈ low-ply ≈
  intrinsically hard. So the seen-vs-unseen gap (37% vs 52%) is a **game-phase
  artifact, not memorization** — exactly what the per-ply view makes visible.
- Practical reading: the model has essentially **no opening signal** (as expected
  from a single position with no move history) and most of its skill is reading
  **resolved, material/king-safety-laden later positions**. A ply window
  (`--min-ply`) would raise headline accuracy by simply dropping the hard early
  band, as Run 2 did.

## Run 10 — elite Elo≥2400 across all TWIC, nano trained to convergence

First run on a **second machine** (a 4-core Linux box, 31 GB) rather than the M1.
Two firsts: (1) the elite Elo≥2400 filter applied across the **entire** TWIC
collection (all 1440 issues, 210–1649), and (2) nano trained for a **full epoch
to convergence** (prior elite runs were small or capped).

### Setup

| | |
|---|---|
| **Filter / sampling** | `--require-elo --min-elo 2400 --min-ply 20 --max-ply 100 --positions-per-game 10` (no upper Elo cap, unlike Run 4's 2899) |
| **Split** | by **numeric** issue number: train = issues < 1630 (twic210–1629, 1420 files); test = twic1630–1639 (10); eval = twic1640–1649 (10); test/eval `--seen-against` train |
| **Train** | **7,769,830** positions (33.1% win / 38.9% draw / 27.7% loss); 6.96M unique |
| **Test / early-stop** | twic1630–1639 → 98,187 (7.2% seen in train) |
| **Held-out eval** | twic1640–1649 → **94,567** (7.4% seen; **less draw-heavy: 37/32/31**) |
| **Model** | `configs/nano.toml` (0.15M, board-only / Elo-free) |
| **Training** | CPU (Intel MKL BLAS), `nice -19`, batch **1024**, lr **5e-4**, 1-epoch cosine = **7,588 steps**, val every 1000, `--early-stop-patience 10`. **~8.4 h**; best val **1.0473** (step 5000), fitted **T = 1.668** |

Build note: this box has no Apple Accelerate, so candle's `mkl` feature supplies
multi-threaded CPU BLAS (~1.25× faster steps here — modest on 4 cores / a 0.15M
model). Metal/Accelerate are now macOS-only via `cfg(target_os = "macos")`.

### Held-out results (twic1640–1649, 94,567 positions)

| Metric | **Model** (cal., T=1.67) | Material | Base-rate |
|---|---|---|---|
| **Accuracy** | **44.5%** | 38.4% | 31.9% |
| **Log-loss** ↓ | **1.046** | 1.089 | 1.106 |
| **Brier** ↓ | **0.631** | 0.660 | 0.672 |
| **ECE** ↓ | **2.4%** | 2.1% | 7.2% |
| **Score MAE** ↓ | **0.336** | 0.344 | 0.347 |

Confusion (rows = true; recall win **37.9%** / draw **72.5%** / loss **23.4%**):

```
          win   draw   loss
win     13372  18548   3361
draw     5888  21893   2426
loss     5840  16430   6809
```

Ply bands (calibrated): **38.1%** (20–39) → 45.7% (40–59) → 50.0% (60–79) →
**54.3%** (80–99). Seen/unseen (7.4% seen): **46.2% vs 44.4%** — no memorization
gap. Expected-score bias −0.008 (near-unbiased); after the T=1.67 softening the
model is slightly *under*-confident in the high-E bins (E∈[0.7,0.8]: pred 0.74 vs
realized 0.84).

### Interpretation (Run 10)

- **Clears every baseline** (+6.1 pts acc over material, +12.5 over base-rate;
  lower log-loss/Brier) and stays **well-calibrated** (ECE 2.4%). It reads real
  positional signal, and scale + convergence give a much steadier eval than the
  ~15k-position Run-4 elite set (this eval is ~95k).
- **The 44.5% headline is lower than Run 4's 53.8%, but not a regression** — it's
  a harder, fairer comparison. Two mechanical reasons: (a) this held-out set
  (recent 2024-era elite games, twic1640–1649) is **far less draw-heavy** (32%
  draws vs Run 4's 48%), which lowers the achievable accuracy for a draw-hedging
  model — "always draw" scores only 31.9% here vs ~48% there; and (b) the model
  is **Elo-free** (Run 5 showed that alone costs ~2 pts on TWIC).
- **Draw bias persists** (predicts draw for ~60% of positions; loss recall just
  23%), the same elite-distribution hedge as Run 4 — draws dominate elite games
  and a single quiet position rarely reveals the eventual decisive result.
- **Signs of mild overfit at convergence:** train loss fell to ~1.01 while best
  val was 1.047 (step 5000, then drifted up to ~1.055), and calibration needed a
  large **T = 1.668**. One epoch of 7.77M positions at lr 5e-4 slightly overshot;
  best-checkpoint selection + temperature scaling recovered a calibrated model.
  A shorter horizon or lighter LR would likely land the same val with less
  overconfidence.
- **Takeaway:** on the elite distribution the ceiling is the **draw dominance**,
  not the model. Levers to try next: `--draw-weighting` (trade calibration for
  balanced win/loss recall, per Run 5b), and — the recurring theme — testing a
  **larger model** now that there is finally enough elite data (7.77M) to feed it.

## Run 11 — `tiny` (0.9M) to convergence: capacity finally beats nano

The Run-10 takeaway ("try a larger model now that data is 7.77M") tested directly.
Every prior head-to-head — Run 3 (`tiny` on ⅓ data), Run 6 (`tiny` under-trained),
Run 7 (`tiny` converged, ~36 h) — found `tiny` **≤** nano, but all on ≤0.78M
positions. This run gives `tiny` the full elite corpus and a clean apples-to-apples
setup against Run 10.

### Setup

Identical to Run 10 **except the model config** (so the comparison isolates
capacity): same `data/shards/` (7.77M train / 98k test / 94.6k eval), same
batch 1024, lr 5e-4, 1-epoch cosine (7,588 steps), val every 1000, MKL build,
`nice -19`.

| | |
|---|---|
| **Model** | `configs/tiny.toml` — d_model 128, 4 layers, 4 heads, FFN 512 → **0.9M** (6× nano), board-only / Elo-free |
| **Training** | CPU (MKL), **~18 s/step** on 4 cores (~5× nano's 3.8 s), **~2 days** wall-clock. Best val **1.0177**, fitted **T = 1.449** |
| **Convergence** | val fell 1.065 → 1.026 (step 3k) → 1.020 (4k) → **1.018 (7.6k)**, **still improving at the horizon** — mildly *under*-converged (as expected: `tiny` optimizes slower than nano, cf. Run 6/7) |

### Held-out results (twic1640–1649, 94,567 positions) — vs nano Run 10

| Metric | **`tiny`** (cal., T=1.45) | nano Run 10 (T=1.67) | Material | Base-rate |
|---|---|---|---|---|
| **Accuracy** | **47.1%** | 44.5% | 38.4% | 31.9% |
| **Log-loss** ↓ | **1.022** | 1.046 | 1.089 | 1.106 |
| **Brier** ↓ | **0.615** | 0.631 | 0.660 | 0.672 |
| **ECE** ↓ | **1.6%** | 2.4% | 2.1% | 7.2% |
| **Score MAE** ↓ | **0.330** | 0.336 | 0.344 | 0.347 |

Confusion (rows = true; recall win **41.4%** / draw **67.9%** / loss **32.5%**):

```
          win   draw   loss
win     14602  16280   4399
draw     6048  20517   3642
loss     5917  13713   9449
```

Ply bands (calibrated): **39.7%** (20–39) → 49.6% (40–59) → 53.5% (60–79) →
**55.7%** (80–99). Seen/unseen (7.4% seen): **46.5% vs 47.2%** — no memorization
gap. Expected-score bias −0.013.

### Interpretation (Run 11) — the first time bigger wins

- **`tiny` beats nano on every held-out metric** (+2.6 pts acc, −0.024 log-loss,
  lower Brier, lower ECE), and against nano's identical setup — so this is a clean
  **capacity** win, not a data or hyperparameter artifact. Ply bands improve
  uniformly (nano 38.1→54.3 vs `tiny` 39.7→55.7).
- **Why now, when Runs 3/6/7 said "bigger isn't worth it"?** Those verdicts were
  all at **≤0.78M positions**, where `tiny` overfit (Run 3) or couldn't converge
  in budget (Runs 6/7). At **7.77M** the extra capacity finally has the data to
  earn its keep — exactly the data-efficiency hypothesis Run 3 raised. The lesson
  isn't "nano vs tiny"; it's that the **optimal model size scales with data**, and
  the previous elite corpora were too small to see it.
- **`tiny` is also *better* calibrated** (T 1.45 vs 1.67, ECE 1.6% vs 2.4%) and
  **less draw-biased** (predicts draw for 53% of positions vs nano's 60%; loss
  recall 33% vs 23%). More capacity here reduces the draw-hedging, the opposite of
  the usual "bigger overfits more" worry — because the bottleneck was
  representational, not data-fitting.
- **Still under-converged:** val was falling at the 7,588-step horizon, so a longer
  run (2 epochs) or lower final LR would likely widen the gap further — this is a
  *lower bound* on `tiny`'s edge, not its ceiling.
- **Cost:** ~2 days vs nano's 8.4 h — **~5.4×** the wall-clock for +2.6 pts /
  −0.024 log-loss. A real, clean win, but an expensive one on a 4-core CPU;
  the natural next step (`small`, 5M) is ~2 weeks here and wants a GPU.

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

## Expected-score (points) metric

Alongside log-loss/accuracy/Brier/ECE, `chess-wdl-eval` reports an
**expected-score** error: collapse the WDL prediction to expected points
`E = P(win) + 0.5·P(draw)` and compare to the realized game points
(win=1, draw=0.5, loss=0), side-to-move relative. It prints `score_mae` /
`score_rmse` on every metrics line (so it flows into the seen/unseen and
ply-band splits too), a signed `bias`, and a 10-bin reliability table.

On the Run-9 model / eval set (twic999): **MAE 0.333, RMSE 0.393, bias +0.005**
(essentially unbiased), and the reliability table tracks the diagonal closely
(e.g. E∈[0.9,1.0]: predicted 0.935 vs realized 0.931). Consistent with the phase
story, point estimates are least accurate in the openings (MAE ~0.368) and
sharpest in resolved later positions (~0.324).

## Inspecting a game move-by-move

`chess-wdl-replay` runs the model over every position of every game in a PGN and
prints, per move, the WDL prediction from the side-to-move's perspective, plus
each game's result and its single most confident position.

```bash
./target/release/chess-wdl-replay --checkpoint checkpoints/nano-twic9 \
  --pgn data/pgn/twic212.pgn --device cpu --max-games 1
```

```
=== Game 1 : Kasparov,G vs Kramnik,V  [result 1/2-1/2] ===
  ply  move        stm    win  draw  loss  pred  conf
    0  1. e4        w   0.364 0.297 0.339  win  0.364
   ...
   45  23... bxc6   b   0.101 0.140 0.759  loss 0.759
 most confident: 40... Kf4 (ply 79)  conf 0.872 -> win (Black)  | game result 1/2-1/2
```

Opening rows hug the base rate (~0.36/0.30/0.34) and sharpen as the position
resolves — the same game-phase effect quantified by the ply bands in Run 9.

## Ideas to push further (not yet done)

- `--positions-per-game N` to decorrelate within-game samples (smaller, cleaner
  effective dataset).
- Class weighting / focal loss to recover draw recall.
- More data (additional TWIC issues) — the biggest lever, given the ~1,900-game ceiling.
- Resolve the batch-1024 Metal hang to make GPU training practical at scale.
