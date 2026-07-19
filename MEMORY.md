# MEMORY.md — working notes & handoff

Portable project memory (lives in the repo so it travels between machines).
Use this instead of `~/.claude` auto-memory for this project. Keep it short;
record only what isn't obvious from the code, `CLAUDE.md`, `DESIGN.md`, or git.

## Working preferences

- **Training runs at `nice -n 19`** (lowest priority) so the machine stays responsive.
- **Run the test suite often**, but: `cargo test` includes `tests/overfit.rs`,
  whose 4 memorization tests take **~53 min** (they train on CPU). Use
  `cargo test --lib` (~0.5 s, 26 tests) for normal checks; only run the full
  suite when you've touched `src/fused.rs` or model/gradient code.
- **Commit straight to `main`** (solo research repo; that's the established
  workflow). End commit messages with the
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer. Commit/push
  only when asked.
- **Build is cross-platform (as of 2026-07):** macOS (M1) builds with the Metal
  GPU + Accelerate BLAS; **Linux/other targets build CPU-only** (candle's
  `metal`/`accelerate` features are gated behind a `cfg(target_os = "macos")`
  dep table, and `select_device`'s `new_metal` call is `#[cfg]`-guarded). So on
  the Linux box just `cargo build --release` — everything runs `--device cpu`.
- **Toolchain on the Linux box:** needs **rustup stable** (was 1.96.1); the
  distro's apt `cargo` (1.75) is **too old** for this repo's v4 `Cargo.lock`.
  `source ~/.cargo/env` (or prepend `~/.cargo/bin` to PATH) so cargo/rustc
  resolve to the rustup toolchain, not `/usr/bin/cargo`.

## Data locations (IMPORTANT — most data is NOT in git)

- The repo's `data/pgn/` holds only **twic210/211/212** (3 small files). The
  full TWIC corpus lives outside the repo (won't sync via git; must exist on
  whatever machine you continue on) — path is **machine-dependent**:
  - Linux box (current): **`/home/dave/Projects/ChessData/Twic/`**
  - macOS (M1): `/Users/dave/Projects/ChessData/Twic/`

  It contains `twic*.pgn` — **1440 issues numbered 210–1649** (note: numbers
  are non-contiguous with lexicographic order; sort **numerically** on the digits
  after `twic`) — plus `mega2400.pgn` (3.4 GB, 4.24M games, rated 2400+) and
  `mega2600_part_*.pgn`.
- `data/shards/` (prepared shards) and `checkpoints/` are **gitignored** — they
  do not travel. Regenerate shards with `scripts/regen-data.sh <SRC> <OUT> <GLOB>`
  (e.g. `scripts/regen-data.sh /home/dave/Projects/ChessData/Twic data/shards 'twic9??.pgn'`).
  It builds train first, then stamps test/eval `--seen-against` it.

## Current state (as of 2026-07)

- **Elite Elo≥2400 all-TWIC dataset (`data/shards/`, regenerated
  2026-07-05):** numeric split — **train = issues < 1630** (twic210–1629, 1420
  files), **test = twic1630–1639** (10), **eval = twic1640–1649** (10); test/eval
  stamped `--seen-against` train. Recipe `--require-elo --min-elo 2400 --min-ply
  20 --max-ply 100 --positions-per-game 10`. ~8M train positions, ~560 MB shards,
  draw-heavy (~39%). Not built by `regen-data.sh` (that splits lexicographically
  by whole files, holding out only the last 2); use the numeric-split prep script.
- **`checkpoints/nano-elite2400` (trained 2026-07-05):** `nano`, MKL build,
  batch 1024, lr 5e-4, 1-epoch cosine (7588 steps), ~8.4 h CPU. Best val
  1.0473 (step 5000), fitted **T=1.668** (high → some overfit; best-checkpoint
  selection caught it). Held-out eval (twic1640–1649, 94.6k): **acc 44.5% /
  log-loss 1.046 / Brier 0.631 / ECE 2.4%**, vs material 38.4% / base-rate
  31.9%. Draw-biased (recall win 38 / draw 73 / loss 23). Lower headline than
  elite Run 4 (53.8%) but **not comparable** — this eval is far less draw-heavy
  (~32% vs 48%) and the model is Elo-free (~-2 pts, per Run 5). Well clear of
  baselines and well-calibrated. Checkpoint is gitignored (local only).
  Documented as REPORT.md Run 10.
- **`checkpoints/tiny-elite2400` (trained 2026-07-06, ~2 days CPU/MKL):** `tiny`
  (0.9M), same setup as nano-elite2400 except the config (batch 1024, lr 5e-4,
  1-epoch cosine 7588 steps, ~18 s/step on 4 cores). Best val **1.0177**, T=1.449.
  Held-out eval (twic1640–1649, 94.6k): **acc 47.1% / log-loss 1.022 / Brier
  0.615 / ECE 1.6%** — **beats nano-elite2400 on every metric** (+2.6 pts acc,
  −0.024 log-loss, better-calibrated) and is less draw-biased (recall win 41 /
  draw 68 / loss 33). **First time in the whole run history a bigger model beats
  nano** — because prior tests were ≤0.78M positions (Runs 3/6/7); at 7.77M the
  capacity finally pays off (confirms Run 3's hypothesis). Was still mildly
  UNDER-converged at the horizon (val improving at the last step), so more steps
  would likely widen the gap. Cost ~5.4× nano's wall-clock. Checkpoint gitignored.
  Documented as REPORT.md Run 11. Takeaway: on this elite corpus,
  **more capacity now helps given the data scale** — worth trying `small` if a
  GPU becomes available (infeasible on this 4-core CPU box, ~2 weeks).
- **`checkpoints/tiny-elite2400-2ep` (trained 2026-07-10, ~2.8 days CPU/MKL) —
  CURRENT BEST on the elite set:** same as `tiny-elite2400` but **2 epochs**
  (15176-step cosine, ~18 s/step). Best val **1.0004**, T=1.295; val CONVERGED
  (plateaued at end, unlike the 1-epoch run which was under-converged). Held-out
  (twic1640–1649, 94.6k): **acc 48.3% / log-loss 1.011 / Brier 0.607 / ECE 1.5%**
  — beats 1-epoch tiny (Run 11: 47.1% / 1.022) on every metric (+1.2 pts acc,
  −0.011 log-loss) and is even less draw-biased (recall win 51 / draw 62 / loss
  31; predicts draw only 47% vs Run 11's 53%, nano's 60%). So the 2nd epoch
  helped, modestly (~1.75× cost for +1.2 pts) — diminishing returns, near tiny's
  ceiling on this data. Clean monotonic trend nano→tiny-1ep→tiny-2ep: acc
  44.5→47.1→48.3, T 1.67→1.45→1.30, draw-hedging 60→53→47% (more
  capacity + convergence = sharper, better-calibrated, less-hedging). Checkpoint
  gitignored. Documented as REPORT.md Run 12.
- **Capacity conclusion (Runs 10–12):** on the elite 7.77M corpus the optimal
  model size scales with data — `tiny` (0.9M) > `nano` (0.15M), and 2 epochs >
  1 (diminishing returns). **`tiny-elite2400-2ep` is the best elite model.** Next
  lever is `small` (5M), GPU-only here. General lesson (vs Runs 3/6/7): "nano is
  best" was a small-data artifact; re-test model size whenever the corpus grows.
- **`checkpoints/nano-twic9`** — earlier best on a *different, broad-Elo* eval
  (twic999): acc **48.5%** / log-loss 0.988 (T=0.860), `nano` on twic900–997
  (22.6M positions, ~0.3 epoch). **Not comparable** to the elite checkpoints
  above (different eval set + broad vs 2400-only Elo); kept as a reference point.
  Local/gitignored.
- Eval reports, beyond log-loss/acc/Brier/ECE: a **seen/unseen memorization
  split**, **ply-band** breakdown (buckets of 20), and **expected-score**
  (points) error. Key finding (REPORT Run 8/9): "seen" positions score *worse*
  because shared positions are mostly openings — the split tracks game phase,
  not memorization; ply bands show acc rising 37% (openings) → ~66% (middlegame).
- `chess-wdl-replay` prints per-move WDL (from **White's POV**) + FEN + result +
  most-confident position per game.

## RAM ceiling & the "go big" runs

The pipeline loads *all* samples into RAM (`prepare` builds one `Vec`;
`read_shard_dir` loads everything), so RAM caps the dataset. **The ceiling is
machine-dependent:**
- macOS (M1) box: **16 GB** — full-density mega2400/broad corpus won't fit.
- Linux box (current): **31 GB** (~17 GB free in practice).

**Full-density all-TWIC** (~330–370M positions) needs ~25–27 GB → won't fit
even on the 31 GB box. **But the `--min-elo 2400` filter shrinks it enough to
fit**: only ~17% of TWIC games have both players ≥2400, and with the decorrelated
recipe that's ~8M positions (~600 MB). A `Sample` is ~74 B in RAM.

- **Elo≥2400 all-TWIC (done, 2026-07):** `--require-elo --min-elo 2400
  --min-ply 20 --max-ply 100 --positions-per-game 10` over all 1440 issues →
  ~8M train positions (33/39/28 W/D/L), ~560 MB shards. See the elite-split note
  in Current state.

**`mega2400.pgn` single-file run — still pending** (`~4.24M games`). Even
Elo-implicit (it's already 2400+), full density is ~200M+ positions ≈ 16–24 GB;
tractable on the 31 GB box only with the decorrelated recipe, or after a
streaming/chunked loader rewrite of `prepare` + `read_shard_dir`. Two blockers:
1. **Split the single PGN into train/test/eval by games** — `regen-data.sh` (and
   the numeric-split variant) only split by *whole files*, so they can't handle
   one file. Need a game-level splitter (hold out ~10k games each for test/eval).
2. **Contamination:** mega2400 likely contains TWIC games, so split mega itself
   for eval rather than reusing TWIC-derived held-out issues.

## Perf / budget facts

- `nano` on CPU (Accelerate), batch 512: **~1.1 s/step**. One epoch over 22.6M
  positions ≈ 44k steps (~13 h). Cap `--max-steps` for a wall-clock budget
  (13k steps ≈ 3.8 h). Data load of 22.6M samples ≈ 1 s; final temperature
  calibration over the full val set is a one-time cost.
- Metal: batch ≥ 1024 hangs at step 0 (use ≤ 512); CPU keeps the system
  responsive for these small models.
- **Linux 4-core box (MKL, batch 1024) measured rates:** `nano` ~4 s/step (1
  epoch of 7.77M / 7588 steps ≈ 8.4 h); `tiny` ~18 s/step (1 epoch ≈ ~1.6 days,
  2 epochs/15176 steps ≈ ~2.8 days). Rule of thumb: `tiny` ≈ 5× `nano` per step.
- **MKL only bought ~1.25× over candle's default CPU gemm here** (2.49→1.97
  s/step for nano) — small because the box is 4-core and the models' matmuls are
  tiny. Needed a `hgemm_` link stub (candle 0.10 references the f16 symbol that
  intel-mkl-src 2020.1 lacks; we train f32 so it's dead code). See commit 927ce3d.
- **`small`/larger on CPU is impractical** (~2 weeks/epoch, extrapolated) — those
  need a GPU. `data/*.log` and `data/*-restarts.count` are scratch from the
  detached-training + hourly-loop workflow (all under gitignored `data/`).
