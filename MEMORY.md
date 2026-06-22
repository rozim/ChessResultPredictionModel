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

## Data locations (IMPORTANT — most data is NOT in git)

- The repo's `data/pgn/` holds only **twic210/211/212** (3 small files). The
  full TWIC corpus is at **`/Users/dave/Projects/ChessData/Twic/`** —
  `twic*.pgn` (1440 issues) plus `mega2400.pgn` (3.4 GB, 4.24M games, rated 2400+).
  **This directory is outside the repo and won't sync via git** — it must exist
  on whatever machine you continue on.
- `data/shards/` (prepared shards) and `checkpoints/` are **gitignored** — they
  do not travel. Regenerate shards with `scripts/regen-data.sh <SRC> <OUT> <GLOB>`
  (e.g. `scripts/regen-data.sh /Users/dave/Projects/ChessData/Twic data/shards 'twic9??.pgn'`).
  It builds train first, then stamps test/eval `--seen-against` it.

## Current state (as of 2026-06)

- Best model so far: **`checkpoints/nano-twic9`** — `nano` trained on twic900–997
  (22.6M positions), 13k-step cap (~0.3 epoch), ~3 h 48 m, best val 0.9767,
  T=0.860; held-out (twic999) acc **48.5%**, log-loss 0.988. Checkpoint is
  local/gitignored — retrain on the other machine if needed.
- Eval reports, beyond log-loss/acc/Brier/ECE: a **seen/unseen memorization
  split**, **ply-band** breakdown (buckets of 20), and **expected-score**
  (points) error. Key finding (REPORT Run 8/9): "seen" positions score *worse*
  because shared positions are mostly openings — the split tracks game phase,
  not memorization; ply bands show acc rising 37% (openings) → ~66% (middlegame).
- `chess-wdl-replay` prints per-move WDL (from **White's POV**) + FEN + result +
  most-confident position per game.

## Pending: the "go big" mega2400 run (blocked by RAM here)

Goal: train on `mega2400.pgn` (~4.24M games). **Blocked on this machine: only
16 GB RAM**, and the pipeline loads *all* samples into RAM (`prepare` builds one
`Vec`; `read_shard_dir` loads everything). Full density (~200M+ positions ≈
16–24 GB) won't fit. To do on a bigger-RAM machine:

1. **Split the single PGN into train/test/eval by games** — `regen-data.sh` only
   splits by *whole files*, so it can't handle one file. Need a game-level
   splitter (route games to 3 sets; hold out ~10k games each for test/eval).
2. **Sampling:** if RAM-limited, use the Run-2 decorrelated recipe
   (`--min-ply 20 --max-ply 100 --positions-per-game 10` → ~40M positions, also
   the best-performing recipe). With ample RAM, full density is possible but may
   want a streaming/chunked loader rewrite of `prepare` + `read_shard_dir`.
3. Watch contamination: mega2400 likely contains TWIC games, so don't reuse
   twic998/999 as eval — split mega itself.

## Perf / budget facts

- `nano` on CPU (Accelerate), batch 512: **~1.1 s/step**. One epoch over 22.6M
  positions ≈ 44k steps (~13 h). Cap `--max-steps` for a wall-clock budget
  (13k steps ≈ 3.8 h). Data load of 22.6M samples ≈ 1 s; final temperature
  calibration over the full val set is a one-time cost.
- Metal: batch ≥ 1024 hangs at step 0 (use ≤ 512); CPU keeps the system
  responsive for these small models.
