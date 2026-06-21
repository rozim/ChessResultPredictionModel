# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Rust implementation of a Chessformer/Maia-3-inspired model that predicts
**win/draw/loss** probabilities for the side to move from a single chess position
(no move history). Full design and rationale: [`DESIGN.md`](DESIGN.md). First
training run and held-out results: [`REPORT.md`](REPORT.md).

## Commands

```bash
cargo build --release        # build lib + 5 binaries (release; needed for training speed)
cargo test --lib             # run the unit-test suite (CPU; ~20 tests)
cargo test <name>            # run a single test
cargo clippy                 # lint

# Pipeline (binaries land in target/release/)
chess-wdl-prepare --input <pgn>... --output <shard-dir>      # PGN -> shards
chess-wdl-train --model-config configs/<size>.toml \         # train (default --device metal)
    --data <train-shards> --val-data <test-shards> --device cpu --checkpoint-dir <dir>
chess-wdl-eval --checkpoint <dir> --data <eval-shards> --baseline   # metrics vs baselines
chess-wdl-predict --checkpoint <dir> --fen "<FEN>"           # single-position WDL
chess-wdl-replay --checkpoint <dir> --pgn <pgn>             # per-move WDL for every game + result
```

The TWIC data lives under `data/pgn/` and prepared shards/checkpoints are
gitignored. See `REPORT.md` for the exact reproduce commands.

## Architecture (big picture)

Single Cargo package, library + 5 thin binaries (not the multi-crate workspace
sketched in DESIGN.md — collapsed for iteration speed). Module map (`src/`):

- `encoding.rs` — position → `Sample`: 64 square-tokens of 12-way one-hot
  occupancy in the **side-to-move frame** (board flipped + colors swapped when
  Black to move), plus aux (castling/EP/STM) and the WDL label. Uses `shakmaty`.
- `data.rs` — streaming PGN parse (`pgn-reader` visitor; skips variations/NAGs,
  drops `*`-result games) and the compact fixed-record `.bin` shard format.
- `config.rs` — `ModelConfig` loaded from the `configs/*.toml` arch files.
- `model.rs` — the Candle model: input projection, learned Elo embeddings,
  encoder blocks with **Geometric Attention Bias (GAB)**, and the mean-pool →
  MLP → 3-logit WDL head.
- `metrics.rs` — log-loss / accuracy / Brier / ECE / confusion, the base-rate &
  material-logistic baselines, and temperature calibration. Pure Rust, no Candle.
- `runtime.rs` — device selection, checkpoint save/load (`safetensors` weights +
  `model.toml` + `meta.json`), batched inference, and the WDL loss.

A checkpoint dir is self-describing (`model.toml` travels with the weights), so
`eval`/`predict` rebuild the right architecture without any arch flags.

## Gotchas (learned the hard way — read before touching the model)

- **Candle 0.10 Metal lacks fused kernels** for `layer_norm`, `softmax`, and
  `cross_entropy`/`log_softmax`, and its CPU fused ops are inference-only
  (no backward). So we provide our own: `softmax_lastdim`/`Ln::forward` in
  `model.rs` **dispatch by device** — on **CPU** they call the fused,
  rayon-parallel `CustomOp1`s in `src/fused.rs` (multi-threaded forward + correct
  primitive backward); on **Metal** they use primitive ops. `wdl_loss`
  (`runtime.rs`) is primitive. **Do not** reintroduce
  `candle_nn::{LayerNorm, ops::softmax_last_dim, loss::cross_entropy}` — they
  panic on Metal and aren't differentiable on CPU. If you change `src/fused.rs`,
  re-run the **memorization tests** (`tests/overfit.rs`) — they're what verifies
  the hand-written fused backward gradients.
- **Batch ≥1024 hangs at step 0 on Metal** (≤512 is fine). Unresolved.
- For these small models, **CPU (`--device cpu`, Accelerate) keeps the system
  responsive**; Metal works but its kernel-launch overhead dominates and can make
  the machine sluggish. Both paths are correct.
- Candle `Var`s share storage with the model tensors, so `varmap.load(...)`
  updates the live model in place (used to restore the best checkpoint before
  calibration in `train.rs`).
