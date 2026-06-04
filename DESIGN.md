# Chess WDL Prediction Model — Design & Implementation Plan

Status: **Implemented.** A first end-to-end run is complete — see
[`REPORT.md`](REPORT.md) for results and [`CLAUDE.md`](CLAUDE.md) for the build
commands and code map. Design choices are settled in the resolution log in
[§12](#12-resolved-decisions).

**Deviations from this design, as built** (rationale in `CLAUDE.md`):
- Single Cargo package (lib + 4 binaries), not the multi-crate workspace of §5 —
  collapsed for iteration speed; module boundaries match §5.
- Added a smaller `configs/nano.toml` (~0.15M): the §3.6 `tiny` (0.93M) overfit
  the ~1,890-game corpus within one epoch, so `nano` is the working default.
- Candle's Metal backend has no fused `layer_norm`/`softmax`/`cross_entropy`
  kernels, so those are reimplemented from primitive ops (still GPU-resident).
  Batch ≥1024 hangs on Metal; the reported run used CPU for responsiveness.
- **Elo conditioning was removed from the model** (§3.1 below describes the
  original Maia-style design). The corpus is filtered to a fixed strong-GM Elo
  band (`--require-elo --min-elo 2400 --max-elo 2899`), so Elo carries no signal;
  the model input is **board occupancy + aux state only** (`input_dim = 18`).
  Elo filtering at `prepare` time is retained; Elo is still parsed and stored in
  shards as metadata, just not fed to the network.
- Added optional inverse-frequency class weighting (`--draw-weighting`) to
  counter draw dominance in the elite-GM distribution (§7.2, §8).

---

## 1. Goal

Train a model that, given a chess position, predicts the probability that the
game ends in a **Win / Draw / Loss** for the side to move. The model has three
outputs in `[0, 1]` that **sum to 1** (a 3-way softmax over mutually-exclusive
outcomes), trained from the recorded results in a PGN file.

Hard constraints from the brief:

- 100% Rust.
- Extensive automated tests.
- Training and inference must use the **Apple M1 GPU**.
- Architecture **inspired by Maia-3 / Chessformer**
  ([repo](https://github.com/CSSLab/maia3),
  [paper, ICLR 2026](https://arxiv.org/pdf/2605.19091)).

### Non-goals

- We are **not** building a playing engine or move generator. Maia-3's *policy*
  (move-prediction) head is **dropped** — the model is **value-only** (WDL, 3
  outputs). This keeps the architecture, shard format, CLI, and tests simpler.
  Move-prediction is noted as possible future work in §3.5, nothing more.
- No UCI interface, no tree search, no self-play data generation.

---

## 2. What we borrow from Maia-3 / Chessformer

Maia-3 is built on **Chessformer**, an *encoder-only transformer* that treats
the 64 board squares as tokens. The pieces we adopt:

| Chessformer component        | Adopt? | Notes |
|------------------------------|:------:|-------|
| Squares-as-tokens encoding   | ✅ Yes | Natural board geometry, few parameters. |
| Geometric Attention Bias (GAB) | ✅ Yes | The paper's headline contribution; +0.3% on outcome accuracy vs. absolute bias. |
| Board history (n past states) | ❌ No  | **Dropped by design** — predictions use the current position only (see §3.1). |
| Elo / skill conditioning     | ❌ Removed | Maia design described in §3.1; dropped as built (fixed Elo band → no signal). |
| **Value (WDL) head**         | ✅ Yes | **Our only output.** |
| Policy (from-to) head        | ❌ No  | Out of scope (value-only); future work, §3.5. |

Everything below is grounded in the paper's §3 (Methodology) and §4 (Maia-3).

---

## 3. Model architecture

### 3.1 Input encoding (per position)

- **64 tokens**, one per square (a1..h8 from the **side-to-move's perspective** —
  the board is flipped when it is Black to move, so the network always "plays up
  the board").
- Each square carries a **dim-12 one-hot** (or all-zero for an empty square)
  indicating which of the 12 piece types `{P,N,B,R,Q,K} × {own, opponent}`
  occupies it.
- **No history.** Each prediction uses **only the current position** — no past
  board states are encoded or required. (Maia-3 conditions on `n=7` past
  positions; we deliberately drop this so the model is a pure function of the
  position in front of it. → `12` occupancy features per token.)
- **Strength conditioning**: prepend two soft embeddings of dim
  `elo_embed_dim = 128` to every token — one for the side-to-move's rating, one
  for the opponent's. Each is a learned interpolation between a "weak"
  (rating 0) and "strong" (rating 5000) embedding:
  `e_k = γ·e_weak + (1−γ)·e_strong`, with `γ = (5000 − k) / 5000`.
  > Kept **always on** (config `elo_conditioning = true`). On TWIC's narrow
  > ~2400–2650 band it carries little signal, but it transfers to the
  > wider-Elo corpora expected later and costs almost nothing.

Per-token input width: `12 + 2·128 = 12 + 256 = 268` (plus `aux_dim`, below).
A linear projection maps this `→ d_model`.

> **Auxiliary state (castling rights ×4, en-passant file, side-to-move).**
> **Decision: encode just these as a small global feature vector (~6 dims) and
> concatenate it to every token's input** before the `→ d_model` projection.
> We deliberately **exclude repetition count and the halfmove (50-move) clock**:
> those are near-direct "heading to a draw" signals and would leak the label
> rather than let the model predict it.
> This keeps the sequence at exactly **64 tokens**, so GAB stays `64×64` with no
> special-casing. Per-token width becomes `268 + aux_dim`.
>
> Note: these are flags of the *current* position only (no prior-state
> dependence), consistent with the no-history decision.

### 3.2 Geometric Attention Bias (GAB)

GAB produces **dynamic, content-dependent additive biases** for the attention
logits, capturing chess geometry (e.g. diagonals, sliding-piece reach) that
varies with the board state. Per the paper:

1. Project each token to dim `d1`, flatten the 64 tokens, project to `d2` with
   GELU + LayerNorm (this is the *compressed board representation*).
   - **Small-model variant:** replace this projection+flatten with **average
     pooling** over tokens (saves parameters; ~0.2% accuracy cost). We use the
     avg-pool variant for the small preset.
2. Project to depth `h · d3` (one set per attention head) with activation +
   normalization, reshape to `h × d3`.
3. A final **shared** linear projection (`d3 → 4096`) yields biases of shape
   `h × 64 × 64`, added to the QKᵀ dot-product logits **before softmax**.

So each attention layer computes `softmax(QKᵀ/√d + GAB_bias)`. The bias is the
same additive term the paper uses; it lets us reuse standard fused-attention
math (just an additive mask term).

### 3.3 Encoder body

Standard pre-norm transformer encoder blocks:
`x += Attn(LN(x))` then `x += FFN(LN(x))`, FFN = `Linear → GELU → Linear`.
Attention is the GAB-augmented multi-head attention above. Sequence length is
only 64 (+ a couple of special tokens), so attention is cheap and **no flash/
fused kernel is required** — plain batched matmul + softmax is fast on M1.

### 3.4 Value (WDL) head — the required output

Exactly Maia-3's value head:

```
encoder_out  : [B, 64, d_model]
h = mean_pool(encoder_out, dim=tokens)      -> [B, d_model]
h = LayerNorm(h)
h = ReLU(Linear(h, head_hid_dim=128))       -> [B, 128]
logits = Linear(h, 3)                         -> [B, 3]   # (win, draw, loss)
probs  = softmax(logits)                      -> [B, 3]   # sums to 1
```

**Label** for a position = the actual game result, expressed from the
**side-to-move's** perspective (because the board is flipped):

| Game result | White to move | Black to move |
|-------------|---------------|---------------|
| 1-0         | win           | loss          |
| 0-1         | loss          | win           |
| ½-½         | draw          | draw          |

**Loss:** 3-class cross-entropy (optionally with small label smoothing).

### 3.5 Policy head — out of scope (future work)

Dropped per the value-only decision. For the record, Maia-3's interpretable
**from-to** head projects encoder tokens to source-square queries and
destination-square keys, scaled-dot-products them into a `64×64` move-logit
matrix (+ a promotion sub-head). It could later be bolted on as an auxiliary
loss to sharpen the shared representation, but it is **not** built now: no
policy code, flags, shard fields, or tests.

### 3.6 Default size presets

Architecture is **not** set on the command line — it lives in a **TOML config
file** passed via `--model-config` (see §3.8). Five ready-made configs ship in
`configs/`. Numbers are targets (no history dimension — each preset sees a
single position). **`nano` is the working default for the TWIC corpus** — in
practice (see [`REPORT.md`](REPORT.md)) the larger configs overfit this little
data, and `nano` gave the best held-out result.

| Config (`configs/…`) | d_model | layers | heads | ffn | head_hid | gab (d1/d2/d3) | ≈ params | Use |
|------------|:------:|:------:|:-----:|:---:|:--------:|:---:|:--------:|-----|
| `nano.toml`  | 64  | 2  | 2  | 256  | 32  | avg-pool / 64 / 8   | **~0.15M** | **TWIC default (best held-out)** |
| `tiny.toml`  | 128 | 4  | 4  | 512  | 64  | avg-pool / 128 / 16 | ~0.9M | small step up; overfits TWIC |
| `small.toml` | 256 | 8  | 8  | 1024 | 128 | avg-pool / 256 / 32 | ~5M | larger corpora; M1 dev |
| `medium.toml`| 384 | 12 | 12 | 1536 | 128 | 8 / 384 / 32 | ~23M | M1 (slower) |
| `large.toml` | 512 | 16 | 16 | 2048 | 128 | 8 / 512 / 64 | ~79M | only with lots of data |

(`small`/`medium`/`large` mirror Maia-3's released 5M / 23M / 79M family;
`nano` is below the Maia family, added for this small corpus.)

### 3.7 Regularization (important at this data scale)

~150k positions vs. even ~0.8M params means overfitting is the central risk.
Defaults in play:

- **Dropout** `0.1` on attention and FFN — in the **model config** (§3.8).
- **Weight decay** `0.05` (AdamW) — `--weight-decay`.
- **Label smoothing** `0.05` on the WDL target — `--label-smoothing`.
- **Early stopping** on the twic211 validation log-loss — `--early-stop-patience`.
- Optionally **down-sample positions per game** (`--positions-per-game`) so a
  single long game doesn't dominate, and to reduce intra-game correlation.

### 3.8 Architecture config file (`--model-config <FILE>`)

The full model architecture is declared in one TOML file, so runs are
reproducible from a versioned artifact rather than a long flag string. Schema
(see `configs/tiny.toml` for the annotated reference):

```toml
name = "tiny"                 # label, echoed in logs/checkpoints

[model]
d_model         = 128         # embedding / hidden width
layers          = 4           # encoder blocks
num_heads       = 4           # attention heads (must divide d_model)
ffn_dim         = 512         # feed-forward hidden width
head_hidden     = 64          # WDL value-head hidden width
dropout         = 0.1         # attention/FFN dropout (training only)
pos_encoding    = "gab"       # "gab" | "learned_bias" | "none"

[gab]                         # used only when pos_encoding = "gab"
avg_pool       = true         # average-pool variant (cheaper at small scale)
per_square_dim = 0            # d1 — ignored when avg_pool = true
compress_dim   = 128          # d2 — compressed board representation
templates      = 16           # d3 — number of attention-bias templates
```

- **`pos_encoding`** selects the positional encoding: `gab` (default, the
  paper's contribution), `learned_bias` (a static per-head `64×64` bias — useful
  as an ablation and to validate the pipeline before GAB), or `none`.
- **Validation at load:** `num_heads` divides `d_model`; dims > 0; if
  `pos_encoding = "gab"` and `avg_pool = false` then `per_square_dim > 0`.
  Friendly error otherwise.
- **Provenance:** `chess-wdl-train-cli` copies the resolved config **into the
  checkpoint** (`checkpoints/<run>/model.toml`). `eval` and `predict` then load
  the architecture straight from the checkpoint — **no `--model-config` needed**
  at eval/predict time, and the model can never be rebuilt with a mismatched
  shape.

---

## 4. Technology stack & Apple M1 strategy

### 4.1 ML framework — **Decided: Candle + Metal**

| Option | Pure Rust? | M1 GPU path | Verdict |
|--------|:----------:|-------------|---------|
| **[Candle](https://github.com/huggingface/candle)** (HF) | ✅ | First-class **Metal** backend (`Device::new_metal(0)`) | **Chosen.** Low-level control for custom GAB attention; `safetensors` checkpoints; built-in AdamW; small dep tree. |
| **[Burn](https://burn.dev)** | ✅ | Metal via `wgpu`, or `candle`/`tch` backends | Strong alt: nicer `Learner`/metrics/checkpointing. Heavier abstraction; backend-swap is a perk. |
| `tch-rs` (libtorch) | ❌ (C++ dep) | MPS | Mature but not pure Rust; rejected on the "all Rust" constraint. |

**Chosen: Candle + Metal.** GAB is just an additive bias on standard attention,
which Candle expresses directly. (Burn remains a fallback if its training-loop
ergonomics prove worth a migration.)

### 4.2 Apple M1 utilization

- **Device:** `Device::new_metal(0)`; CPU fallback for tests/CI via `--device cpu`.
- **Precision:** train in **f32** on Metal (most robust; Metal bf16/f16 coverage
  in Candle is partial). Offer `--dtype f16` for inference. AMP is *not* assumed.
- **Throughput:** 64-token sequences are tiny, so the GPU is rarely the
  bottleneck — **data loading is.** We overlap CPU-side shard decoding/collation
  (rayon worker threads) with GPU compute via a bounded prefetch channel, and use
  large batches (e.g. 1024+).
- **Memory:** unified memory on M1 means host↔device copies are cheap; we still
  pre-tensorize to mmap'd shards (§6) so epochs don't re-parse PGN.

### 4.3 Key crates

- `shakmaty` — board state, legal moves, FEN, piece placement (by the
  `pgn-reader` author; the de-facto standard).
- `pgn-reader` — streaming PGN parser for multi-GB files.
- `candle-core`, `candle-nn` — tensors, autodiff, NN layers, AdamW, safetensors.
- `clap` (derive) — CLI for all binaries.
- `rayon` — parallel preprocessing & dataloading.
- `serde` + `toml` — model architecture config files (§3.8); `serde_json` for
  metric reports and shard headers.
- `zstd` — optional compressed PGN/shard I/O.
- `anyhow` / `thiserror` — errors. `tracing` — logging.
- Dev: `proptest`, `approx`, `insta` (snapshots), `criterion` (benches).

---

## 5. Workspace layout

A Cargo **workspace** keeps library logic testable and binaries thin:

```
chess-wdl/
├── Cargo.toml                  # [workspace]
├── crates/
│   ├── core/      (chess-wdl-core)   # encoding, WDL labels, position types
│   ├── data/      (chess-wdl-data)   # PGN parsing, shard format, dataloader
│   ├── model/     (chess-wdl-model)  # Chessformer: embeds, GAB attn, heads
│   └── train/     (chess-wdl-train)  # loop, optim, schedule, loss, metrics, ckpt
├── bins/
│   ├── prepare/   -> chess-wdl-prepare
│   ├── train/     -> chess-wdl-train-cli
│   ├── eval/      -> chess-wdl-eval
│   └── predict/   -> chess-wdl-predict
├── configs/       # committed model-arch TOML files (§3.8)
│   ├── nano.toml  ├── tiny.toml  ├── small.toml  ├── medium.toml  └── large.toml
└── tests/fixtures/  # tiny committed .pgn + golden encodings
```

> **Decided: separate binaries** (as above) — matches the brief's "call out all
> binaries"; trivial to merge into one multitool later if desired.

---

## 6. Data pipeline

### 6.1 Corpus — TWIC 210 / 211 / 212

Local copies (gitignored) live under `data/pgn/`:

| Role  | File          | Games | ≈ Positions | Used by |
|-------|---------------|------:|------------:|---------|
| Train | `twic210.pgn` | 1 890 | ~150 k | `prepare` → train shards |
| Test  | `twic211.pgn` | 1 027 | ~80 k  | validation **during** training (early stopping) |
| Eval  | `twic212.pgn` | 963   | ~75 k  | final held-out `eval` only |

Three pre-split files means **no in-corpus game split is needed**
(`--val-fraction` is unused for this dataset; kept for other corpora).

**Properties of this corpus that shape the design:**

- **Master games, narrow Elo (~2400–2650).** Elo conditioning carries little
  signal here (§3.1); decisive-vs-draw balance skews **draw-heavy**, so we watch
  per-class calibration and may apply class weighting / label smoothing.
- **No `%clk` annotations** → `--drop-time-pressure` is a **no-op** here
  (documented; needs clock comments to do anything).
- **Small** (~150k train positions) → `nano` preset + regularization (§3.6–3.7),
  early stopping on the twic211 test set.
- TWIC PGNs occasionally carry annotation glyphs/comments/variations — the
  parser must **skip recursive variations and NAGs** and tolerate missing Elo
  (treat as a sentinel rating, e.g. 1500).
- **Labelling rules:** games with result `*` (unknown/unfinished) are
  **dropped** (unlabelable). **Terminal positions are kept** (checkmate/stalemate
  and the final position are included). `--min-game-plies` can drop very short
  games (byes/forfeits); default `0` (off).

### 6.2 Flow

```
PGN file  ── pgn-reader (stream) ──► replay with shakmaty ──► per-position records
  ▼  filter (--min-ply, Elo range; time-pressure = no-op for TWIC)
  ▼  encode (64×12 occupancy + elo + aux-state, current position only)  +  WDL label
  ▼  write fixed-layout binary SHARDS (mmap-friendly), one shard set per file
```

Each input file is prepared independently into its own output dir (train / test
/ eval), so there is no cross-file leakage by construction.

### 6.3 Shard format

A simple, versioned, memory-mappable binary: a JSON header (record schema,
dims, count, dtype) followed by tightly-packed fixed-size records:

```
record = { board_planes: u8[ceil(12*64 / 8) = 96 bytes],  # bit-packed occupancy
           aux_state: u8[2],          # castling/EP/STM/repetition flags
           self_elo: u16, oppo_elo: u16,
           wdl: u8 }                  # 0=win, 1=draw, 2=loss  (no move field)
```

Bit-packing keeps shards compact; elo embeddings and float tensors are built on
the fly during collation. Round-trip (write→read) is a unit test (§9).
**Decided: custom `bin`** (fastest mmap); `safetensors` remains a selectable
alternative via `--format`.

---

## 7. Binaries & command-line flags

All binaries share: `--device <metal|cpu>` (default `metal`, auto-fallback to
`cpu`), `--seed <N>`, `-v/--verbose`, `--help`/`--version`.

### 7.1 `chess-wdl-prepare` — PGN → training shards

| Flag | Default | Meaning |
|------|---------|---------|
| `--input <PATH>...` | — (required) | One or more `.pgn` / `.pgn.zst` files, or `-` for stdin. |
| `--output <DIR>` | — (required) | Shard output directory. |
| `--shard-size <N>` | `1_000_000` | Positions per shard file. |
| `--min-elo <N>` | `0` | Drop games with either player below this. |
| `--max-elo <N>` | `4000` | Drop games above. |
| `--min-ply <N>` | `0` | Only keep positions at least N plies into the game (skips the opening; e.g. ~20). |
| `--min-game-plies <N>` | `0` | Drop entire games shorter than N plies (byes/forfeits). |
| `--drop-time-pressure` | off | Drop moves made under time pressure. |
| `--min-clock-seconds <F>` | `30` | Time-pressure threshold (needs clock comments). |
| `--resample-elo` | off | Flatten the Elo distribution across games. |
| `--val-fraction <F>` | `0.02` | Fraction of **games** held out for validation. |
| `--positions-per-game <N>` | all | Cap sampled positions per game (decorrelation). |
| `--threads <N>` | #cores | Parser/encoder workers. |
| `--limit-games <N>` | ∞ | Process at most N games (debugging). |
| `--format <bin\|safetensors>` | `bin` | Shard encoding. |

### 7.2 `chess-wdl-train-cli` — train the model

**Data**
| Flag | Default | Meaning |
|------|---------|---------|
| `--data <DIR>` | required | Training shards. |
| `--val-data <DIR>` | `<data>/val` | Validation shards. |

**Architecture** — a single flag points at a TOML file (§3.8); all structural
hyperparameters live there, not on the command line.
| Flag | Default | Meaning |
|------|---------|---------|
| `--model-config <FILE>` | `configs/nano.toml` | Model architecture TOML (§3.8). Copied into the checkpoint for reproducibility. |

**Optimization**
| Flag | Default | Meaning |
|------|---------|---------|
| `--epochs <N>` | `30` | Passes over train shards (early stopping usually ends sooner). |
| `--max-steps <N>` | — | Optional hard cap on optimizer steps. |
| `--batch-size <N>` | `512` | Positions per step. |
| `--lr <F>` | `3e-4` | Peak learning rate. |
| `--weight-decay <F>` | `0.05` | AdamW weight decay. |
| `--warmup-steps <N>` | `200` | LR warmup (data is small). |
| `--lr-schedule <cosine\|linear\|constant>` | `cosine` | Decay schedule. |
| `--grad-clip <F>` | `1.0` | Global-norm gradient clip. |
| `--value-loss-weight <F>` | `1.0` | Weight on WDL loss. |
| `--label-smoothing <F>` | `0.05` | WDL label smoothing. |
| `--draw-weighting` | off | Balance the loss by inverse class frequency (counters draw dominance). |

**System / bookkeeping**
| Flag | Default | Meaning |
|------|---------|---------|
| `--dtype <f32\|f16>` | `f32` | Compute dtype on Metal. |
| `--workers <N>` | #cores | Dataloader threads. |
| `--checkpoint-dir <DIR>` | `./checkpoints` | Output dir. |
| `--save-interval <STEPS>` | `200` | Checkpoint cadence. |
| `--keep-last <N>` | `5` | Checkpoint retention. |
| `--resume <PATH>` | — | Resume from checkpoint. |
| `--val-interval <STEPS>` | `200` | Validation cadence (on the test shards). |
| `--early-stop-patience <N>` | `10` | Stop after N val checks without log-loss improvement (0 = off). |
| `--log-interval <STEPS>` | `50` | Log cadence. |
| `--run-name <NAME>` | timestamp | Names log/metrics files. |

> The `--val-data` here points at the **twic211** ("test") shards — it drives
> early stopping. The untouched **twic212** ("eval") shards are only ever seen
> by `chess-wdl-eval` after training, so model selection never peeks at them.

### 7.3 `chess-wdl-eval` — evaluate a checkpoint

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint <PATH>` | required | Model to evaluate. |
| `--data <DIR>` | — | Eval shards… |
| `--pgn <PATH>` | — | …or evaluate a raw PGN directly. |
| `--min-ply <N>` | `0` | With `--pgn`: only score positions at least N plies into the game. |
| `--batch-size <N>` | `1024` | Eval batch. |
| `--metrics <list>` | `all` | `logloss,accuracy,brier,ece,confusion`. |
| `--baseline` | off | Also report base-rate and material-logistic baselines (§8). |
| `--by-elo` | off | Bucket metrics by rating. |
| `--by-phase` | off | Bucket by opening/middlegame/endgame. |
| `--output <PATH>` | stdout | Write JSON report. |

### 7.4 `chess-wdl-predict` — inference on positions

| Flag | Default | Meaning |
|------|---------|---------|
| `--checkpoint <PATH>` | required | Model. |
| `--fen <FEN>` | — | Score one position. |
| `--pgn <PATH>` | — | Score every position reached in a game (each independently). |
| `--min-ply <N>` | `0` | With `--pgn`: only score positions at least N plies into the game. |
| `--batch` | off | Read FENs from stdin, one per line. |
| `--json` | off | Machine-readable output. |

Example output (`--json`): `{"win":0.41,"draw":0.34,"loss":0.25}`.

> Optional 5th binary `chess-wdl-export` (safetensors → quantized/portable
> bundle) is a stretch goal, not in the core plan.

---

## 8. Loss, metrics, and calibration

- **Training loss:** `value_loss_weight · CE_wdl` (3-class cross-entropy with
  label smoothing; optional per-class weighting for draw imbalance).
- **Reported metrics:** accuracy, **log-loss** (primary), **Brier score**,
  **Expected Calibration Error (ECE)** + reliability buckets, confusion matrix.
  Calibration matters for a probabilistic predictor, so ECE/Brier are first-class.
- Optional bucketing by Elo and by game phase to mirror the paper's analysis.
- **Baselines (`eval --baseline`).** Always report trivial predictors next to
  the model so its added value is unambiguous: (a) **base-rate** — the global
  W/D/L frequencies from the training set; (b) **material-logistic** — a 1-feature
  logistic on material balance. If the model can't beat these, we say so.
- **Probability calibration.** After training we fit a single **temperature**
  `T` on the twic211 logits (minimise log-loss), report ECE before/after, and
  store `T` in the checkpoint so `predict` emits calibrated probabilities.

---

## 9. Testing strategy (extensive)

Correctness of a learned model rests on the **deterministic plumbing** around it,
so most tests target encoding, labeling, shapes, and the data path. Layers:

**Unit — `core`**
- Board encoding: known FEN → exact expected occupancy planes (golden/`insta`).
- Side-to-move flip: a position and its color-mirror encode identically.
- WDL label table (§3.4) for all `result × side-to-move` combinations.
- Elo interpolation `e_k`: endpoints (`k=0`, `k=5000`) and `γ` midpoint exact.
- A position encodes identically regardless of the moves that preceded it
  (no history dependence) — same FEN ⇒ same planes.

**Unit — `model`**
- Shape contracts for every layer (GAB bias is `h×64×64`; value logits `B×3`).
- Softmax outputs are non-negative and **sum to 1** (`approx`).
- GAB additive-bias math: a zero bias ⇒ attention equals plain dot-product.
- **Numerical-gradient check** of the custom GAB layer (finite differences vs.
  autodiff) on a tiny tensor.
- Determinism: same seed + input ⇒ identical logits.

**Unit — `data`**
- Shard round-trip: write records → mmap read → byte-for-byte equal.
- Bit-packing/unpacking of occupancy planes is lossless.
- `--min-ply` / Elo filters drop exactly the right records; missing-Elo games
  fall back to the sentinel rating; variations/NAGs are skipped during parse.

**Property tests (`proptest`)**
- For arbitrary legal positions (random playouts via shakmaty): encoding has
  exactly one piece bit set per occupied square, zero for empty; round-tripping
  through the shard packer reproduces the original planes.

**Integration**
- **Overfit/memorization tests (key signal):** `tests/overfit.rs` trains the
  `nano` and `tiny` models on a single example and on a 16-sample batch for a
  few hundred steps on **CPU**; asserts loss → ~0 and predictions match labels.
  Exercises the whole forward/backward/optimizer path (and validates gradients).
- **End-to-end:** the committed fixture PGN (a handful of games sliced from
  `twic210`) → `prepare` → `train` (a few steps, CPU) → `eval` → `predict`;
  assert files produced and probabilities are valid (non-negative, sum to 1).
- Metrics: log-loss of a uniform predictor equals `ln 3`; perfect predictor → 0.

**CI:** all tests run on `--device cpu` (no GPU in CI). A separate, ignored
`#[test]` (`cargo test -- --ignored`) runs a Metal smoke test locally on the M1.

**Benches (`criterion`, optional):** encoding throughput, single training step.

---

## 10. Implementation milestones

1. **Workspace + `core`**: types, board/Elo encoding, WDL labels, golden tests.
2. **`data`**: PGN streaming (variations/NAGs skipped), filters, shard format,
   dataloader; round-trip tests.
3. **`prepare` binary**: wire `core`+`data`; run on `twic210/211/212` → shards.
4. **`model`**: embeddings → GAB attention → encoder → value head; shape/grad tests.
5. **`train`**: loss, AdamW, schedule, checkpointing, early stopping, metrics;
   **overfit test**.
6. **`train-cli` binary** on Metal; train the `tiny` preset on twic210 with
   early stopping on twic211; confirm M1 GPU utilization.
7. **`eval` + `predict` binaries**; calibration metrics; final eval on twic212;
   end-to-end test.
8. **Tuning**: dropout / label-smoothing / class-weight sweep; consider `small`
   preset only if the val curve supports it (watch overfitting).

---

## 11. Risks & mitigations

- **Custom GAB autodiff in Candle.** Mitigate by composing GAB from standard
  differentiable ops (matmul, GELU, LayerNorm, add) — no hand-written backward —
  and gating it behind a numerical-gradient test.
- **Metal op coverage / dtype gaps in Candle.** Default to f32; keep a CPU path
  so nothing blocks on Metal. Smoke-test Metal early (milestone 6, not last).
- **Overfitting (the #1 risk here).** ~150k positions vs. ~0.8M params.
  Mitigate with the `nano` preset, dropout, weight decay, label smoothing, and
  early stopping on twic211; report the twic212 gap honestly.
- **Draw imbalance.** TWIC master games are draw-heavy; monitor per-class
  calibration, use label smoothing and optional `--class-weights`.
- **Narrow Elo band** (~2400–2650) means Elo conditioning is nearly inert on
  this corpus — kept for architectural fidelity / transfer, not expected to help
  here.
- **Dataloader becomes the bottleneck** (64-token seqs are GPU-cheap). Mitigate
  with mmap shards + rayon prefetch.
- **PGN quirks**: TWIC files carry variations, NAGs, comments, and occasionally
  missing Elo — the parser must skip the first three and fall back to a sentinel
  rating for the last. `--drop-time-pressure` is a **no-op** (no `%clk`).

---

## 12. Resolved decisions

| # | Decision | Resolution |
|---|----------|------------|
| 1 | ML framework | **Candle + Metal** *(§4.1)* |
| 2 | WDL output | **3-way softmax** (sums to 1) *(§3.4)* |
| 3 | Policy head | **Dropped — value-only** *(§3.5)* |
| 4 | Training corpus | **TWIC**: `twic210` train · `twic211` test/early-stop · `twic212` eval; copied to `data/pgn/` (gitignored) *(§6.1)* |
| 5 | Auxiliary state | **In**, as per-token global features (seq stays 64) *(§3.1)* |
| 6 | Binaries | **4 separate executables** *(§5, §7)* |
| 7 | Shard format | **Custom `bin`** (mmap); `safetensors` selectable *(§6.3)* |
| 8 | Default preset | **`nano` (~0.15M)** for the small TWIC corpus *(§3.6)* |
| 9 | Board history | **Dropped** — current position only, no past states *(§3.1)* |
| 10 | Architecture config | **TOML file via `--model-config`** (not CLI flags); 4 presets in `configs/`; copied into the checkpoint *(§3.8)* |

**Remaining minor items to revisit during tuning** (not blockers): exact `tiny`
param budget and GAB dims; whether `--class-weights` is needed for draw
imbalance; whether a nonzero `--min-ply` helps on master games.

No further sign-off needed to begin Milestone 1 — awaiting your go-ahead.
```
