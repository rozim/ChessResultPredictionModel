# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository is a fresh skeleton. As of this writing it contains only `README.md`, `LICENSE`, and `.gitignore` — there is no `Cargo.toml` or source code yet. The goal implied by the name is a model that predicts chess game results.

When you add the first code, **update this file** with the real build/test commands and architecture once they exist.

## Language & tooling

The `.gitignore` is configured for **Rust / Cargo**, including a `cargo mutants` entry, so this is intended to be a Cargo project with mutation testing.

Once a `Cargo.toml` exists, the standard workflow will be:

```bash
cargo build              # compile
cargo test               # run all tests
cargo test <name>        # run tests matching <name>
cargo fmt                # format (rustfmt)
cargo clippy             # lint
cargo mutants            # mutation testing (requires: cargo install cargo-mutants)
```
