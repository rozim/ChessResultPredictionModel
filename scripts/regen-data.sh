#!/usr/bin/env bash
# Regenerate train/test/eval shards from a TWIC PGN collection.
# Sorted glob; last two issues held out (test, eval), the rest -> train.
# Test/eval positions are stamped with a `seen` flag (also in the train set?)
# for memorization-split metrics in `chess-wdl-eval`.
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-data/pgn}"
OUT="${2:-data/shards}"
PREP=./target/release/chess-wdl-prepare

files=( $(ls -1 "$SRC"/twic*.pgn | sort) )
n=${#files[@]}
echo "Matched $n files from $SRC"
[ "$n" -ge 3 ] || { echo "ERROR: need >=3 files"; exit 1; }

eval_file="${files[$((n-1))]}"
test_file="${files[$((n-2))]}"
train_files=( "${files[@]:0:$((n-2))}" )
echo "TRAIN: ${#train_files[@]} files (${train_files[0]} .. ${train_files[$((${#train_files[@]}-1))]})"
echo "TEST : $test_file"
echo "EVAL : $eval_file"

rm -rf "$OUT/train" "$OUT/test" "$OUT/eval"

# Train first, so test/eval can be stamped against it.
echo "=== prepare TRAIN ==="; "$PREP" --input "${train_files[@]}" --output "$OUT/train"
echo "=== prepare TEST ===";  "$PREP" --input "$test_file" --output "$OUT/test" --seen-against "$OUT/train"
echo "=== prepare EVAL ===";  "$PREP" --input "$eval_file" --output "$OUT/eval" --seen-against "$OUT/train"

echo "=== done ==="; du -sh "$OUT"/train "$OUT"/test "$OUT"/eval
