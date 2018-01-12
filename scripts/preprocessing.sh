#!/bin/bash
moses="$1"
corpus="$2"
src="$3"
trg="$4"
min_sent_length="$5"
max_sent_length="$6"

# Tokenize parallel corpus.
"${moses}/scripts/tokenizer/tokenizer.perl" -l "${src}" < "${corpus}.${src}" > "${corpus}.tok.${src}" -threads 4
"${moses}/scripts/tokenizer/tokenizer.perl" -l "${trg}" < "${corpus}.${trg}" > "${corpus}.tok.${trg}" -threads 4

# Process special characters.
"${moses}/scripts/tokenizer/deescape-special-chars.perl" < "${corpus}.tok.${src}" > "${corpus}.tmp.${src}"
mv "${corpus}.tmp.${src}" "${corpus}.tok.${src}"
"${moses}/scripts/tokenizer/deescape-special-chars.perl" < "${corpus}.tok.${trg}" > "${corpus}.tmp.${trg}"
mv "${corpus}.tmp.${trg}" "${corpus}.tok.${trg}"

# Train truecaser models.
"${moses}/scripts/recaser/train-truecaser.perl" --model "${corpus}_truecase-model.${src}" --corpus "${corpus}.tok.${src}"
"${moses}/scripts/recaser/train-truecaser.perl" --model "${corpus}_truecase-model.${trg}" --corpus "${corpus}.tok.${trg}"

# Apply truecaser models.
"${moses}/scripts/recaser/truecase.perl" --model "${corpus}_truecase-model.${src}" < "${corpus}.tok.${src}" > "${corpus}.true.${src}"
"${moses}/scripts/recaser/truecase.perl" --model "${corpus}_truecase-model.${trg}" < "${corpus}.tok.${trg}" > "${corpus}.true.${trg}"

# Clean final parallel corpus.
"${moses}/scripts/training/clean-corpus-n.perl" "${corpus}.true" "${src}" "${trg}" "${corpus}.clean" "${min_sent_length}" "${max_sent_length}"
rm "${corpus}.tok.${src}" "${corpus}.tok.${trg}" "${corpus}.true.${src}" "${corpus}.true.${trg}"
