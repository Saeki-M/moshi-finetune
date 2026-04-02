#!/usr/bin/env bash
set -euo pipefail

# Change to project root so module paths and relative paths work correctly
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <step_dir>"
    exit 1
fi

input_path="${1%/}"
output_path="${input_path}_fp32"
cleaned_output_path="${output_path}_cleaned"

if [[ -e "$output_path" ]]; then
    echo "Skipping zero_to_fp32: $output_path already exists"
else
    uv run -m tools.zero_to_fp32 "$input_path" "$output_path"
fi

if [[ -e "$cleaned_output_path" ]]; then
    echo "Skipping clean_moshi: $cleaned_output_path already exists"
else
    ln -s "$SCRIPT_DIR/../moshi_finetune_data/llm-jp-moshi-v1-finetuned/moshi_lm_kwargs.json" "$output_path/"
    uv run -m tools.clean_moshi \
        --moshi_ft_dir "$output_path" \
        --save_dir "$cleaned_output_path" \
        --model_dtype float32
fi

uv run -m moshi.server \
    --moshi-weight "$cleaned_output_path/model.safetensors" \
    --tokenizer moshi_finetune_data/llm-jp-moshi-v1-finetuned/tokenizer_spm_32k_3.model
