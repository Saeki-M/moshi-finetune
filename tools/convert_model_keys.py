"""Convert old-format model keys (linear_in.weight) to new flat format (linear_in_weight)."""
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

from safetensors.torch import load_file, save_file


def convert_keys(state_dict: dict) -> tuple[dict, int]:
    patterns = [
        (re.compile(r"(transformer\.layers\.\d+\.gating)\.linear_in\.weight"), r"\1.linear_in_weight"),
        (re.compile(r"(transformer\.layers\.\d+\.gating)\.linear_out\.weight"), r"\1.linear_out_weight"),
        (re.compile(r"(depformer\.layers\.\d+\.gating\.\d+)\.linear_in\.weight"), r"\1.linear_in_weight"),
        (re.compile(r"(depformer\.layers\.\d+\.gating\.\d+)\.linear_out\.weight"), r"\1.linear_out_weight"),
        (re.compile(r"(depformer\.layers\.\d+\.self_attn)\.out_proj\.weight"), r"\1.out_proj_weight"),
    ]

    new_state_dict = OrderedDict()
    converted = 0
    for key, tensor in state_dict.items():
        new_key = key
        for pattern, replacement in patterns:
            new_key, n = pattern.subn(replacement, new_key)
            converted += n
        new_state_dict[new_key] = tensor

    return new_state_dict, converted


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_dir>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    safetensors_path = model_dir / "model.safetensors"

    if not safetensors_path.exists():
        print(f"Error: {safetensors_path} not found")
        sys.exit(1)

    print(f"Loading {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    new_state_dict, converted = convert_keys(state_dict)

    if converted == 0:
        print("No keys needed conversion. Model is already in the new format.")
        return

    print(f"Converted {converted} keys.")

    backup_path = safetensors_path.with_suffix(".safetensors.bak")
    print(f"Backing up original to {backup_path}...")
    shutil.copy2(safetensors_path, backup_path)

    print(f"Saving converted model to {safetensors_path}...")
    save_file(new_state_dict, safetensors_path)
    print("Done.")


if __name__ == "__main__":
    main()
