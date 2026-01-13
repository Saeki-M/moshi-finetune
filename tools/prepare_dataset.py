import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from tools.utils import execute_data_processing


def merge_text_audio(
    text_ids: np.ndarray, audio_ids: np.ndarray, text_padding_id: int
) -> np.ndarray:
    """
    Merge the tokenized text and audio stream of a single speaker.
    Args:
        text_ids: Tokenized text stream. Shape: [T_text]
        audio_ids: Tokenized audio stream. Shape: [K=8, T_audio]
        text_padding_id: Padding id for text stream to fill the gap between audio and text streams.
    Returns:
        Merged tokenized text and audio stream. Shape: [K=8+1, T_audio]
    """
    assert text_ids.ndim == 1, f"Expected 1D tensor, got {text_ids.ndim}D tensor."
    assert audio_ids.ndim == 2, f"Expected 2D tensor, got {audio_ids.ndim}D tensor."
    # pad the text stream to match the audio stream
    audio_len = audio_ids.shape[-1]
    if text_ids.shape[0] > audio_len:
        text_ids = text_ids[:audio_len]
    elif text_ids.shape[0] < audio_len:
        text_ids = np.concat(
            [text_ids, np.full(audio_len - text_ids.shape[0], text_padding_id)], axis=0
        )
    return np.concat([text_ids[None], audio_ids], axis=0).astype(np.int32).tolist()


def process_parquet(
    worker_id: int,
    parquet_info: tuple,
    tokenized_text_dir: str,
    tokenized_audio_dir: str,
    output_prefix: str,
    text_padding_id: int,
    num_parquets: int,
):
    """Process a single parquet file.

    Args:
        worker_id: Worker ID (not used but required by execute_data_processing)
        parquet_info: Tuple containing (dialogue_names, parquet_idx)
        tokenized_text_dir: Path to tokenized text directory
        tokenized_audio_dir: Path to tokenized audio directory
        output_prefix: Output prefix for parquet files
        text_padding_id: Padding ID for text stream
        num_parquets: Total number of parquet files
    """
    dialogue_names, parquet_idx = parquet_info

    # load the tokenized text and audio data
    data = []
    for dialogue_name in tqdm(
        dialogue_names,
        desc=f"Worker {worker_id}: Parquet {parquet_idx + 1}/{num_parquets}",
        position=worker_id + 1,
        leave=False,
    ):
        text_path = os.path.join(tokenized_text_dir, f"{dialogue_name}.npz")
        try:
            text_ids = np.load(text_path)
            assert "A" in text_ids and "B" in text_ids, f"Missing speakers in {text_path}"
        except Exception as e:
            print(f"Error loading text file {text_path}: {e}")
            continue
        audio_path = os.path.join(tokenized_audio_dir, f"{dialogue_name}.npz")
        audio_ids = np.load(audio_path)
        data.append(
            {
                "dialogue_id": os.path.join(output_prefix, dialogue_name),  # unique identifier
                "A": merge_text_audio(text_ids["A"], audio_ids["A"], text_padding_id),
                "B": merge_text_audio(text_ids["B"], audio_ids["B"], text_padding_id),
            }
        )

    # save the merged data
    df = pd.DataFrame(data)
    output_path = f"{output_prefix}-{parquet_idx + 1:03d}-of-{num_parquets:03d}.parquet"
    df.to_parquet(output_path, index=False)


def main(args):
    text_dialogue_names = [os.path.splitext(f)[0] for f in os.listdir(args.tokenized_text_dir)]
    audio_dialogue_names = [os.path.splitext(f)[0] for f in os.listdir(args.tokenized_audio_dir)]
    missing_text_dialogue_names = set(audio_dialogue_names) - set(text_dialogue_names)
    missing_audio_dialogue_names = set(text_dialogue_names) - set(audio_dialogue_names)
    if missing_text_dialogue_names:
        print(f"Missing tokenized text for {len(missing_text_dialogue_names)} dialogues.")
        open("missing_text_dialogue_names.txt", "w").write("\n".join(missing_text_dialogue_names))
    if missing_audio_dialogue_names:
        print(f"Missing tokenized audio for {len(missing_audio_dialogue_names)} dialogues.")
        open("missing_audio_dialogue_names.txt", "w").write("\n".join(missing_audio_dialogue_names))
    if not args.ignore_missing and (missing_text_dialogue_names or missing_audio_dialogue_names):
        print("Both text and audio tokenized dialogues should match.")
        return

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)
    dialogue_names = text_dialogue_names
    num_dialogues = len(dialogue_names)
    num_parquets = -(-num_dialogues // args.num_examples_per_parquet)

    # Prepare data for each parquet file
    def parquet_data_iterator():
        for i in range(num_parquets):
            dials_per_parquet = dialogue_names[
                i * args.num_examples_per_parquet : (i + 1) * args.num_examples_per_parquet
            ]
            yield (dials_per_parquet, i)

    # Process parquet files in parallel using partial to bind fixed parameters
    process_func = partial(
        process_parquet,
        tokenized_text_dir=args.tokenized_text_dir,
        tokenized_audio_dir=args.tokenized_audio_dir,
        output_prefix=args.output_prefix,
        text_padding_id=args.text_padding_id,
        num_parquets=num_parquets,
    )

    execute_data_processing(
        dataset=parquet_data_iterator(),
        process_func=process_func,
        num_workers=args.num_workers,
        data_count=num_parquets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge the tokenized text and audio data into a single dataset in parquet format."
    )
    parser.add_argument(
        "--ignore-missing",
        action="store_true",
        help="Ignore data which doesn't have both audio and text tokens. Print the number of ignored data.",
    )
    parser.add_argument(
        "--tokenized_text_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized text data.",
    )
    parser.add_argument(
        "--tokenized_audio_dir",
        type=str,
        required=True,
        help="Path to the directory containing the tokenized audio data.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=True,
        help=(
            "Prefix for the output dataset. Output files will be named as "
            "`{{output_prefix}}-001-of-002.parquet` etc."
        ),
    )
    parser.add_argument(
        "--text_padding_id",
        type=int,
        default=3,
        help="Padding id for text stream to fill the gap between audio and text streams.",
    )
    parser.add_argument(
        "--num_examples_per_parquet",
        type=int,
        default=100_000,
        help="Number of samples per parquet file.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing.",
    )
    args = parser.parse_args()

    main(args)
