import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

from .utils import execute_data_processing

processor: SentencePieceProcessor | None = None


def detokenize_token_ids(
    token_ids: list[int],
    text_tokenizer: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
    audio_tokenizer_frame_rate: float,
    no_whitespace_before_word: bool = False,
) -> list[dict[str, str | float]]:
    """
    Detokenize token IDs back to word-level transcript with timestamps.

    Args:
        token_ids: List of token IDs with padding
        text_tokenizer: SentencePiece tokenizer
        text_padding_id: ID for padding tokens
        end_of_text_padding_id: ID for end of text padding
        audio_tokenizer_frame_rate: Frame rate of audio tokenizer
        no_whitespace_before_word: If True, don't add spaces between words (for Japanese, Chinese, etc.)

    Returns:
        List of dictionaries with 'word', 'start', and 'end' keys
    """
    seconds_per_frame = 1 / audio_tokenizer_frame_rate

    # Extract tokens with their frame positions
    token_positions = []
    for frame_idx, token_id in enumerate(token_ids):
        if token_id not in [text_padding_id, end_of_text_padding_id]:
            token_positions.append(
                {
                    "frame_idx": frame_idx,
                    "token_id": token_id,
                    "time": frame_idx * seconds_per_frame,
                }
            )

    if not token_positions:
        return []

    # Decode tokens to pieces
    decoded_pieces = []
    for pos in token_positions:
        piece = text_tokenizer.id_to_piece(pos["token_id"])
        decoded_pieces.append({"piece": piece, "start": pos["time"], "frame_idx": pos["frame_idx"]})

    # Reconstruct words from pieces
    # SentencePiece uses ▁ to denote word boundaries (beginning of word)
    word_transcript = []

    if no_whitespace_before_word:
        # For languages without whitespace (Japanese, Chinese, etc.),
        # treat each token piece as a separate word entry
        for item in decoded_pieces:
            piece = item["piece"]
            # Remove the underscore marker if present
            word_text = piece.replace("▁", "")
            if word_text:  # Skip empty strings
                word_transcript.append(
                    {
                        "word": word_text,
                        "start": item["start"],
                        "end": item["start"] + seconds_per_frame,
                    }
                )
    else:
        # For languages with whitespace, group tokens into words
        current_word = ""
        word_start = None

        for i, item in enumerate(decoded_pieces):
            piece = item["piece"]

            # Check if this is the start of a new word
            if piece.startswith("▁"):
                # Save previous word if exists
                if current_word:
                    word_end = decoded_pieces[i - 1]["start"] + seconds_per_frame
                    word_transcript.append(
                        {"word": current_word, "start": word_start, "end": word_end}
                    )

                # Start new word (remove the underscore marker)
                current_word = piece.replace("▁", " ").lstrip()
                word_start = item["start"]
            else:
                # Continue current word
                if word_start is None:
                    # First piece doesn't start with ▁ (rare case)
                    current_word = piece
                    word_start = item["start"]
                else:
                    current_word += piece

        # Add the last word
        if current_word and word_start is not None:
            word_end = decoded_pieces[-1]["start"] + seconds_per_frame
            word_transcript.append({"word": current_word, "start": word_start, "end": word_end})

    return word_transcript


def process_dialogue(worker_id: int, dialogue_name: str, args: argparse.Namespace):
    """Process a single dialogue by detokenizing its token IDs.

    Args:
        worker_id: ID of the worker process (not used but required by execute_data_processing).
        dialogue_name: Name of the dialogue to process.
        args: Command-line arguments namespace.
    """
    global processor
    if processor is None:
        processor = SentencePieceProcessor(
            hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
        )

    # Load tokenized data
    tokenized_path = Path(args.tokenized_dir) / f"{dialogue_name}.npz"
    try:
        tokenized_data = np.load(tokenized_path)
    except Exception as e:
        print(f"Error loading {tokenized_path}: {e}")
        return

    # Detokenize speaker A
    token_ids_A = tokenized_data["A"].tolist()
    word_transcript_A = detokenize_token_ids(
        token_ids=token_ids_A,
        text_tokenizer=processor,
        text_padding_id=args.text_padding_id,
        end_of_text_padding_id=args.end_of_text_padding_id,
        audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
        no_whitespace_before_word=args.no_whitespace_before_word,
    )

    # Add speaker label
    for segment in word_transcript_A:
        segment["speaker"] = "A"

    # Detokenize speaker B
    token_ids_B = tokenized_data["B"].tolist()
    word_transcript_B = detokenize_token_ids(
        token_ids=token_ids_B,
        text_tokenizer=processor,
        text_padding_id=args.text_padding_id,
        end_of_text_padding_id=args.end_of_text_padding_id,
        audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
        no_whitespace_before_word=args.no_whitespace_before_word,
    )

    # Add speaker label
    for segment in word_transcript_B:
        segment["speaker"] = "B"

    # Combine and sort by start time
    word_transcript = word_transcript_A + word_transcript_B
    word_transcript = sorted(word_transcript, key=lambda x: x["start"])

    # Save to JSON
    output_path = Path(args.output_dir) / f"{dialogue_name}.json"
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(word_transcript, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        output_path.unlink(missing_ok=True)


def main(args):
    tokenized_input = Path(args.tokenized_input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if input is a file or directory
    if tokenized_input.is_file():
        # Process single file
        if tokenized_input.suffix != ".npz":
            raise ValueError(f"Input file must be a .npz file, got: {tokenized_input}")

        dialogue_names = {tokenized_input.stem}
        # Update args to use the parent directory for processing
        args.tokenized_dir = str(tokenized_input.parent)
    elif tokenized_input.is_dir():
        # Process directory
        args.tokenized_dir = str(tokenized_input)
        dialogue_names = {p.stem for p in tokenized_input.glob("*.npz")}

        if args.resume:
            detokenized_dialogue_names = {p.stem for p in output_dir.glob("*.json")}
            print(f"Skipping {len(detokenized_dialogue_names)} already detokenized dialogues.")
            dialogue_names = dialogue_names - detokenized_dialogue_names
    else:
        raise ValueError(f"Input path does not exist: {tokenized_input}")

    if not dialogue_names:
        print("No files to process.")
        return

    # Execute data processing using the utility function
    execute_data_processing(
        dataset=iter(sorted(dialogue_names)),
        process_func=partial(process_dialogue, args=args),
        num_workers=args.num_workers,
        data_count=len(dialogue_names),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detokenize token IDs back to word-level transcripts with timestamps."
    )
    parser.add_argument(
        "--tokenized_input",
        type=str,
        required=True,
        help="Path to a tokenized .npz file or directory containing tokenized data (.npz files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the detokenized JSON transcripts.",
    )

    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Repository of the text tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
        help="Name of the text tokenizer.",
    )

    parser.add_argument(
        "--no_whitespace_before_word",
        action="store_true",
        help=(
            "No whitespace before each word. Set this flag if the language "
            "has no whitespace between words (e.g., Japanese and Chinese)."
        ),
    )

    parser.add_argument("--text_padding_id", type=int, default=3, help="Padding id for text.")
    parser.add_argument(
        "--end_of_text_padding_id", type=int, default=0, help="End of text padding id."
    )
    parser.add_argument(
        "--audio_tokenizer_frame_rate",
        type=float,
        default=12.5,
        help="Frame rate for the audio tokenizer.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument("--resume", action="store_true", help="Resume detokenization.")

    args = parser.parse_args()

    main(args)
