import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

from .utils import execute_data_processing


def encode_as_pieces_wo_byte_fallback(sp: SentencePieceProcessor, text: str) -> list[str]:
    """
    Tokenize the text without using byte fallback.
    """
    tokens = sp.encode_as_pieces(text)
    if not tokens:
        return []

    tokens_wo_byte = []
    last_byte_tokens = []
    for token in tokens:
        if not token.startswith("<0x"):
            tokens_wo_byte.append(token)
            text = text[len(token) :]
        else:
            last_byte_tokens.append(token)
            token = sp.decode_pieces(last_byte_tokens)
            if text.startswith(token):  # token is successfully decoded
                tokens_wo_byte.append(token)
                text = text[len(token) :]
                last_byte_tokens = []
    if last_byte_tokens:
        raise ValueError(f"Failed to decode the last byte tokens: {last_byte_tokens} in {text}")
    return tokens_wo_byte


def get_char_level_transcript(word_transcript):
    """convert word-level transcript to character-level transcript,
    by splitting time evenly across words into characters."""
    char_transcript = []
    for seg in word_transcript:
        num_chars = len(seg["word"])
        start = seg["start"]
        end = seg["end"]
        # split the duration into num_chars
        char_duration = (end - start) / num_chars
        for i, char in enumerate(seg["word"]):
            char_transcript.append(
                {
                    "speaker": seg["speaker"],
                    "start": start + i * char_duration,
                    "end": start + (i + 1) * char_duration,
                    "char": char,
                }
            )
    return char_transcript


def get_token_level_transcript(char_transcript, text_tokenizer):
    # make token-level transcript by aligning the timestamps
    text = "".join([seg["char"] for seg in char_transcript])
    tokens = encode_as_pieces_wo_byte_fallback(text_tokenizer, text)

    token_transcript = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "▁":
            # skip the first underscore of sentencepiece
            continue
        if i == 0 and token.startswith("▁"):
            # don't count the first underscore
            chars = char_transcript[: len(token) - 1]
        else:
            chars = char_transcript[: len(token)]

        # print(f"{i}\t|{token}|\t|{''.join([c['char'] for c in chars])}|")
        token_transcript.append(
            {
                "speaker": chars[0]["speaker"],
                "start": chars[0]["start"],
                "end": chars[-1]["end"],
                "token": token,
            }
        )
        # remove the characters that are already processed
        char_transcript = char_transcript[len(chars) :]
    assert not char_transcript, f"Remaining characters: {char_transcript}"
    return token_transcript


TEXT_NORMALIZATION_TABLE = str.maketrans(
    {
        "…": "", "℃": "°", "�": "", "゛": "",
        "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5", 
        "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
    }
)  # fmt: skip


def normalize_word_transcript(word_transcript):
    """Normalize the transcript by converting or removing unwanted characters."""
    normalized_transcript = []
    for seg in word_transcript:
        word = seg["word"]
        word = word.translate(TEXT_NORMALIZATION_TABLE)
        if word != "":
            normalized_transcript.append(
                {
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "word": word,
                }
            )
    return normalized_transcript


def normalize_char_transcript(char_transcript, no_whitespace_before_word: bool):
    """Further normalize the character-level transcript if needed."""
    normalized_transcript = []
    del_leading_space = no_whitespace_before_word
    for seg in char_transcript:
        char = seg["char"]
        if char == " " and del_leading_space:
            continue
        # if last char is Japanese or space delete leading space in the next word
        if "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9faf" or char == " ":
            del_leading_space = True
        else:
            del_leading_space = False

        normalized_transcript.append(
            {
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "char": char,
            }
        )
    # remove trailing spaces
    while normalized_transcript and normalized_transcript[-1]["char"] == " ":
        normalized_transcript.pop()

    return normalized_transcript


def tokenize_and_pad_text(
    word_transcript: list[dict[str, str | float]],
    no_whitespace_before_word: bool,
    text_tokenizer: SentencePieceProcessor,
    text_padding_id: int,
    end_of_text_padding_id: int,
    audio_tokenizer_frame_rate: float,
) -> list[int]:
    """
    Tokenize the word transcript of single speaker.
    Fill the appropriate frames with the tokens based on the word-level timestamps,
    and frames without tokens are filled with the padding token.
    """
    # assert single speaker
    speakers = [seg["speaker"] for seg in word_transcript]
    assert 0 <= len(set(speakers)) < 2, "Multiple speakers found in the transcript."

    # sort the word transcript by the start time
    word_transcript = sorted(word_transcript, key=lambda x: x["start"])

    # add whitespace to the beginning of each transcript word
    if not no_whitespace_before_word:
        # ensure that the first word has no whitespace before it
        word_transcript[0]["word"] = word_transcript[0]["word"].strip()
        for seg in word_transcript[1:]:
            seg["word"] = " " + seg["word"].strip()

    word_transcript = normalize_word_transcript(word_transcript)
    char_transcript = get_char_level_transcript(word_transcript)
    char_transcript = normalize_char_transcript(char_transcript, no_whitespace_before_word)
    token_transcript = get_token_level_transcript(char_transcript, text_tokenizer)

    # make tokenized ids with padding
    if token_transcript:
        num_frames = int((token_transcript[-1]["end"] + 1) * audio_tokenizer_frame_rate)
    else:
        num_frames = 0
    seconds_per_frame = 1 / audio_tokenizer_frame_rate
    token_ids = [text_padding_id] * num_frames
    token_count = 0
    for seg in token_transcript:
        frame_index = int(seg["start"] // seconds_per_frame)
        try:
            # find the next padding index to insert the token
            while token_ids[frame_index] != text_padding_id:
                frame_index += 1
        except IndexError:
            warnings.warn(  # noqa: B028
                "frames are dropped due to the insufficient number of frames."
            )
            break
        token_ids[frame_index] = text_tokenizer.piece_to_id(seg["token"])
        token_count += 1
        if frame_index > 0 and token_ids[frame_index - 1] == text_padding_id:
            # insert end_of_text_padding_id
            token_ids[frame_index - 1] = end_of_text_padding_id
    return token_ids


processor: SentencePieceProcessor | None = None


def process_dialogue(worker_id: int, data: tuple[str, argparse.Namespace]):
    """Process a single dialogue by tokenizing its word-level transcript.

    Args:
        worker_id: ID of the worker process (not used but required by execute_data_processing).
        data: Tuple containing (dialogue_name, args).
    """
    dialogue_name, args = data

    # initialize processor once per process
    global processor
    if processor is None:
        processor = SentencePieceProcessor(
            hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
        )

    # load word-level transcript
    transcript_path = Path(args.word_transcript_dir) / f"{dialogue_name}.json"
    with open(transcript_path) as f:
        word_transcript = json.load(f)

    # tokenize text
    word_transcript_A = [seg for seg in word_transcript if seg["speaker"] == "A"]
    token_ids_A = tokenize_and_pad_text(
        word_transcript=word_transcript_A,
        no_whitespace_before_word=args.no_whitespace_before_word,
        text_tokenizer=processor,
        text_padding_id=args.text_padding_id,
        end_of_text_padding_id=args.end_of_text_padding_id,
        audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
    )
    word_transcript_B = [seg for seg in word_transcript if seg["speaker"] == "B"]
    token_ids_B = tokenize_and_pad_text(
        word_transcript=word_transcript_B,
        no_whitespace_before_word=args.no_whitespace_before_word,
        text_tokenizer=processor,
        text_padding_id=args.text_padding_id,
        end_of_text_padding_id=args.end_of_text_padding_id,
        audio_tokenizer_frame_rate=args.audio_tokenizer_frame_rate,
    )

    # save the tokenized text
    output_path = Path(args.output_dir) / f"{dialogue_name}.npz"
    try:
        np.savez_compressed(output_path, A=token_ids_A, B=token_ids_B)
    except Exception as e:
        print(f"Error in saving {output_path}: {e}")
        output_path.unlink(missing_ok=True)


def main(args):
    word_transcript_dir = Path(args.word_transcript_dir)
    output_dir = Path(args.output_dir)

    dialogue_names = sorted(p.stem for p in word_transcript_dir.glob("*.json"))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        tokenized_dialogue_names = [p.stem for p in output_dir.glob("*.npz")]
        print(f"Skipping {len(tokenized_dialogue_names)} already tokenized dialogues.")
        dialogue_names = sorted(set(dialogue_names) - set(tokenized_dialogue_names))

    # Create dataset iterator that yields (dialogue_name, args) tuples
    dataset = ((name, args) for name in dialogue_names)

    # Execute data processing using the utility function
    execute_data_processing(
        dataset=dataset,
        process_func=process_dialogue,
        num_workers=args.num_workers,
        data_count=len(dialogue_names),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize word-level transcripts using a text tokenizer."
    )
    parser.add_argument(
        "--word_transcript_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the transcripts with word level timestamps. "
            "Each file should contain a list of dictionaries "
            "(`{{'speaker': str, 'start': float, 'end': float, 'word': str}}`) "
            "and the filename should be the same as the corresponding audio file."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
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
        type=int,
        default=12.5,
        help="Frame rate for the audio tokenizer.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument("--resume", action="store_true", help="Resume tokenization.")

    args = parser.parse_args()

    main(args)
