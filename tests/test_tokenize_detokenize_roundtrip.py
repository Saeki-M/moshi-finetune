"""Test tokenization and detokenization round trip."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download
from sentencepiece import SentencePieceProcessor

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.detokenize_text import detokenize_token_ids
from tools.tokenize_text import tokenize_and_pad_text


@pytest.fixture(scope="module")
def text_tokenizer():
    """Load the text tokenizer."""
    tokenizer_path = hf_hub_download("kyutai/moshiko-pytorch-bf16", "tokenizer_spm_32k_3.model")
    return SentencePieceProcessor(tokenizer_path)


tokenizer_params = {
    "text_padding_id": 3,
    "end_of_text_padding_id": 0,
    "audio_tokenizer_frame_rate": 12.5,
}


def test_tokenize_roundtrip(text_tokenizer, tmp_path: Path):
    """Test roundtrip for SpokenWOZ sample data."""
    text_dir = Path("data/spokenwoz_sample/text")

    for json_file in text_dir.glob("*.json"):
        dialogue_name = json_file.stem

        # Load original transcript
        with open(json_file) as f:
            original_transcript = json.load(f)

        # Tokenize both speakers
        tokenized_speakers = {}
        for speaker in ["A", "B"]:
            word_transcript = [seg for seg in original_transcript if seg["speaker"] == speaker]
            tokenized_speakers[speaker] = tokenize_and_pad_text(
                word_transcript=word_transcript,
                no_whitespace_before_word=False,
                text_tokenizer=text_tokenizer,
                **tokenizer_params,
            )

        # Save tokenized data to temp directory
        tokenized_file = tmp_path / f"{dialogue_name}.npz"
        np.savez_compressed(tokenized_file, **tokenized_speakers)

        # Load tokenized data
        tokenized_data = np.load(tokenized_file)

        # Detokenize both speakers
        detokenized_transcript = []
        for speaker in ["A", "B"]:
            token_ids = tokenized_data[speaker].tolist()
            detokenized = detokenize_token_ids(
                token_ids=token_ids,
                text_tokenizer=text_tokenizer,
                **tokenizer_params,
            )
            # Add speaker labels
            for seg in detokenized:
                seg["speaker"] = speaker
            detokenized_transcript.extend(detokenized)

        # Sort by start time
        detokenized_transcript.sort(key=lambda x: x["start"])

        # Compare transcripts for both speakers
        for speaker in ["A", "B"]:
            original_text = " ".join(
                seg["word"].strip() for seg in original_transcript if seg["speaker"] == speaker
            )
            detokenized_text = " ".join(
                seg["word"].strip() for seg in detokenized_transcript if seg["speaker"] == speaker
            )

            # Assert exact match
            assert original_text == detokenized_text, (
                f"Speaker {speaker} mismatch in {dialogue_name}:\n"
                f"  Original: {original_text}\n"
                f"  Detokenized: {detokenized_text}"
            )


if __name__ == "__main__":
    # Run tests directly
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
