import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders

from .utils import execute_data_processing


def ceil(x, y):
    return int(-(-x // y))


def tokenize_audio(
    wav: torch.Tensor,
    mimi: MimiModel,
    audio_chunk_size: int,
) -> torch.LongTensor:
    """
    Tokenize the audio of a single channel.
    """
    assert wav.dim() == 1, f"Expected 1D tensor, got {wav.dim()}D tensor."

    wav_chunk_size = audio_chunk_size * mimi.sample_rate
    num_chunks = ceil(wav.shape[0], wav_chunk_size)
    device = next(mimi.parameters()).device

    list_of_audio_ids = []
    for i in range(num_chunks):
        wav_chunk = wav[i * wav_chunk_size : (i + 1) * wav_chunk_size]
        with torch.no_grad():
            list_of_audio_ids.append(
                mimi.encode(wav_chunk.reshape(1, 1, -1).to(device)).cpu()  # [B=1, K=8, T_chunk]
            )
    audio_ids = torch.cat(list_of_audio_ids, dim=-1)  # [B=1, K=8, T]
    audio_ids = audio_ids[0]  # [K=8, T]

    num_frames = ceil(wav.shape[-1], (mimi.sample_rate / mimi.frame_rate))
    assert audio_ids.shape == (
        mimi.num_codebooks,
        num_frames,
    ), f"{audio_ids.shape} != ({mimi.num_codebooks}, {num_frames})"
    return audio_ids


mimi: MimiModel | None = None


def process_dialogue(worker_id: int, audio_path: Path, args: argparse.Namespace) -> None:
    """Process a single dialogue: load audio, tokenize, and save.

    Args:
        worker_id: Worker ID, used to determine GPU device
        audio_path: Path to the audio file to process
        args: Command line arguments containing configuration
    """
    # Initialize mimi for this worker
    global mimi
    if mimi is None:
        num_devices = torch.cuda.device_count()
        device_id = worker_id % num_devices
        torch.cuda.set_device(device_id)
        mimi = loaders.get_mimi(
            filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
            device="cuda",
        )

    wavs, sr = torchaudio.load(audio_path)
    assert wavs.shape[0] == 2, f"Expected stereo audio, got {wavs.shape[0]} channels."
    resampler = torchaudio.transforms.Resample(sr, mimi.sample_rate).to("cuda")
    wavs = resampler(wavs.to("cuda"))

    # Tokenize audio
    audio_ids_A = tokenize_audio(wavs[0], mimi, args.audio_chunk_size)
    audio_ids_B = tokenize_audio(wavs[1], mimi, args.audio_chunk_size)

    # Save tokenized audio
    dialogue_name = audio_path.stem
    output_path = Path(args.output_dir) / f"{dialogue_name}.npz"
    try:
        np.savez_compressed(output_path, A=audio_ids_A.numpy(), B=audio_ids_B.numpy())
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")
        output_path.unlink(missing_ok=True)


def main(args):
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    # Collect dialogue names from both .wav and .flac files
    audio_files = list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.flac"))

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        tokenized_dialogues = {p.stem for p in output_dir.glob("*.npz")}
        print(f"Skipping {len(tokenized_dialogues)} already tokenized dialogues.")
        audio_files = [f for f in audio_files if f.stem not in tokenized_dialogues]

    print(f"Processing {len(audio_files)} dialogues using {args.num_workers} workers.")

    # Use execute_data_processing for parallel execution with partial to pass args
    execute_data_processing(
        dataset=iter(sorted(audio_files)),
        process_func=partial(process_dialogue, args=args),
        num_workers=args.num_workers,
        data_count=len(audio_files),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tokenize audio files using a pretrained audio tokenizer."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the stereo audio files (wav or flac). "
            "Left and right channels should be the audio of speaker A and B respectively. "
            "and filenames should be the same as the dialogue names in the word transcript directory."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the tokenized data.",
    )
    parser.add_argument(
        "--audio_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face Hub repository for the audio tokenizer.",
    )
    parser.add_argument(
        "--audio_tokenizer_name",
        type=str,
        default="tokenizer-e351c8d8-checkpoint125.safetensors",
        help="Model name for the audio tokenizer.",
    )

    parser.add_argument(
        "--audio_chunk_size",
        type=int,
        default=1200,
        help="Split audio into chunks of this size (seconds) to fit into cuda memory.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument("--resume", action="store_true", help="Resume tokenization.")
    args = parser.parse_args()

    main(args)
