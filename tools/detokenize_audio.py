import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from moshi.models import MimiModel, loaders

from .utils import execute_data_processing


def detokenize_audio(
    audio_ids: torch.LongTensor,
    mimi: MimiModel,
    audio_chunk_size: int,
) -> torch.Tensor:
    """
    Detokenize audio tokens to waveform.

    Args:
        audio_ids: Audio tokens of shape [K=8, T]
        mimi: MimiModel instance
        audio_chunk_size: Size of audio chunks in seconds

    Returns:
        Waveform tensor of shape [num_samples]
    """
    assert audio_ids.dim() == 2, f"Expected 2D tensor, got {audio_ids.dim()}D tensor."
    assert audio_ids.shape[0] == mimi.num_codebooks, (
        f"Expected {mimi.num_codebooks} codebooks, got {audio_ids.shape[0]}"
    )

    # Calculate chunk size in frames
    frame_chunk_size = int(audio_chunk_size * mimi.frame_rate)
    num_chunks = int(np.ceil(audio_ids.shape[1] / frame_chunk_size))
    device = next(mimi.parameters()).device

    list_of_wavs = []
    for i in range(num_chunks):
        audio_ids_chunk = audio_ids[:, i * frame_chunk_size : (i + 1) * frame_chunk_size]
        # Add batch dimension
        audio_ids_chunk = audio_ids_chunk.unsqueeze(0).to(device)  # [B=1, K=8, T_chunk]

        with torch.no_grad():
            wav_chunk = mimi.decode(audio_ids_chunk)  # [B=1, C=1, num_samples]
        list_of_wavs.append(wav_chunk.cpu())

    wav = torch.cat(list_of_wavs, dim=-1)  # [B=1, C=1, num_samples]
    wav = wav[0, 0]  # [num_samples]

    return wav


mimi: MimiModel | None = None


def process_tokenized_file(worker_id: int, npz_path: Path, args: argparse.Namespace) -> None:
    """Process a single tokenized file: load tokens, detokenize, and save audio.

    Args:
        worker_id: Worker ID, used to determine GPU device
        npz_path: Path to the .npz file containing tokenized audio
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

    # Load tokenized audio
    try:
        data = np.load(npz_path)
        audio_ids_A = torch.from_numpy(data["A"])
        audio_ids_B = torch.from_numpy(data["B"])
    except Exception as e:
        print(f"Failed to load {npz_path}: {e}")
        return

    # Detokenize audio for both channels
    wav_A = detokenize_audio(audio_ids_A, mimi, args.audio_chunk_size)
    wav_B = detokenize_audio(audio_ids_B, mimi, args.audio_chunk_size)

    # Stack to create stereo audio
    wavs = torch.stack([wav_A, wav_B], dim=0)  # [C=2, num_samples]

    # Save audio
    dialogue_name = npz_path.stem
    output_path = Path(args.output_dir) / f"{dialogue_name}.{args.output_format}"
    try:
        torchaudio.save(
            output_path,
            wavs,
            sample_rate=mimi.sample_rate,
            encoding=args.encoding,
            bits_per_sample=args.bits_per_sample,
        )
    except Exception as e:
        print(f"Failed to save {output_path}: {e}")
        output_path.unlink(missing_ok=True)


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Determine if input is a file or directory
    if input_path.is_file():
        if input_path.suffix != ".npz":
            raise ValueError(f"Input file must be .npz format, got {input_path.suffix}")
        npz_files = [input_path]
    elif input_path.is_dir():
        npz_files = list(input_path.glob("*.npz"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        detokenized_files = {p.stem for p in output_dir.glob(f"*.{args.output_format}")}
        print(f"Skipping {len(detokenized_files)} already detokenized files.")
        npz_files = [f for f in npz_files if f.stem not in detokenized_files]

    print(f"Processing {len(npz_files)} files using {args.num_workers} workers.")

    # Use execute_data_processing for parallel execution
    execute_data_processing(
        dataset=iter(sorted(npz_files)),
        process_func=partial(process_tokenized_file, args=args),
        num_workers=args.num_workers,
        data_count=len(npz_files),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detokenize audio tokens back to audio files.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a .npz file or directory containing .npz files with tokenized audio.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory to save the detokenized audio files.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="flac",
        choices=["flac", "wav", "ogg", "mp3"],
        help="Output audio format (default: flac).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default=None,
        help="Audio encoding (e.g., PCM_S, PCM_U, PCM_F). Default depends on format.",
    )
    parser.add_argument(
        "--bits_per_sample",
        type=int,
        default=None,
        help="Bits per sample for PCM encoding (e.g., 16, 24, 32).",
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
        help="Process audio in chunks of this size (seconds) to fit into cuda memory.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume detokenization, skipping already processed files.",
    )
    args = parser.parse_args()

    main(args)
