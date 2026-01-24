import argparse
import multiprocessing as mp
import os
from glob import glob

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm


def decode_text(text_tokens: np.ndarray, tokenizer: SentencePieceProcessor) -> str:
    """
    Decode text tokens.
    Args:
        tokens (np.ndarray): Text tokens. Shape: (seq_len,)
        tokenizer (SentencePieceProcessor): SentencePiece tokenizer.
    Returns:
        str: Decoded text.
    """

    text = tokenizer.decode(text_tokens.tolist())
    return text


def decode_audio(audio_tokens: np.ndarray, mimi: loaders.MimiModel) -> np.ndarray:
    """
    Decode audio tokens.
    Args:
        audio_tokens (np.ndarray): Audio tokens. Shape: (K, T).
            K: Number of audio codebooks, T: Number of audio tokens.
        mimi (loaders.MimiModel): Mimi model.
    Returns:
        np.ndarray: Decoded audio. Shape: (C=2, wav_len).
            C: Number of audio channels, wav_len: Length of the audio waveform.
    """
    K, T = audio_tokens.shape
    assert K // 2 == mimi.num_codebooks, (
        f"Number of codebooks mismatch: {K}//2 != {mimi.num_codebooks}"
    )

    device = next(mimi.parameters()).device
    # split the audio tokens to 2 channels
    tokens = np.stack(np.split(audio_tokens, 2, axis=0), axis=0)  # (2, K/2, T)
    with torch.no_grad():
        # use batch dimension of mimi as channel dimension
        wavs = mimi.decode(torch.from_numpy(tokens).to(device=device)).cpu().numpy()
        # wavs: (2, 1, wav_len)
    wav = wavs.squeeze(1)  # (2, wav_len)
    return wav


def decode_tokens(rank: int, list_of_tokens_path: list[str], args: argparse.Namespace):
    """
    Decode tokens.
    Args:
        rank (int): Rank of the process.
        list_of_tokens_path (list[str]): List of paths to the token files to decode.
            each file should be `*.npy` file.
        args (argparse.Namespace): Arguments.
    """
    # Load the text tokenizer
    text_tokenizer = SentencePieceProcessor(  # noqa: F841
        hf_hub_download(args.text_tokenizer_repo, args.text_tokenizer_name)
    )

    # Load the audio tokenizer
    mimi = loaders.get_mimi(
        filename=hf_hub_download(args.audio_tokenizer_repo, args.audio_tokenizer_name),
        device=torch.device("cuda", rank),
    )

    for tokens_path in tqdm(list_of_tokens_path, desc=f"Rank {rank}"):
        tokens = np.load(tokens_path)  # (1+K, T)

        text_tokens: np.ndarray = tokens[0]
        audio_tokens = tokens[1:]

        text_tokens = text_tokens[~np.isin(text_tokens, [0, 3])]
        text = decode_text(text_tokens, text_tokenizer)  # str
        audio = decode_audio(audio_tokens, mimi)  # (2, wav_len)

        base_name = os.path.basename(tokens_path).replace(".npy", "")

        # save the text
        output_txt_path = os.path.join(args.output_dir, f"{base_name}.txt")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # save the audio
        output_wav_path = os.path.join(args.output_dir, f"{base_name}.wav")
        sf.write(output_wav_path, audio.astype(np.float32).T, samplerate=mimi.sample_rate)


def main(args):
    # Get the list of token files
    list_of_tokens_path = glob(os.path.join(args.tokens_dir, "*.npy"))

    num_files = len(list_of_tokens_path)
    print(f"Number of token files: {num_files}")

    # Split the list of token files
    file_indices_per_rank = np.array_split(np.arange(num_files), args.num_workers)

    # Make the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Start the processes
    processes = []
    for rank, file_indices in enumerate(file_indices_per_rank):
        list_of_tokens_path_per_rank = [list_of_tokens_path[i] for i in file_indices]
        p = mp.Process(target=decode_tokens, args=(rank, list_of_tokens_path_per_rank, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens_dir",
        type=str,
        required=True,
        help="Directory containing the token files to decode. Each file should be `*.npy` file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the decoded output"
    )
    parser.add_argument(
        "--text_tokenizer_repo",
        type=str,
        default="kyutai/moshiko-pytorch-bf16",
        help="Hugging Face Hub repository for the text tokenizer.",
    )
    parser.add_argument(
        "--text_tokenizer_name",
        type=str,
        default="tokenizer_spm_32k_3.model",
        help="Model name of the text tokenizer.",
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
        help="Model name of the audio tokenizer.",
    )

    parser.add_argument(
        "--num_workers", type=int, default=1, help="Number of workers for multiprocessing."
    )

    args = parser.parse_args()

    main(args)
