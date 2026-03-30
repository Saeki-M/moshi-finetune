import argparse
import json
import logging
import os

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    MoshiForConditionalGeneration,
    MoshiForFinetuning,
)
from utils import (
    DataCollator,
    preprocess_function,
    set_mpi_env_vars,
    undelay_tokens,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference script for prompted dialogue continuation with finetuned Moshi."
    )

    parser.add_argument(
        "--launcher",
        choices=["accelerate", "mpi"],
        default="accelerate",
        help="Launcher type to use for distributed inference",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the output"
    )
    parser.add_argument(
        "--eval_data_files",
        type=str,
        required=True,
        help="Pattern to the evaluation data files. Each file should be parquet file.",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help=(
            "Path to the directory containing the pre-trained Moshi model (`model.safetensors`) "
            "and config for initializing model (`init_moshi_lm_kwargs.json`)."
        ),
    )
    parser.add_argument(
        "--model_dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Model data type",
    )

    parser.add_argument(
        "--moshi_speakers",
        choices=["A", "B"],
        nargs="+",
        default=["A"],
        help="Speakers to use as the main stream",
    )
    parser.add_argument(
        "--model_user_stream",
        action="store_true",
        help=(
            "Whether the model includes the user's audio stream. "
            "If not set, user audio codebooks will be set to zero during generation."
        ),
    )
    parser.add_argument(
        "--dataset_processing_workers",
        type=int,
        default=16,
        help="Number of workers to use for processing the dataset.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=".cache/huggingface/datasets",
        help="Directory to cache the datasets.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate on.",
    )
    parser.add_argument(
        "--example_length",
        type=int,
        default=None,
        help=(
            "Split the examples longer than this length into smaller examples. "
            "If None, no splitting will be done. Useful for long examples."
        ),
    )

    # Generation parameters
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=125,
        help=(
            "Length of the prompt. A sub-sequence from the beginning to this length "
            "of each example is input to the model as a prompt. Samples shorter than "
            "this will be ignored."
        ),
    )
    parser.add_argument(
        "--generation_length",
        type=int,
        default=250,
        help="Maximum length of the generated sequence.",
    )
    parser.add_argument(
        "--use_sampling",
        action="store_true",
        default=True,
        help="Use sampling for generation.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k for sampling. Set to 0 for no top-k.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
        help="Top-p for sampling. Set to 0 for no top-p.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )

    args = parser.parse_args()

    # post process args
    if args.launcher == "mpi":
        set_mpi_env_vars()

    return args


def main():
    args = parse_args()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load the model
    logger.info(f"Loading Moshi model from {args.model_dir}")
    moshi_lm = MoshiForFinetuning.from_pretrained(
        args.model_dir,
        device=accelerator.device,
        dtype=getattr(torch, args.model_dtype),
    )

    moshi_cg = MoshiForConditionalGeneration(moshi_lm=moshi_lm)

    # Load the dataset
    with accelerator.main_process_first():
        logger.info(f"Loading eval dataset from {args.eval_data_files}")
        eval_dataset = load_dataset(
            "parquet",
            split="evaluation",
            data_files={"evaluation": args.eval_data_files},
            cache_dir=args.dataset_cache_dir,
        )

    # Preprocess the dataset
    preprocessing_kwargs = {
        "speakers": args.moshi_speakers,
        "max_length": args.example_length,  # cut the examples to this length
        "min_length": args.prompt_length,  # ensure that each example has at least this length
        "delays": moshi_lm.delays,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
        "model_user_stream": args.model_user_stream,
    }
    with accelerator.main_process_first():
        eval_dataset = eval_dataset.map(
            preprocess_function,
            remove_columns=eval_dataset.column_names,
            batched=True,
            num_proc=args.dataset_processing_workers,
            fn_kwargs=preprocessing_kwargs,
            desc="Preprocessing dataset",
        )
    eval_dataset = eval_dataset.add_column("example_id", list(range(len(eval_dataset))))

    if args.num_examples is None:
        args.num_examples = len(eval_dataset)

    global_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
    local_num_steps = -(-args.num_examples // global_batch_size)

    # Cutoff the dataset if needed
    eval_dataset = eval_dataset.select(range(args.num_examples))

    # Create the dataloader
    data_collator = DataCollator(zero_token_id=moshi_lm.zero_token_id)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.per_device_eval_batch_size,
        collate_fn=data_collator,
    )

    # prepare
    eval_dataloader = accelerator.prepare(eval_dataloader)
    assert len(eval_dataloader) == local_num_steps, f"{len(eval_dataloader)=} != {local_num_steps=}"

    sampling_params = {
        "use_sampling": args.use_sampling,
        "temp": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }

    # save config
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # make directory for results
    result_dir = os.path.join(args.output_dir, "generated_tokens")
    os.makedirs(result_dir, exist_ok=True)

    # Start the evaluation
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
    logger.info(
        f"  Total batch size = {args.per_device_eval_batch_size * accelerator.num_processes}"
    )
    logger.info(f"  Num steps = {local_num_steps}")
    logger.info(f"  Generation length = {args.generation_length}")
    logger.info(f"  Use sampling = {args.use_sampling}")
    if args.use_sampling:
        logger.info(f"  Sampling parameters = {sampling_params}")
    logger.info(f"  Model user stream = {args.model_user_stream}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(local_num_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True
    )

    moshi_lm.eval()

    for batch in eval_dataloader:
        batch = batch.to(accelerator.device)
        prompt_tokens = batch.input_ids[..., : args.prompt_length]
        gen_tokens = moshi_cg.generate(
            prompt_tokens=prompt_tokens,
            generation_length=args.generation_length,
            text_sampling_params=sampling_params,
            audio_sampling_params=sampling_params,
            model_user_stream=args.model_user_stream,
        )
        # gen_tokens = torch.cat([prompt_tokens, gen_tokens], dim=-1).cpu().numpy()
        gen_tokens = undelay_tokens(gen_tokens, moshi_lm.delays)
        gen_tokens = gen_tokens.cpu().numpy()

        # Save the generated tokens
        for i in range(batch.input_ids.shape[0]):
            example_id = batch.example_ids[i]
            tokens = gen_tokens[i]
            np.save(os.path.join(result_dir, f"{example_id}.npy"), tokens)

        progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":
    main()
