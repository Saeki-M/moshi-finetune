import argparse
import collections
import glob
import itertools
import json
import logging
import math
import os
from datetime import timedelta

import deepspeed
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DummyOptim,
    DummyScheduler,
    InitProcessGroupKwargs,
    set_seed,
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.moshi_for_finetuning import MoshiForFinetuning
from utils import (
    Batch,
    DataCollator,
    preprocess_function,
    set_mpi_env_vars,
)

logger = get_logger(__name__)


# Parsing input arguments
def setup_argparser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--launcher",
        choices=["accelerate", "mpi"],
        default="accelerate",
        help="Launcher type to use for distributed training",
    )
    parser.add_argument(
        "--use_deepspeed",
        action="store_true",
        help="Whether to use deepspeed for training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--train_data_files",
        type=str,
        nargs="+",
        required=True,
        help="Patterns to the training data files. Each file should be parquet file.",
    )
    parser.add_argument(
        "--eval_data_files",
        type=str,
        nargs="+",
        default=None,
        help="Patterns to the evaluation data files. Each file should be parquet file.",
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
        help="Whether to train the user's audio stream",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of the input sequence",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=None,
        help="Minimum length of the input sequence",
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
        "--process_group_timeout",
        type=int,
        default=3600,
        help="Timeout for the process group initialization (in seconds).",
    )

    parser.add_argument(
        "--deepspeed_config_file",
        type=str,
        default=None,
        help="Path to a DeepSpeed config file. Required if `use_deepspeed` is set.",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action="store_true",
        help="Whether to use activation checkpointing in the model.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--tempformer_learning_rate",
        type=float,
        default=3e-5,
        help="The initial learning rate for the Temporal Transformer model.",
    )
    parser.add_argument(
        "--depformer_learning_rate",
        type=float,
        default=3e-5,
        help="The initial learning rate for the Depth Transformer model.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        help="Weight decay to apply.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Stop training after this number of steps.",
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the learning rate scheduler.",
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run evaluation every X updates steps.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="Log every X updates steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=1,
        help="Number of most recent step checkpoints to keep. Older ones are deleted. Set to 0 to keep all.",
    )
    parser.add_argument(
        "--report_to",
        choices=["wandb"],
        default=None,
        help='The integration to report the results and logs to. Currently only "wandb" is supported.',
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="moshi-finetuning",
        help="The name of the project to which the training run belongs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )

    parser.add_argument(
        "--parameters_to_finetune",
        choices=["all", "tempformer", "depformer", "text_embedding"],
        default="all",
        help="Which parameters to finetune.",
    )
    parser.add_argument(
        "--text_padding_loss_weight",
        type=float,
        default=0.5,
        help="Weight for the padding loss in the text loss.",
    )
    parser.add_argument(
        "--semantic_loss_weight",
        type=float,
        default=100.0,
        help="Weight for the semantic loss in the audio loss.",
    )
    parser.add_argument(
        "--acoustic_loss_weight",
        type=float,
        default=1.0,
        help="Weight for the acoustic loss in the audio loss.",
    )


def postprocess_args(args: argparse.Namespace):
    # post process
    if args.launcher == "accelerate":
        args.use_deepspeed = os.environ.get("ACCELERATE_USE_DEEPSPEED") == "true"
        args.deepspeed_config_file = os.environ.get("ACCELERATE_DEEPSPEED_CONFIG_FILE")
        if args.use_deepspeed and "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
            # unset the mpirun's variables for deepspeed to work
            # ref: https://github.com/Lightning-AI/pytorch-lightning/issues/13567
            del os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    elif args.launcher == "mpi":
        _ = set_mpi_env_vars()
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"  # set manually here
    else:
        raise ValueError(f"Unknown launcher: {args.launcher}")

    # check the deepspeed settings
    if not args.use_deepspeed:
        raise NotImplementedError("Only DeepSpeed is supported for now.")
    if args.deepspeed_config_file is None:
        raise ValueError(
            "DeepSpeed config file (`--deepspeed_config_file`) is required when "
            "`use_deepspeed` is set."
        )

    # check the dataset files
    if not args.train_data_files:
        raise ValueError("No training data files provided.")

    # Expand glob patterns and verify existence for train files
    expanded_train: list[str] = []
    for p in args.train_data_files:
        expanded_train.extend(sorted(glob.glob(p)))
    if not expanded_train:
        raise FileNotFoundError(
            f"No training files matched the provided patterns: {args.train_data_files}"
        )
    args.train_data_files = expanded_train

    if args.eval_data_files is not None:
        assert isinstance(args.eval_steps, int), (
            "eval_steps is required when eval_data_files is provided."
        )
        # Expand glob patterns and verify existence for eval files
        expanded_eval: list[str] = []
        for p in args.eval_data_files:
            expanded_eval.extend(sorted(glob.glob(p)))
        if not expanded_eval:
            raise FileNotFoundError(
                f"No evaluation files matched the provided patterns: {args.eval_data_files}"
            )
        args.eval_data_files = expanded_eval

    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), (
            f"Checkpoint file not found: {args.resume_from_checkpoint}"
        )
        resume_config_path = os.path.join(
            os.path.dirname(args.resume_from_checkpoint), "config.json"
        )
        assert os.path.exists(resume_config_path), f"Config file not found: {resume_config_path}"
        prev_config = json.load(open(resume_config_path))
        # check consistency of the config
        different_keys = ["output_dir", "max_train_steps", "resume_from_checkpoint"]
        missmatch_args = []
        for key, value in vars(args).items():
            if key in different_keys:
                continue
            if prev_config[key] != value:
                missmatch_args.append(f"{key}: {prev_config[key]} != {value}")
        if missmatch_args:
            raise ValueError(f"Missmatch in the config: {missmatch_args}")

        # restore the previous run_id
        args.run_id_to_resume = prev_config.get("run_id")
    else:
        args.run_id_to_resume = None


def get_parameters(moshi_lm: MoshiForFinetuning, parameter_name: str):
    if parameter_name == "all":
        return moshi_lm.parameters()
    elif parameter_name == "tempformer":
        return itertools.chain(
            moshi_lm.emb.parameters(),
            moshi_lm.text_emb.parameters(),
            moshi_lm.transformer.parameters(),
            moshi_lm.out_norm.parameters(),
            moshi_lm.text_linear.parameters(),
        )
    elif parameter_name == "depformer":
        return itertools.chain(
            moshi_lm.depformer_in.parameters(),
            moshi_lm.depformer_emb.parameters(),
            moshi_lm.depformer_text_emb.parameters(),
            moshi_lm.depformer.parameters(),
            moshi_lm.linears.parameters(),
        )
    elif parameter_name == "text_embedding":
        return itertools.chain(
            moshi_lm.text_emb.parameters(),
            moshi_lm.depformer_text_emb.parameters(),
        )
    else:
        raise ValueError(f"Unknown parameter name: {parameter_name}")


def tempformer_forward(
    moshi_lm: MoshiForFinetuning, batch: Batch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 1 Encode text
    text_emb = moshi_lm.text_emb(batch.input_ids[:, 0])  # [B, F, 4096]

    # 2 Encode audio
    audio_emb = None
    for acb_index in range(moshi_lm.num_audio_codebooks):
        audio_emb_ = moshi_lm.emb[acb_index](batch.input_ids[:, moshi_lm.audio_offset + acb_index])
        audio_emb = audio_emb_ if audio_emb is None else audio_emb + audio_emb_

    # 3 Feed embeddings to temporal transformer
    tempformer_input = text_emb + audio_emb
    tempformer_out = moshi_lm.transformer(
        tempformer_input, attention_mask=batch.text_attention_mask
    )
    if moshi_lm.out_norm:
        tempformer_out = moshi_lm.out_norm(tempformer_out)
    text_logits = moshi_lm.text_linear(tempformer_out)  # [B, F, text_card(32000))]

    # 4 Compute loss
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    text_logits = text_logits.float()
    # Shift so that tokens < n predict n
    text_logits = text_logits[..., :-1, :].contiguous()
    text_labels = batch.labels[:, 0, 1:].contiguous()

    text_losses = F.cross_entropy(
        input=text_logits.view(-1, moshi_lm.text_card),
        target=text_labels.view(-1),
        ignore_index=moshi_lm.zero_token_id,
        reduction="none",
    ).view(text_labels.size())
    text_accuracy = (text_logits.argmax(-1) == text_labels).float()

    assert text_labels.shape == text_losses.shape, f"{text_labels.shape} != {text_losses.shape}"

    non_pad_indices = (
        (text_labels != moshi_lm.text_padding_token_id)
        & (text_labels != moshi_lm.zero_token_id)  # zero token is ignored
    )
    # we treat moshi_lm.end_of_text_padding_id as non-padding token
    pad_indices = text_labels == moshi_lm.text_padding_token_id

    result = {
        "non_pad_losses": text_losses[non_pad_indices],
        "pad_losses": text_losses[pad_indices],
        "non_pad_accuracy": text_accuracy[non_pad_indices].mean(),
        "pad_accuracy": text_accuracy[pad_indices].mean(),
    }
    return tempformer_out, result


def depformer_forward(
    moshi_lm: MoshiForFinetuning,
    batch: Batch,
    tempformer_out: torch.Tensor,  # [B, F, 4096]
    model_user_stream: bool,
) -> dict[str, torch.Tensor]:
    # only compute system stream if model_user_stream is False
    target_q = moshi_lm.dep_q if model_user_stream else 8

    # 1 Mapping tempformer's output to depformer's input
    depformer_inputs = []
    for acb_index in range(target_q):
        if moshi_lm.depformer_multi_linear:
            # use different linear layers for different audio codebooks
            depformer_input_ = moshi_lm.depformer_in[acb_index](tempformer_out[:, :-1])
        else:
            # use the same linear layer for all audio codebooks
            depformer_input_ = moshi_lm.depformer_in[0](tempformer_out[:, :-1])
        depformer_inputs.append(depformer_input_)
    depformer_input = torch.stack(depformer_inputs, dim=2)  # [B, F, target_q, 1024]

    # 2 Encode last token of each audio streams
    last_token_embs = []
    last_token_emb_ = moshi_lm.depformer_text_emb(batch.input_ids[:, 0, 1:])  # text token
    last_token_embs.append(last_token_emb_)
    for acb_index in range(target_q - 1):
        last_token_emb_ = moshi_lm.depformer_emb[acb_index](
            batch.input_ids[:, moshi_lm.audio_offset + acb_index, 1:]  # audio token
        )
        last_token_embs.append(last_token_emb_)
    last_token_emb = torch.stack(last_token_embs, dim=2)

    # 3 Feed embeddings to rq-transformer
    depformer_input = depformer_input + last_token_emb  # [B, F, target_q, 1024]
    # flatten batch_size and num_frames
    depformer_input = torch.flatten(depformer_input, 0, 1)
    depformer_out = moshi_lm.depformer(depformer_input)  # [B x F, 16, 1024]

    audio_logits = []
    for acb_index in range(target_q):
        if moshi_lm.depformer_multi_linear:
            audio_logits_ = moshi_lm.linears[acb_index](depformer_out[:, acb_index])  # [BxF, 2048]
        else:
            audio_logits_ = moshi_lm.linears[0](depformer_out[:, acb_index])
        audio_logits.append(audio_logits_)
    audio_logits = torch.stack(audio_logits, dim=1)
    audio_logits = audio_logits.float()  # [B x F, target_q, card]
    # >>> depformer_logits.shape
    # torch.Size([batch_size * num_frames, dep_q, card])

    # 4 Compute loss
    if model_user_stream:
        audio_labels = batch.labels[:, 1:, 1:].transpose(1, 2).contiguous()
    else:
        audio_labels = batch.labels[:, 1 : 1 + target_q, 1:].transpose(1, 2).contiguous()

    # >>> depformer_labels.shape
    # torch.Size([batch_size, num_frames, dep_q])

    audio_losses = F.cross_entropy(
        input=audio_logits.view(-1, moshi_lm.card),
        target=audio_labels.view(-1),
        ignore_index=moshi_lm.zero_token_id,
        reduction="none",
    ).view(audio_labels.size())
    audio_accuracy = (audio_logits.argmax(-1).view(audio_labels.shape) == audio_labels).float()

    assert audio_labels.shape == audio_losses.shape, f"{audio_labels.shape} != {audio_losses.shape}"
    assert audio_accuracy.shape == audio_labels.shape, (
        f"{audio_accuracy.shape} != {audio_labels.shape}"
    )

    result = {
        "semantic_losses": audio_losses[..., 0][audio_labels[..., 0] != moshi_lm.zero_token_id],
        "acoustic_losses": audio_losses[..., 1:8][audio_labels[..., 1:8] != moshi_lm.zero_token_id],
        "semantic_accuracy": audio_accuracy[..., 0][
            audio_labels[..., 0] != moshi_lm.zero_token_id
        ].mean(),
        "acoustic_accuracy": audio_accuracy[..., 1:8][
            audio_labels[..., 1:8] != moshi_lm.zero_token_id
        ].mean(),
    }
    if model_user_stream:
        result.update(
            {
                "semantic_losses_user": audio_losses[..., 8][
                    audio_labels[..., 8] != moshi_lm.zero_token_id
                ],
                "acoustic_losses_user": audio_losses[..., 9:][
                    audio_labels[..., 9:] != moshi_lm.zero_token_id
                ],
                "semantic_accuracy_user": audio_accuracy[..., 8][
                    audio_labels[..., 8] != moshi_lm.zero_token_id
                ].mean(),
                "acoustic_accuracy_user": audio_accuracy[..., 9:][
                    audio_labels[..., 9:] != moshi_lm.zero_token_id
                ].mean(),
            }
        )

    return result


def forward(
    moshi_lm: MoshiForFinetuning,
    batch: Batch,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Forward pass of the Moshi model
    1. Temporal Transformer
    2. Depth Transformer
    3. Compute loss

    Args:
        args (argparse.Namespace): input arguments
        moshi_lm (MoshiForFinetuning): Moshi model
        batch (Batch): input batch
    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: final loss and logging values
    """
    log = {}

    # 1 Forward pass of Temporal Transformer
    tempformer_out, temp_result = tempformer_forward(moshi_lm, batch)
    text_loss = 0.0
    if temp_result["non_pad_losses"].size(0) > 0:
        text_loss += temp_result["non_pad_losses"].mean()
    if temp_result["pad_losses"].size(0) > 0:
        text_loss += temp_result["pad_losses"].mean() * args.text_padding_loss_weight
    log["loss/text_total"] = text_loss.detach()
    log["loss/text_non_pad"] = temp_result["non_pad_losses"].mean().detach()
    log["loss/text_pad"] = temp_result["pad_losses"].mean().detach()
    log["accuracy/text_non_pad"] = temp_result["non_pad_accuracy"].detach()
    log["accuracy/text_pad"] = temp_result["pad_accuracy"].detach()
    del temp_result

    # 2 Forward pass of Depth Transformer
    dep_result = depformer_forward(moshi_lm, batch, tempformer_out, args.model_user_stream)
    audio_weight = (
        dep_result["semantic_losses"].size(0) * args.semantic_loss_weight
        + dep_result["acoustic_losses"].size(0) * args.acoustic_loss_weight
    )
    if args.model_user_stream:
        audio_weight += (
            dep_result["semantic_losses_user"].size(0) * args.semantic_loss_weight
            + dep_result["acoustic_losses_user"].size(0) * args.acoustic_loss_weight
        )
    if audio_weight > 0:
        semantic_scale = args.semantic_loss_weight / audio_weight
        acoustic_scale = args.acoustic_loss_weight / audio_weight
    else:
        semantic_scale = 0.0
        acoustic_scale = 0.0
    audio_loss = (
        dep_result["semantic_losses"].sum() * semantic_scale
        + dep_result["acoustic_losses"].sum() * acoustic_scale
    )
    if args.model_user_stream:
        audio_loss += (
            dep_result["semantic_losses_user"].sum() * semantic_scale
            + dep_result["acoustic_losses_user"].sum() * acoustic_scale
        )
    log["loss/audio_total"] = audio_loss.detach()
    log["loss/audio_semantic"] = dep_result["semantic_losses"].mean().detach()
    log["loss/audio_acoustic"] = dep_result["acoustic_losses"].mean().detach()
    log["accuracy/audio_semantic"] = dep_result["semantic_accuracy"].detach()
    log["accuracy/audio_acoustic"] = dep_result["acoustic_accuracy"].detach()
    if args.model_user_stream:
        log["loss/audio_semantic_user"] = dep_result["semantic_losses_user"].mean().detach()
        log["loss/audio_acoustic_user"] = dep_result["acoustic_losses_user"].mean().detach()
        log["accuracy/audio_semantic_user"] = dep_result["semantic_accuracy_user"].detach()
        log["accuracy/audio_acoustic_user"] = dep_result["acoustic_accuracy_user"].detach()
    del dep_result

    # 3 Compute final loss
    total_loss = text_loss + audio_loss
    log["loss/total"] = total_loss.detach()

    return total_loss, log


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Finetune Moshi LM on a custom dataset")
    setup_argparser(parser)
    args = parser.parse_args()
    postprocess_args(args)

    accelerator_kwargs = {
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "kwargs_handlers": [
            InitProcessGroupKwargs(timeout=timedelta(seconds=args.process_group_timeout)),
        ],
    }

    if args.report_to:
        accelerator_kwargs.update(
            {
                "log_with": args.report_to,
                "project_dir": args.output_dir,
            }
        )

    if args.launcher != "accelerate":
        from accelerate import DeepSpeedPlugin

        accelerator_kwargs.update(
            {
                "deepspeed_plugin": DeepSpeedPlugin(
                    hf_ds_config=args.deepspeed_config_file,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
            }
        )

    accelerator = Accelerator(**accelerator_kwargs)
    args.num_processes = accelerator.num_processes

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
        save_dir=args.model_dir,
        device="cpu",  # accelerator will move the model to the correct device
        dtype=torch.float32,
    )

    # Set activation checkpointing
    if args.activation_checkpointing:
        assert os.environ.get("NO_TORCH_COMPILE", "1") == "1", "Not compatible with torch.compile"

        activation_checkpointing_kwargs = accelerator.deepspeed_plugin.hf_ds_config.config.get(
            "activation_checkpointing", {}
        )
        deepspeed.checkpointing.configure(
            mpu_=None, deepspeed_config=None, **activation_checkpointing_kwargs
        )
        moshi_lm.enable_activation_checkpointing(deepspeed.checkpointing.checkpoint)

    # Set active/frozen for the finetuning
    for param in moshi_lm.parameters():
        param.requires_grad = False
    for param in get_parameters(moshi_lm, args.parameters_to_finetune):
        param.requires_grad = True
    for name, param in moshi_lm.named_parameters():
        logger.info(f"{name}: {param.requires_grad}")

    # Load the dataset
    with accelerator.main_process_first():
        logger.info(f"Loading train dataset from {args.train_data_files}")
        train_dataset = load_dataset(
            "parquet",
            split="train",
            data_files={"train": args.train_data_files},
            cache_dir=args.dataset_cache_dir,
        )
    if args.eval_data_files is not None:
        with accelerator.main_process_first():
            logger.info(f"Loading eval dataset from {args.eval_data_files}")
            eval_dataset = load_dataset(
                "parquet",
                split="validation",
                data_files={"validation": args.eval_data_files},
                cache_dir=args.dataset_cache_dir,
            )
    else:
        eval_dataset = None

    # Preprocess the dataset
    preprocessing_kwargs = {
        "speakers": args.moshi_speakers,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "delays": moshi_lm.delays,
        "initial_token_ids": [moshi_lm.text_initial_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "padding_token_ids": [moshi_lm.text_padding_token_id]
        + [moshi_lm.initial_token_id] * moshi_lm.num_audio_codebooks,
        "zero_token_id": moshi_lm.zero_token_id,
        "model_user_stream": args.model_user_stream,
    }
    dataset_columns = train_dataset.column_names
    with accelerator.main_process_first():
        # only main process preprocesses the dataset, then others will use the resulted cache
        train_dataset = train_dataset.map(
            preprocess_function,
            remove_columns=dataset_columns,
            batched=True,
            num_proc=args.dataset_processing_workers,
            fn_kwargs=preprocessing_kwargs,
            desc="Preprocessing train dataset",
        )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                preprocess_function,
                remove_columns=dataset_columns,
                batched=True,
                num_proc=args.dataset_processing_workers,
                fn_kwargs=preprocessing_kwargs,
                desc="Preprocessing validation dataset",
            )

    data_collator = DataCollator(zero_token_id=moshi_lm.zero_token_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        collate_fn=data_collator,
        shuffle=True,
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
    else:
        eval_dataloader = None

    global_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    local_num_steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
    global_num_steps_per_epoch = local_num_steps_per_epoch * accelerator.num_processes
    local_num_steps = args.num_train_epochs * local_num_steps_per_epoch
    global_num_steps = args.num_train_epochs * global_num_steps_per_epoch

    # Prepare optimizer and learning rate scheduler
    param_groups = [
        {  # Temporal Transformer
            "params": get_parameters(moshi_lm, "tempformer"),
            "lr": args.tempformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {  # Depth Transformer
            "params": get_parameters(moshi_lm, "depformer"),
            "lr": args.depformer_learning_rate,
            "weight_decay": args.weight_decay,
        },
    ]

    optimizer = DummyOptim(
        param_groups,
        lr=args.tempformer_learning_rate,  # for accelerator to set deepspeed's lr
        weight_decay=args.weight_decay,  # for accelerator to set deepspeed's weight decay
    )
    # `defaults["lr"]` is used by accelerator to set max_lr of deepspeed's scheduler
    # Ref: Accelerator._prepare_deepspeed()
    optimizer.defaults = {
        "lr": [args.tempformer_learning_rate, args.depformer_learning_rate],
    }

    lr_scheduler_type = None
    if "scheduler" in accelerator.deepspeed_plugin.hf_ds_config.config:
        lr_scheduler_type = accelerator.deepspeed_plugin.hf_ds_config.config["scheduler"]["type"]
    if lr_scheduler_type is None:
        lr_scheduler = None
    elif lr_scheduler_type == "WarmupLR":
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=args.num_warmup_steps,
        )
    elif lr_scheduler_type == "WarmupDecayLR":
        lr_scheduler = DummyScheduler(
            optimizer=optimizer,
            warmup_num_steps=args.num_warmup_steps,
            total_num_steps=global_num_steps,
        )
    else:
        raise NotImplementedError(f"Unknown lr_scheduler_type: {lr_scheduler_type}")

    # Prepare everything with our `accelerator`.
    moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        moshi_lm, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Resume training
    current_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        current_steps = int(os.path.basename(args.resume_from_checkpoint).split("_")[1])
        starting_epoch = current_steps // local_num_steps_per_epoch

    # Initialize tracker
    config = vars(args)
    config["deepspeed_config"] = json.load(open(args.deepspeed_config_file))
    if args.report_to and accelerator.is_main_process:
        wandb_init_kwargs = {
            "name": os.path.basename(args.output_dir),
        }
        if args.run_id_to_resume is not None:
            wandb_init_kwargs.update(
                {
                    "resume": "must",
                    "id": args.run_id_to_resume,
                }
            )
        accelerator.init_trackers(
            project_name=args.project_name,
            config=config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )
        config["run_id"] = accelerator.get_tracker(name="wandb", unwrap=True).id  # for later resume
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # Start training
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {global_batch_size}"
    )
    logger.info(f"  Total optimization steps = {local_num_steps}")
    if args.resume_from_checkpoint:
        logger.info(f"  Resume from step {current_steps}")

    # Only show the progress bar once on each machine.
    pbar = tqdm(
        range(local_num_steps),
        initial=current_steps,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.resume_from_checkpoint and epoch == starting_epoch:
            # skip the first epoch if we resume from the middle
            # if accelerator.use_stateful_dataloader:
            #     active_dataloader = train_dataloader
            # else:
            num_batches_to_skip = (
                current_steps * args.gradient_accumulation_steps  # steps -> batches
            ) % local_num_steps_per_epoch
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, num_batches_to_skip
            )
        else:
            num_batches_to_skip = 0
            active_dataloader = train_dataloader

        logging_buffer = collections.defaultdict(list)

        for step, batch in enumerate(active_dataloader, start=num_batches_to_skip):
            moshi_lm.train()
            batch = batch.to(accelerator.device)
            # Forward pass
            total_loss, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
            for key, value in log.items():
                logging_buffer[f"training_{key}"].append(value)
            # Backward pass
            accelerator.backward(total_loss)
            # The following update steps are handled by deepseed's backward() in accelerator,
            # so we don't need to do them here manually
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(
                train_dataloader
            ) - 1:
                pbar.update(1)
                current_steps += 1

                # Log metrics
                if current_steps % args.logging_steps == 0:
                    lrs = {
                        "tempformer": f"{optimizer.param_groups[0]['lr']:.3e}",
                        "depformer": f"{optimizer.param_groups[1]['lr']:.3e}",
                    }
                    logger.info(
                        f"Epoch: {epoch}, "
                        f"Steps: {current_steps}, "
                        f"LRs: {lrs}, "
                        f"Loss: {total_loss.item():.5f} "
                        f"(text: {log['loss/text_total'].item():.5f}, "
                        f"audio: {log['loss/audio_total'].item():.5f})"
                    )
                    if args.report_to:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {
                                **{
                                    key: values.nanmean()
                                    for key, values in gathered_metrics.items()
                                },
                                "learning_rate/tempformer": optimizer.param_groups[0]["lr"],
                                "learning_rate/depformer": optimizer.param_groups[1]["lr"],
                            },
                            step=current_steps,
                        )
                    logging_buffer = collections.defaultdict(list)  # reset

                # Evaluate the model
                if args.eval_steps is not None and current_steps % args.eval_steps == 0:
                    logger.info("***** Running evaluation *****")
                    logger.info(f"  Num examples = {len(eval_dataset)}")
                    logger.info(f"  Batch size = {args.per_device_eval_batch_size}")
                    logger.info(f"  Num steps = {len(eval_dataloader)}")
                    eval_logging_buffer = collections.defaultdict(list)
                    moshi_lm.eval()
                    for batch in tqdm(
                        eval_dataloader,
                        desc="Evaluating",
                        dynamic_ncols=True,
                        disable=not accelerator.is_main_process,
                    ):
                        batch = batch.to(accelerator.device)
                        with torch.no_grad():
                            _, log = forward(moshi_lm=moshi_lm, batch=batch, args=args)
                            for key, value in log.items():
                                eval_logging_buffer[f"evaluation_{key}"].append(value)
                    if args.report_to:
                        gathered_metrics = accelerator.gather(
                            {
                                key: torch.tensor(values, device=accelerator.device)
                                for key, values in eval_logging_buffer.items()
                            }
                        )
                        accelerator.log(
                            {key: values.nanmean() for key, values in gathered_metrics.items()},
                            step=current_steps,
                        )

                # Save checkpoint
                if args.save_steps is not None and current_steps % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
                    accelerator.save_state(output_dir)
                    # Remove old checkpoints beyond keep_last_n_checkpoints
                    if accelerator.is_main_process and args.keep_last_n_checkpoints > 0:
                        import shutil
                        import re as _re
                        ckpt_dirs = sorted(
                            [
                                d for d in glob.glob(os.path.join(args.output_dir, "step_*"))
                                if os.path.isdir(d) and _re.match(r".*step_\d+$", d)
                            ],
                            key=lambda d: int(d.rsplit("_", 1)[-1]),
                        )
                        for old_dir in ckpt_dirs[: -args.keep_last_n_checkpoints]:
                            logger.info(f"Removing old checkpoint: {old_dir}")
                            shutil.rmtree(old_dir)

                # Stop training
                if args.max_train_steps is not None and current_steps >= args.max_train_steps:
                    break

    output_dir = os.path.join(args.output_dir, f"step_{current_steps}")
    accelerator.save_state(output_dir)


if __name__ == "__main__":
    main()
