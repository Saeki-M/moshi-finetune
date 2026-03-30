from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def main_speaker_streams(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
) -> list[np.ndarray]:
    """
    Pick the main speaker and create the streams.
    """
    list_of_streams = []
    for speaker in speakers:
        other = {"A": "B", "B": "A"}[speaker]
        for main_example, other_example in zip(
            batched_examples[speaker], batched_examples[other], strict=False
        ):
            streams = np.concat(
                [
                    np.array(main_example),  # 1 text + 8 audio
                    np.array(other_example)[1:],  # 8 audio
                ],
                axis=0,
            )
            list_of_streams.append(streams)
    return list_of_streams


def delay_and_pad_streams(
    list_of_streams: list[np.ndarray],
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
) -> list[np.array]:
    """
    Apply delays and padding to the streams.
    """
    max_delay = max(delays)

    list_of_delayed_streams = []

    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        delayed = np.zeros((num_streams, num_frames + max_delay), dtype=streams.dtype)
        for i, delay in enumerate(delays):
            delayed[i, :delay] = initial_token_ids[i]
            delayed[i, delay : delay + num_frames] = streams[i]
            delayed[i, delay + num_frames :] = padding_token_ids[i]
        delayed = np.concat(
            [
                np.array(initial_token_ids, dtype=streams.dtype)[
                    :, None
                ],  # add initial token at the beginning
                delayed,
            ],
            axis=1,
        )

        list_of_delayed_streams.append(delayed)

    return list_of_delayed_streams


def split_streams(
    list_of_streams: list[np.ndarray],
    max_length: int,
) -> list[np.ndarray]:
    """
    Split the streams into chunks of max_length.
    """
    list_of_chunked_streams = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        num_splits = -(-num_frames // max_length)
        list_of_chunked_streams.extend(np.array_split(streams, num_splits, axis=1))
    return list_of_chunked_streams


def filter_out_short_streams(
    list_of_streams: list[np.ndarray],
    min_length: int,
) -> list[np.ndarray]:
    """
    Filter out streams that are shorter than min_length.
    """
    return [s for s in list_of_streams if s.shape[1] >= min_length]


def make_streams_labels(
    list_of_streams: list[np.ndarray],
    initial_token_ids: list[int],
    zero_token_id: int,
    model_user_stream: bool = True,
) -> list[np.ndarray]:
    """
    Make the labels for the streams.
    When model_user_stream is False, set user audio tokens (indices 9-16) to zero_token_id.
    """
    list_of_labels = []
    for streams in list_of_streams:
        num_streams, num_frames = streams.shape
        label = streams.copy()
        for i in range(num_streams):
            # zero out initial tokens
            label[i] = np.where(
                label[i] < initial_token_ids[i],
                label[i],
                zero_token_id,
            )

        # Set user audio tokens (indices 9-16) to zero_token_id when not modeling user stream
        if not model_user_stream:
            label[9:17] = zero_token_id

        list_of_labels.append(label)
    return list_of_labels


def preprocess_function(
    batched_examples: dict[str, list[Any]],
    speakers: list[str],
    max_length: int | None,
    min_length: int | None,
    delays: list[int],
    initial_token_ids: list[int],
    padding_token_ids: list[int],
    zero_token_id: int,
    model_user_stream: bool = True,
) -> dict[str, list[Any]]:
    # 1. make main speaker streams
    list_of_streams = main_speaker_streams(
        batched_examples=batched_examples,
        speakers=speakers,
    )

    # 2. delay and pad streams
    list_of_streams = delay_and_pad_streams(
        list_of_streams=list_of_streams,
        delays=delays,
        initial_token_ids=initial_token_ids,
        padding_token_ids=padding_token_ids,
    )

    # 3. split streams by max length
    if max_length is not None:
        list_of_streams = split_streams(
            list_of_streams=list_of_streams,
            max_length=max_length,
        )

    # 4. filter out short streams
    if min_length is not None:
        list_of_streams = filter_out_short_streams(
            list_of_streams=list_of_streams,
            min_length=min_length,
        )

    # 5. make labels (with user audio masking if not modeling user stream)
    list_of_labels = make_streams_labels(
        list_of_streams=list_of_streams,
        initial_token_ids=initial_token_ids,
        zero_token_id=zero_token_id,
        model_user_stream=model_user_stream,
    )

    list_of_num_streams = [streams.shape[0] for streams in list_of_streams]
    list_of_num_frames = [streams.shape[1] for streams in list_of_streams]

    return {
        "streams": list_of_streams,
        "labels": list_of_labels,
        "num_streams": list_of_num_streams,
        "num_frames": list_of_num_frames,
    }


def undelay_tokens(
    tokens: np.ndarray | torch.LongTensor, delays: list[int]
) -> np.ndarray | torch.LongTensor | None:
    """
    Restore the undelayed tokens from the delayed tokens

    Args:
        tokens (np.ndarray | torch.LongTensor): shape is (B, K, Td).
        delays (list[int]): delays for each of K codebooks

    Returns:
        undelayed_tokens (np.ndarray | torch.LongTensor): shape is (B, K, T).
            T is not necessarily equal to Td because of the delays.
    """
    max_delay = max(delays)
    B, K, Td = tokens.shape

    if Td < max_delay + 1:  # too short
        return None

    assert K == len(delays), f"Expected K == {len(delays)}, but got {K}"

    T = Td - max_delay
    if isinstance(tokens, np.ndarray):
        undelayed_tokens = np.zeros((B, K, T), dtype=tokens.dtype)
    else:
        undelayed_tokens = torch.zeros((B, K, T), dtype=tokens.dtype, device=tokens.device)
    for cb_index, delay in enumerate(delays):
        undelayed_tokens[:, cb_index] = tokens[:, cb_index, delay : delay + T]

    return undelayed_tokens


@dataclass
class Batch:
    example_ids: list[int] | None
    input_ids: torch.LongTensor
    text_attention_mask: torch.LongTensor
    labels: torch.LongTensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            example_ids=self.example_ids,
            input_ids=self.input_ids.to(device),
            text_attention_mask=self.text_attention_mask.to(device),
            labels=self.labels.to(device),
        )


class DataCollator:
    def __init__(self, zero_token_id: int):
        self.zero_token_id = zero_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> Batch:
        """
        Collate the examples into a batch.
        Args:
            examples (list[dict[str, Any]]): list of examples
                ```
                [
                    {
                        "streams": list[list[int]],
                        "labels": list[list[int]],
                        "num_streams": int,
                        "num_frames": int,
                    },
                    ...
                ]
                ```
        Returns:
            batch (Batch): batch of examples
        """
        # pad with zero tokens
        batch_size = len(examples)
        num_streams = examples[0]["num_streams"]
        max_frames = max([e["num_frames"] for e in examples])
        zero_ids = torch.full(
            (batch_size, num_streams, max_frames), fill_value=self.zero_token_id, dtype=torch.long
        )

        input_ids = zero_ids.clone()
        for i, e in enumerate(examples):
            input_ids[i, :, : e["num_frames"]] = torch.tensor(e["streams"], dtype=torch.long)

        text_attention_mask = zero_ids[:, 0, :].clone()  # (batch_size, max_frames)
        for i, e in enumerate(examples):
            text_attention_mask[i, : e["num_frames"]] = 1

        labels = zero_ids.clone()
        for i, e in enumerate(examples):
            labels[i, :, : e["num_frames"]] = torch.tensor(e["labels"], dtype=torch.long)

        example_ids = None
        if "example_id" in examples[0]:
            example_ids = [e["example_id"] for e in examples]

        batch = Batch(
            example_ids=example_ids,
            input_ids=input_ids,
            text_attention_mask=text_attention_mask,
            labels=labels,
        )
        return batch
