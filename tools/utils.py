import logging
import os
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)


def execute_data_processing(
    dataset: Iterator[Any],
    process_func: Callable[[int, Any], Any],
    num_workers: int,
    data_count: int | None = None,
):
    """Execute data processing in parallel using multiple workers.

    Args:
        dataset (Iterator[Any]): An iterator that yields data items to be processed.
        process_func (Callable[[int, Any], Any]): A function that processes a single data
            item. It takes a worker ID and a data item as arguments.
        num_workers (int): The number of worker processes to use.
        data_count (int | None): The total number of data items to process.
    """
    if os.getenv("DEBUG") in ("1", "true", "True"):
        for data in tqdm(dataset, total=data_count):
            process_func(0, data)
        return

    with (
        ProcessPoolExecutor(max_workers=num_workers) as executor,
        tqdm(total=data_count, desc="Processing data") as pbar,
    ):
        # Track which worker is handling which future
        futures: dict[Future[Any], int] = {}
        # Track next worker_id to assign (round-robin)
        next_worker_id = 0

        # Submit initial jobs for each worker
        for _ in range(num_workers):
            try:
                data = next(dataset)
                future = executor.submit(process_func, next_worker_id, data)
                futures[future] = next_worker_id
                next_worker_id = (next_worker_id + 1) % num_workers
            except StopIteration:
                break

        # Process results as they complete and submit new jobs
        while futures:
            # Wait for at least one future to complete
            complete_job = as_completed(futures).__next__()
            worker_id = futures[complete_job]

            # Update progress
            pbar.update()

            try:
                complete_job.result()  # Raise exception if any
            except Exception:
                logger.exception("Error in worker")

            del futures[complete_job]

            try:
                data = next(dataset)
                new_future = executor.submit(process_func, worker_id, data)
                futures[new_future] = worker_id
            except StopIteration:
                # No more data to process
                pass
