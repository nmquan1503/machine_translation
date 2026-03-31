import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from functools import partial

from .dataset import MTDataset
from .tokenizer import Tokenizer
import config

def collate_fn(batch, pad_id, mode):
    input_ids = [item["input_ids"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)

    if mode == "train":
        labels = [item["labels"] for item in batch]
        labels = pad_sequence(labels, batch_first=True, padding_value=pad_id)
        return {
            "input_ids": input_ids,
            "labels": labels
        }

    else:
        output_ids = [item["output_ids"] for item in batch]
        return {
            "input_ids": input_ids,
            "output_ids": output_ids
        }

def build_dataloader(
    src_text_path: str, 
    tgt_text_path: str, 
    tokenizer: Tokenizer,
    shuffle=True,
    mode="train"
):
    dataset = MTDataset(src_text_path, tgt_text_path, tokenizer, mode)
    collate = partial(collate_fn, pad_id=tokenizer.pad_id, mode=mode)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=collate
    )

