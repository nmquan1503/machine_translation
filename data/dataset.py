from torch.utils.data import Dataset
import torch

from .tokenizer import Tokenizer
from .preprocess import preprocess_texts

class MTDataset(Dataset):
    def __init__(
        self, 
        src_text_path: str, 
        tgt_text_path: str, 
        tokenizer: Tokenizer,
        mode="train"
    ):
        with open(src_text_path, "r", encoding="utf-8") as f:
            src_texts = f.readlines()
            src_texts = preprocess_texts(src_texts)
        with open(tgt_text_path, "r", encoding="utf-8") as f:
            tgt_texts = f.readlines()
            tgt_texts = preprocess_texts(tgt_texts)
        
        if len(src_texts) != len(tgt_texts):
            raise ValueError("source and target must have same length")

        self.bos_id = tokenizer.bos_id
        self.pad_id = tokenizer.pad_id
        self.mode = mode
        self.sequences = [
            tokenizer.encode(src, add_bos=False, add_eos=False) +
            tokenizer.encode(tgt, add_bos=True, add_eos=True)
            for src, tgt in zip(src_texts, tgt_texts)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        bos_pos = seq.index(self.bos_id)
        if self.mode == "train":
            input_ids = torch.tensor(seq[:-1], dtype=torch.long)
            labels = torch.tensor(seq[1:], dtype=torch.long)
            labels[:bos_pos] = self.pad_id

            return {
                "input_ids": input_ids,
                "labels": labels
            }
        
        else:
            input_ids = torch.tensor(seq[:bos_pos + 1], dtype=torch.long)
            
            return {
                "input_ids": input_ids,
                "output_ids": seq[bos_pos:]
            }