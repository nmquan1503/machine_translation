from torch.utils.data import Dataset
import torch

from .tokenizer import Tokenizer
from .preprocess import preprocess_texts

class MTDataset(Dataset):
    def __init__(
        self, 
        src_text_path: str, 
        tgt_text_path: str, 
        tokenizer: Tokenizer
    ):
        with open(src_text_path, "r", encoding="utf-8") as f:
            src_texts = f.readlines()
            src_texts = preprocess_texts(src_texts)
        with open(tgt_text_path, "r", encoding="utf-8") as f:
            tgt_texts = f.readlines()
            tgt_texts = preprocess_texts(tgt_texts)
        
        if len(src_texts) != len(tgt_texts):
            raise ValueError("source and target must have same length")

        self.src_ids = tokenizer.encode(src_texts, add_bos=False, add_eos=False)
        self.tgt_ids = tokenizer.encode(tgt_texts, add_bos=True, add_eos=True)

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, index):
        src = self.src_ids[index]
        tgt = self.tgt_ids[index]

        return {
            "input_ids": torch.tensor(src, dtype=torch.long),
            "target_ids": torch.tensor(tgt, dtype=torch.long) 
        }