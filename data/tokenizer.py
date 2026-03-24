import sentencepiece as spm
from pathlib import Path
from typing import List

import config

def train_tokenizer():
    spm.SentencePieceTrainer.Train(
        input=f"{config.TRAIN_SRC_PATH},{config.TRAIN_TGT_PATH}",
        model_prefix=config.SPM_MODEL_PATH.split(".model")[0],
        vocab_size=config.VOCAB_SIZE,
        model_type="unigram",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        minloglevel=2
    )

class Tokenizer:
    def __init__(self):
        if not Path(config.SPM_MODEL_PATH).exists():
            train_tokenizer()
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(config.SPM_MODEL_PATH)

        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
    
    def encode(self, texts: str | List[str], add_bos=True, add_eos=True):
        ids = self.sp.Encode(texts, out_type=int)

        if add_bos:
            if isinstance(texts, str):
                ids = [self.bos_id] + ids
            else:
                ids = [[self.bos_id] + i for i in ids]
        
        if add_eos:
            if isinstance(texts, str):
                ids = ids + [self.eos_id]
            else:
                ids = [[self.eos_id] + i for i in ids]
        
        return ids

    def decode(self, ids: List[int] | List[List[int]]):
        return self.sp.Decode(ids)