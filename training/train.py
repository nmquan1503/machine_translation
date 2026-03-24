import torch

from ssm_mamba import Seq2SeqModel, Seq2SeqModelConfig
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from .trainer import Trainer
import config

def train():
    tokenizer = Tokenizer()
    train_loader = build_dataloader(config.TRAIN_SRC_PATH, config.TRAIN_TGT_PATH, tokenizer)
    dev_loader = build_dataloader(config.DEV_SRC_PATH, config.DEV_TGT_PATH, tokenizer, False)
    model = Seq2SeqModel(Seq2SeqModelConfig(
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        model_dim=config.MODEL_DIM,
        state_dim=config.STATE_DIM,
        conv_kernel=config.CONV_KERNEL,
        num_layers=config.NUM_LAYERS
    ))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    trainer.train()

if __name__ == "__main__":
    train()