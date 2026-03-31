import torch

from mamba.mamba_ssm.models.config_mamba import MambaConfig
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from .trainer import Trainer
import config

def train():
    tokenizer = Tokenizer()
    train_loader = build_dataloader(config.TRAIN_SRC_PATH, config.TRAIN_TGT_PATH, tokenizer)
    dev_loader = build_dataloader(config.DEV_SRC_PATH, config.DEV_TGT_PATH, tokenizer, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaLMHeadModel(MambaConfig(
        d_model=config.MODEL_DIM,
        n_layer=config.NUM_LAYERS,
        d_intermediate=config.INNER_DIM,
        vocab_size=config.VOCAB_SIZE,
        ssm_cfg={
            "layer": config.TYPE,
            "d_state": config.STATE_DIM,
            "d_conv": config.CONV_KERNEL
        },
        attn_layer_idx=config.ATTENTION_LAYERS,
        attn_cfg={
            "num_heads": config.NUM_HEADS
        },
        rms_norm=config.USE_RMS_NORM, 
        fused_add_norm=config.USE_FUSE_ADD_NORM,
        residual_in_fp32=True,
        tie_embeddings=True,
        pad_vocab_size_multiple=1
    ), device)

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