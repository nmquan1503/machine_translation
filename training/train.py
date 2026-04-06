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
    ssm_cfg={
        "layer": config.TYPE,
        "d_state": config.STATE_DIM,
        "d_conv": config.CONV_KERNEL,
    }
    if config.TYPE == "Mamba1":
        ssm_cfg["use_fast_path"] = False
        ssm_cfg["conv_bias"] = True
        ssm_cfg["bias"] = True
    elif config.TYPE == "Mamba2":
        ssm_cfg["use_mem_eff_path"] = False
        ssm_cfg["sequence_parallel"] = False
        ssm_cfg["chunk_size"] = config.CHUNK_SIZE
        ssm_cfg["headdim"] = config.MODEL_DIM * 2 / config.NUM_HEADS
        ssm_cfg["bias"] = True
        ssm_cfg["conv_bias"] = True
        ssm_cfg["ngroups"] = config.NUM_GROUPS
    elif config.TYPE == "Mamba3":
        ssm_cfg["mimo"] = True
        ssm_cfg["mimo_rank"] = config.MIMO_RANK
        ssm_cfg["chunk_size"] = 64 // config.MIMO_RANK
        ssm_cfg["headdim"] = config.MODEL_DIM * 2 / config.NUM_HEADS
        ssm_cfg["ngroups"] = config.NUM_GROUPS
    model = MambaLMHeadModel(MambaConfig(
        d_model=config.MODEL_DIM,
        n_layer=config.NUM_LAYERS,
        d_intermediate=config.INNER_DIM,
        vocab_size=config.VOCAB_SIZE,
        ssm_cfg=ssm_cfg,
        attn_layer_idx=config.ATTENTION_LAYERS,
        attn_cfg={
            "num_heads": config.NUM_HEADS,
            "causal": True,
            "embed_dim": config.MODEL_DIM
        },
        rms_norm=True, 
        fused_add_norm=True,
        residual_in_fp32=True,
        tie_embeddings=True,
        pad_vocab_size_multiple=1
    ), device="cuda", dtype=torch.float32)

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