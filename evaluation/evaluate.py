import torch
from tqdm import tqdm
import sacrebleu

import config
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from mamba.mamba_ssm.models.config_mamba import MambaConfig
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def evaluate():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(config.TEST_SRC_PATH, config.TEST_TGT_PATH, tokenizer, False, mode="test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MambaLMHeadModel(MambaConfig(
        d_model=config.MODEL_DIM,
        n_layer=config.NUM_LAYERS,
        d_intermediate=config.INNER_DIM,
        vocab_size=config.VOCAB_SIZE,
        ssm_cfg={
            "layer": config.TYPE,
            "d_state": config.STATE_DIM,
            "d_conv": config.CONV_KERNEL,
            "use_mem_eff_path": False
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
    ), device=device)

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_refs = []

    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch["input_ids"].to(device)
        output_ids = batch["output_ids"]

        for i in range(input_ids.size(0)):
            single_input = input_ids[i].unsqueeze(0)
            single_output = output_ids[i]

            with torch.no_grad():
                seq_ids = model.generate(single_input, max_length=config.MAX_NEW_TOKENS, eos_token_id=tokenizer.eos_id, cg=True).cpu()

            pred = seq_ids[0].tolist()
            pred = pred[pred.index(tokenizer.bos_id)]

            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(single_output)

            all_preds.append(pred_text)
            all_refs.append(tgt_text)

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    print(f"\nBLEU: {bleu.score:.4f}")

if __name__ == "__main__":
    evaluate()