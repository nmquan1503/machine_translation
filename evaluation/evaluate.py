import torch
from tqdm import tqdm
import sacrebleu

import config
from data.tokenizer import Tokenizer
from data.dataloader import build_dataloader
from ssm_mamba import Seq2SeqModel, Seq2SeqModelConfig

def evaluate():
    tokenizer = Tokenizer()
    test_loader = build_dataloader(config.TEST_SRC_PATH, config.TEST_TGT_PATH, tokenizer, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Seq2SeqModel(Seq2SeqModelConfig(
        vocab_size=config.VOCAB_SIZE,
        pad_token_id=tokenizer.pad_id,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        model_dim=config.MODEL_DIM,
        state_dim=config.STATE_DIM,
        conv_kernel=config.CONV_KERNEL,
        num_layers=config.NUM_LAYERS
    )).to(device)

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))

    model.eval()

    all_preds = []
    all_refs = []

    for batch in tqdm(test_loader, desc="Test"):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        seq_ids = model.generate(input_ids, config.MAX_NEW_TOKENS, config.TEMPERATURE).cpu()
        target_ids = target_ids.cpu()

        for pred, tgt in zip(seq_ids, target_ids):
            pred = pred.tolist()
            tgt = tgt.tolist()

            pred_text = tokenizer.decode(pred)
            tgt_text = tokenizer.decode(tgt)

            all_preds.append(pred_text)
            all_refs.append(tgt_text)

    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    print(f"\nBLEU: {bleu.score:.4f}")

if __name__ == "__main__":
    evaluate()