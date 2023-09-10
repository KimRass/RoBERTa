import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import gc
from pathlib import Path
from time import time
from tqdm.auto import tqdm
import argparse

import pretrain.config as config
from utils import get_elapsed_time
from model import BERTForPretraining
from pretrain.wordpiece import load_fast_bert_tokenizer
from pretrain.bookcorpus import BookCorpusForBERT, BookCorpusForRoBERTa
from pretrain.masked_language_model import MaskedLanguageModel
from pretrain.loss import BERTPretrainingLoss, RoBERTaPretrainingLoss
from pretrain.evalute import get_nsp_acc, get_mlm_acc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epubtxt_dir", type=str, required=False, default="../bookcurpus/epubtxt",
    )
    parser.add_argument("--batch_size", type=int, required=False, default=256)
    parser.add_argument("--tokenize_in_advance", action="store_true")
    parser.add_argument("--ckpt_path", type=str, required=False)

    args = parser.parse_args()
    return args


def save_checkpoint(step, model, optim, ckpt_path):
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "optimizer": optim.state_dict(),
    }
    if config.N_GPUS > 1:
        ckpt["model"] = model.module.state_dict()
    else:
        ckpt["model"] = model.state_dict()
    torch.save(ckpt, str(ckpt_path))


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)

    gc.collect()
    torch.cuda.empty_cache()

    args = get_args()

    print(f"BATCH_SIZE = {args.batch_size}")
    print(f"N_WORKERS = {config.N_WORKERS}")
    print(f"MAX_LEN = {config.MAX_LEN}")
    print(f"SEQ_LEN = {config.SEQ_LEN}")

    # "We train with batch size of 256 sequences (256 sequences * 512 tokens
    # = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs
    # over the 3.3 billion word corpus." (Comment: 256 * 512 * 1,000,000 / 3,300,000,000
    # = 39.7)
    N_STEPS = (256 * 512 * 1_000_000) // (args.batch_size * config.SEQ_LEN)
    print(f"N_STEPS = {N_STEPS:,}", end="\n\n")

    # tokenizer = load_bert_tokenizer(config.VOCAB_PATH)
    tokenizer = load_fast_bert_tokenizer(vocab_dir=config.VOCAB_DIR)
    # train_ds = BookCorpusForBERT(
    train_ds = BookCorpusForRoBERTa(
        epubtxt_dir=args.epubtxt_dir,
        tokenizer=tokenizer,
        seq_len=config.SEQ_LEN,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.N_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    model = BERTForPretraining( # Smaller than BERT-Base
        vocab_size=config.VOCAB_SIZE,
        max_len=config.MAX_LEN,
        pad_id=train_ds.pad_id,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        mlp_size=config.MLP_SIZE,
    ).to(config.DEVICE)
    if config.N_GPUS > 1:
        model = nn.DataParallel(model)

    mlm = MaskedLanguageModel(
        vocab_size=config.VOCAB_SIZE,
        mask_id=tokenizer.mask_token_id,
        no_mask_token_ids=[
            train_ds.unk_id, train_ds.cls_id, train_ds.sep_id, train_ds.pad_id, train_ds.unk_id,
        ],
        select_prob=config.SELECT_PROB,
        mask_prob=config.MASK_PROB,
        randomize_prob=config.RANDOMIZE_PROB,
    )

    optim = Adam(
        model.parameters(),
        lr=config.MAX_LR,
        betas=(config.BETA1, config.BETA2),
        weight_decay=config.WEIGHT_DECAY,
    )

    # crit = BERTPretrainingLoss(vocab_size=config.VOCAB_SIZE)
    crit = RoBERTaPretrainingLoss(vocab_size=config.VOCAB_SIZE)

    ### Resume
    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=config.DEVICE)
        if config.N_GPUS > 1:
            model.module.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        prev_ckpt_path = Path(args.ckpt_path)
        print(f"Resuming from checkpoint\n    '{str(Path(args.ckpt_path).name)}'...")
    else:
        step = 0
        prev_ckpt_path = Path(".pth")

    print("Training...")
    start_time = time()
    # accum_nsp_loss = 0
    # accum_nsp_acc = 0
    accum_mlm_loss = 0
    accum_mlm_acc = 0
    step_cnt = 0
    while True:
        # for gt_token_ids, seg_ids, gt_is_next in train_dl:
        for gt_token_ids, seg_ids in train_dl:
            if step < N_STEPS:
                step +=1

                gt_token_ids = gt_token_ids.to(config.DEVICE)
                seg_ids = seg_ids.to(config.DEVICE)
                # gt_is_next = gt_is_next.to(config.DEVICE)

                masked_token_ids, select_mask = mlm(gt_token_ids)

                pred_is_next, pred_token_ids = model(token_ids=masked_token_ids, seg_ids=seg_ids)
                # nsp_loss, mlm_loss = crit(
                mlm_loss = crit(
                    # pred_is_next=pred_is_next,
                    # gt_is_next=gt_is_next,
                    pred_token_ids=pred_token_ids,
                    gt_token_ids=gt_token_ids,
                    select_mask=select_mask,
                )
                # loss = nsp_loss + mlm_loss
                loss = mlm_loss

                optim.zero_grad()
                loss.backward()
                optim.step()

                # accum_nsp_loss += nsp_loss.item()
                accum_mlm_loss += mlm_loss.item()

                # nsp_acc = get_nsp_acc(pred_is_next=pred_is_next, gt_is_next=gt_is_next)
                # accum_nsp_acc += nsp_acc
                mlm_acc = get_mlm_acc(pred_token_ids=pred_token_ids, gt_token_ids=gt_token_ids)
                accum_mlm_acc += mlm_acc
                step_cnt += 1

                if (step % (config.N_CKPT_SAMPLES // args.batch_size) == 0) or (step == N_STEPS):
                    print(f"[ {step:,}/{N_STEPS:,} ][ {get_elapsed_time(start_time)} ]", end="")
                    # print(f"[ NSP loss: {accum_nsp_loss / step_cnt:.4f} ]", end="")
                    # print(f"[ NSP acc: {accum_nsp_acc / step_cnt:.3f} ]", end="")
                    print(f"[ MLM loss: {accum_mlm_loss / step_cnt:.4f} ]", end="")
                    print(f"[ MLM acc: {accum_mlm_acc / step_cnt:.3f} ]")

                    start_time = time()
                    # accum_nsp_loss = 0
                    # accum_nsp_acc = 0
                    accum_mlm_loss = 0
                    accum_mlm_acc = 0
                    step_cnt = 0

                    cur_ckpt_path = config.CKPT_DIR/f"bookcorpus_step_{step}.pth"
                    save_checkpoint(step=step, model=model, optim=optim, ckpt_path=cur_ckpt_path)
                    if prev_ckpt_path.exists():
                        prev_ckpt_path.unlink()
                    prev_ckpt_path = cur_ckpt_path
