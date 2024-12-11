import os
import time
import math

from contextlib import nullcontext
from tqdm.auto import tqdm

import numpy as np
import random
import torch

import tokenizer
from dataset import BERTdataset2, BalancedBatchSampler
from model import BERTconfig, BERT

# =========================================================
DISTRIBUTED = True
# Default config values
# I/O
init_from = "scratch"
eval_interval = 2000
eval_only = False
eval_iters = 200
log_interval = 1
always_save_checkpoint = True

# wandb logging
wandb_log = False

# Data
seq_len = 100
#vocab_size = 30000 # définie par tokenizer.vocab_size
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 32

# Model
hid_dim = 256 # Taille des embeddings
n_layer = 3#6
n_head = 4#8
dropout = 0.3

# Optimizer
learning_rate = 5e-5 
max_iters = 60000 # nombre de batch calculé

alpha = 1
beta = 1

# System
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# ==========================================================
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# ==========================================================
#torch.manual_seed(1337)
device_type = 'cuda' if 'cuda' in device else 'cpu'
# ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Import tokenizer
tokenizer_dir = '../models/tokenizer'
tokenizer = tokenizer.load_tokenizer(tokenizer_dir)

# Import data
data_dir = "../data"

def collate_fn(batch):
    bert_inputs, bert_labels, segment_ids, is_next = zip(*batch)

    bert_inputs = torch.stack(bert_inputs)
    bert_labels = torch.stack(bert_labels)
    segment_ids = torch.stack(segment_ids)
    is_next = torch.stack(is_next)

    if device_type == 'cuda':
        bert_inputs = bert_inputs.pin_memory().to(device, non_blocking=True)
        bert_labels = bert_labels.pin_memory().to(device, non_blocking=True)
        segment_ids = segment_ids.pin_memory().to(device, non_blocking=True)
        is_next = is_next.pin_memory().to(device, non_blocking=True)
    else:
        bert_inputs = bert_inputs.to(device)
        bert_labels = bert_labels.to(device)
        segment_ids = segment_ids.to(device)
        is_next = is_next.to(device)
    
    return [bert_inputs, segment_ids, bert_labels, is_next]

all_files_path = [os.path.join(data_dir, file_name) for file_name in os.listdir(data_dir)]

train_dataset = BERTdataset2(all_files_path[:1], tokenizer, seq_len)
val_dataset = BERTdataset2(all_files_path[1:2], tokenizer, seq_len)

train_sampler = BalancedBatchSampler(train_dataset)
val_sampler = BalancedBatchSampler(val_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=train_sampler, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, sampler=val_sampler, collate_fn=collate_fn)

iter_train_loader = iter(train_loader)
iter_val_loader = iter(val_loader)

def get_batch(split, iter_train_loader=iter_train_loader, iter_val_loader=iter_val_loader):
    try:
        if split == "train":
            return next(iter_train_loader), iter_train_loader, iter_val_loader
        else:
            return next(iter_val_loader), iter_train_loader, iter_val_loader

    except StopIteration:
        # Si l'un des deux iterateurs est épuisé, on redéfinit les iterateurs
        #print("Réinitialisation des itérateurs")
        iter_train_loader = iter(train_loader)
        iter_val_loader = iter(val_loader)

        if split == "train":
            return next(iter_train_loader), iter_train_loader, iter_val_loader
        else:
            return next(iter_val_loader), iter_train_loader, iter_val_loader

iter_num = 1
best_val_loss = 1e9


model_args = dict(n_layer=n_layer, n_head=n_head, hid_dim=hid_dim, seq_len=seq_len,
                  vocab_size=tokenizer.vocab_size, dropout=dropout)


if init_from == "scratch":
    print("Initializing a new model from scratch")

    bertconf = BERTconfig(**model_args)
    model = BERT(bertconf)

optimizer = model.configure_optimizers(learning_rate)

model.to(device)


checkpoint = None

@torch.no_grad()
def estimate_loss_acc(iter_train_loader, iter_val_loader):
    out = {}
    model.eval()

    ignored_tokens = [0, 1, 2, 3, 4]

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        losses_lm = torch.zeros(eval_iters)
        losses_cls = torch.zeros(eval_iters)
        acc_lm = torch.zeros(eval_iters)
        acc_cls = torch.zeros(eval_iters)

        for k in range(eval_iters):
            inputs, iter_train_loader, iter_val_loader = get_batch(split, iter_train_loader, iter_val_loader)
            logits_lm, logits_cls, loss_lm, loss_cls = model(*inputs)

            bert_labels = inputs[2]
            is_next = inputs[-1]

            # Calcul des loss
            losses[k] = (loss_lm + loss_cls).item()
            losses_lm[k] = loss_lm.item()
            losses_cls[k] = loss_cls.item()

            # Calcul de l'accuracy pour lm
            preds_lm = torch.argmax(logits_lm, dim=-1)
            valid_mask = ~torch.isin(bert_labels, torch.tensor(ignored_tokens, device=bert_labels.device))

            correct_lm = (preds_lm[valid_mask] == bert_labels[valid_mask]).sum().item()
            total_lm = valid_mask.sum().item()
            acc_lm[k] = correct_lm / total_lm if total_lm > 0 else 0

            # Calcul de l'accuracy pour cls
            preds_cls = torch.argmax(logits_cls, dim=-1)
            correct_cls = (preds_cls == is_next).sum().item()
            total_cls = is_next.size(0)
            acc_cls[k] = correct_cls / total_cls


        out[split] = {
            "loss": losses.mean().item(), 
            "loss_lm": losses_lm.mean().item(), 
            "loss_cls": losses_cls.mean().item(),
            "acc_lm": acc_lm.mean().item(),
            "acc_cls": acc_cls.mean().item()
            }
        
    model.train()
    return out, iter_train_loader, iter_val_loader

# Wandb logging
if wandb_log:
    import wandb
    wandb.init(
        project="bert_from_scratch", 
        config=model_args
    )

# Training loop
out_dir = "../checkpoints"
inputs, iter_train_loader, iter_val_loader = get_batch("train", iter_train_loader, iter_val_loader) # Fetch du premier batch

with tqdm(total=max_iters, desc=f"Training ", unit="batch") as main_bar:
    while True:
        main_bar.update(1)
    # evaluate the loss on train/val sets and write checkpoints
    
        if iter_num % eval_interval == 0:
            losses_accuracies, iter_train_loader, iter_val_loader = estimate_loss_acc(iter_train_loader, iter_val_loader)
            #print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if wandb_log:
                wandb.log({
                    "iter": iter_num,

                    "train_loss": losses_accuracies["train"]["loss"],
                    "val_loss": losses_accuracies["val"]["loss"],

                    "train_loss_lm": losses_accuracies["train"]["loss_lm"],
                    "val_loss_lm": losses_accuracies["val"]["loss_lm"],

                    "train_loss_cls": losses_accuracies["train"]["loss_cls"],
                    "val_loss_cls": losses_accuracies["val"]["loss_cls"],

                    "train_acc_lm": losses_accuracies["train"]["acc_lm"],
                    "val_acc_lm": losses_accuracies["val"]["acc_lm"],

                    "train_acc_cls": losses_accuracies["train"]["acc_cls"],
                    "val_acc_cls": losses_accuracies["val"]["acc_cls"]
                })

            if losses_accuracies['val']['loss'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses_accuracies['val']['loss']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    main_bar.set_postfix_str(f"train_loss {losses_accuracies['train']['loss']:.4f}, val_loss {losses_accuracies['val']['loss']:.4f}")
                    print(f"Saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num == 0 and eval_only:
            break

        # for micro_step in range(gradient_accumulation_steps):
        #     #with ctx:
        #     #print("avant")
        #     logits_lm, logits_cls, loss = model(*inputs)
        #     #print("après")
        #     loss = loss / gradient_accumulation_steps

        logits_lm, logits_cls, loss_lm, loss_cls = model(*inputs)

        loss = alpha*loss_lm + beta*loss_cls
        
        main_bar.set_postfix_str(f"train_loss {loss:.4f} train_loss_lm {loss_lm:.4f} train_loss_cls {loss_cls:.4f}")
        
        inputs, iter_train_loader, iter_val_loader = get_batch("train", iter_train_loader, iter_val_loader)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

        iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break