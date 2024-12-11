import os
import random

import torch
import torch.nn as nn

from torch.utils.data import Dataset, Sampler

class BERTdataset(Dataset):
    def __init__(self, files_path, tokenizer, seq_len):
        self.files_path = files_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.pairs = []

        for i in range(len(files_path)):
            lines = []
            with open(files_path[i], "r", encoding="utf-8") as f:
                count = 0
                for line in f:
                    lines.append(line)
                    count += 1
                    if count == 10000:
                        break
            
            for j in range(len(lines)-1):
                self.pairs.append((lines[j], lines[j+1]))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        s1, s2, is_next = self.get_pair(index, balance=True)

        masked_numericalized_s1, s1_mask = self.mask_sentence(s1)
        masked_numericalized_s2, s2_mask = self.mask_sentence(s2)

        t1 = [self.tokenizer.vocab['[CLS]']] + masked_numericalized_s1 + [self.tokenizer.vocab['[SEP]']]
        t2 = masked_numericalized_s2 + [self.tokenizer.vocab['[SEP]']]
        t1_mask = [self.tokenizer.vocab['[PAD]']] + s1_mask + [self.tokenizer.vocab['[PAD]']]
        t2_mask = s2_mask + [self.tokenizer.vocab['[PAD]']]

        segment_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_mask + t2_mask)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_ids.extend(padding)

        return (torch.tensor(bert_input),
                torch.tensor(bert_label),
                torch.tensor(segment_ids),
                torch.tensor(is_next))
          
    def get_pair(self, index, balance=False):
        if balance:
            if index % 2 == 0:  # Alterner pour garantir l'équilibre
                # Paire positive
                s1, s2 = self.pairs[index]
                is_next = 1
            else:
                # Paire négative
                s1 = self.pairs[index][0]
                random_index = random.randrange(len(self.pairs))
                s2 = self.pairs[random_index][1]
                is_next = 0
        else:
            s1, s2 = self.pairs[index]
            is_next = 1
            if random.random() > 0.5:
                random_index = random.randrange(len(self.pairs))
                s2 = self.pairs[random_index][1]
                is_next = 0

        return s1, s2, is_next
    
    def mask_sentence(self, s):
        words = s.split()
        masked_numericalized_s = []
        mask = []
        for word in words:
            prob = random.random()
            token_ids = self.tokenizer(word)['input_ids'][1:-1]
            if prob < 0.15:
                prob /= 0.15
                for token_id in token_ids:
                    if prob < 0.8:
                        masked_numericalized_s.append(self.tokenizer.vocab['[MASK]'])
                    elif prob < 0.9:
                        masked_numericalized_s.append(random.randrange(len(self.tokenizer.vocab)))
                    else:
                        masked_numericalized_s.append(token_id)
                    mask.append(token_id) 
            else:
                masked_numericalized_s.extend(token_ids)
                mask.extend([0] * len(token_ids))

        assert len(masked_numericalized_s) == len(mask)
        return masked_numericalized_s, mask
    

# ======================

class BERTdataset2(Dataset):
    def __init__(self, files_path, tokenizer, seq_len):
        self.files_path = files_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.pairs_pos = []
        self.pairs_neg = []

        for i in range(len(files_path)):
            lines = []
            with open(files_path[i], "r", encoding="utf-8") as f:
                for count, line in enumerate(f):
                    # if count == 10000:
                    #     break
                    lines.append(line)

            for j in range(len(lines) - 1):
                self.pairs_pos.append((lines[j], lines[j+1], 1))
                random_index = random.randrange(len(lines))
                self.pairs_neg.append((lines[j], lines[random_index], 0))

    def __len__(self):
        return len(self.pairs_pos) + len(self.pairs_neg)

    def __getitem__(self, index):
        if index % 2 == 0:
            s1, s2, is_next = self.pairs_pos[index // 2]
        else:
            s1, s2, is_next = self.pairs_neg[index // 2]

        masked_numericalized_s1, s1_mask = self.mask_sentence(s1)
        masked_numericalized_s2, s2_mask = self.mask_sentence(s2)

        t1 = [self.tokenizer.vocab['[CLS]']] + masked_numericalized_s1 + [self.tokenizer.vocab['[SEP]']]
        t2 = masked_numericalized_s2 + [self.tokenizer.vocab['[SEP]']]
        t1_mask = [self.tokenizer.vocab['[PAD]']] + s1_mask + [self.tokenizer.vocab['[PAD]']]
        t2_mask = s2_mask + [self.tokenizer.vocab['[PAD]']]

        segment_ids = ([0 for _ in range(len(t1))] + [1 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_mask + t2_mask)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_ids.extend(padding)

        return (torch.tensor(bert_input),
                torch.tensor(bert_label),
                torch.tensor(segment_ids),
                torch.tensor(is_next))

    def mask_sentence(self, s):
        words = s.split()
        masked_numericalized_s = []
        mask = []
        for word in words:
            prob = random.random()
            token_ids = self.tokenizer(word)['input_ids'][1:-1]
            if prob < 0.15:
                prob /= 0.15
                for token_id in token_ids:
                    if prob < 0.8:
                        masked_numericalized_s.append(self.tokenizer.vocab['[MASK]'])
                    elif prob < 0.9:
                        masked_numericalized_s.append(random.randrange(len(self.tokenizer.vocab)))
                    else:
                        masked_numericalized_s.append(token_id)
                    mask.append(token_id)
            else:
                masked_numericalized_s.extend(token_ids)
                mask.extend([0] * len(token_ids))

        assert len(masked_numericalized_s) == len(mask)
        return masked_numericalized_s, mask


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.pos_indices = list(range(len(dataset.pairs_pos)))
        self.neg_indices = list(range(len(dataset.pairs_neg)))

        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)

    def __iter__(self):
        indices = []
        for pos_idx, neg_idx in zip(self.pos_indices, self.neg_indices):
            indices.append(pos_idx * 2)
            indices.append(neg_idx * 2 + 1)
        return iter(indices)

    def __len__(self):
        return min(len(self.pos_indices), len(self.neg_indices)) * 2
