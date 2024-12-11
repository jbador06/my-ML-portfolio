import math
from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn
from torch.nn import functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, query, key, value, mask):
        assert query.size(-1) == key.size(-1) == value.size(-1), "Mismatch in input dimensions"
        matmul_qk = torch.matmul(query, torch.transpose(key,2,3))

        dk = key.shape[-1]
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
    
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        # Define dense layers corresponding to WQ, WK, and WV
        self.query = nn.Linear(config.hid_dim, config.n_head * config.d_head)
        self.key = nn.Linear(config.hid_dim, config.n_head * config.d_head)
        self.value = nn.Linear(config.hid_dim, config.n_head * config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(config)
        self.dense = nn.Linear(config.n_head * config.d_head, config.hid_dim)
        self.dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.d_head = config.d_head
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # 1. Pass through the dense layer corresponding to WQ
        # q : (bs, n_head, n_q_seq, d_head)
        query = self.query(Q).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        # 2. Pass through the dense layer corresponding to WK
        # k : (bs, n_head, n_k_seq, d_head)
        key   = self.key(K).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)
        
        # 3. Pass through the dense layer corresponding to WV
        # v : (bs, n_head, n_v_seq, d_head)
        value = self.value(V).view(batch_size, -1, self.n_head, self.d_head).transpose(1,2)

        # 4. Scaled Dot Product Attention. Using the previously implemented function
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        scaled_attention, attn_prob = self.scaled_dot_attn(query, key, value, attn_mask)
        
        # 5. Concatenate the heads
        # (bs, n_head, n_q_seq, h_head * d_head)
        concat_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head * self.d_head)
        
        # 6. Pass through the dense layer corresponding to WO
        # (bs, n_head, n_q_seq, e_embd)
        outputs = self.dense(concat_attention)
        outputs = self.dropout(outputs)
        # (bs, n_q_seq, hid_dim), (bs, n_head, n_q_seq, n_k_seq)
        return outputs, attn_prob

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hid_dim, config.pf_dim)
        self.linear_2 = nn.Linear(config.pf_dim, config.hid_dim)

    def forward(self, attention):
        output = self.linear_1(attention)
        output = F.relu(output)
        output = self.linear_2(output)
        return output

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = MultiHeadAttentionLayer(config)
        self.layernorm1 = nn.LayerNorm(config.hid_dim)
        self.dropout1 = nn.Dropout(config.dropout)

        self.ffn = PositionwiseFeedforwardLayer(config)
        self.layernorm2 = nn.LayerNorm(config.hid_dim)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, inputs, padding_mask):
        
        # 1. Encoder mutihead attention is defined
        attention, attn_prob = self.attention(inputs, inputs, inputs, padding_mask)
        attention   = self.dropout1(attention)
        
        # 2. 1 st residual layer
        attention   = self.layernorm1(inputs + attention)  # (batch_size, input_seq_len, hid_dim)
        
        # 3. Feed Forward Network
        ffn_outputs = self.ffn(attention)  # (batch_size, input_seq_len, hid_dim)
        
        ffn_outputs = self.dropout2(ffn_outputs)
        
        # 4. 2 nd residual layer
        ffn_outputs = self.layernorm2(attention + ffn_outputs)  # (batch_size, input_seq_len, hid_dim)

        # 5. Encoder output of each encoder layer
        return ffn_outputs, attn_prob

@dataclass
class BERTconfig:
    seq_len: int = 175#100
    vocab_size: int = 30000

    hid_dim: int = 32#256 # Taille des embeddings
    n_layer: int = 1#6
    n_head: int = 1#8
    dropout: float = 0.3

    n_seg_type: int = 2
    pf_dim: int = 1024
    i_pad: int = 0
    d_head: int = 64

class BERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hid_dim)
        self.position_embeddings = nn.Embedding(config.seq_len + 1, config.hid_dim)
        self.token_type_embeddings = nn.Embedding(config.n_seg_type, config.hid_dim)

        self.layer = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.linear = nn.Linear(config.hid_dim, config.hid_dim)
        self.activation = torch.tanh

        # Classifier
        self.projection_cls = nn.Linear(config.hid_dim, 2, bias=False)
        # LM
        self.projection_lm = nn.Linear(config.hid_dim, config.vocab_size, bias=False)
        self.projection_lm.weight = self.word_embeddings.weight

        def create_padding_mask(seq_q, seq_k, i_pad):
            batch_size, len_q = seq_q.size()
            batch_size, len_k = seq_k.size()
            mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
            return mask

        self.create_padding_mask = create_padding_mask
        self.i_pad = config.i_pad
        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len
    
    def forward(self, inputs, segments, labels=None, is_next=None):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1
        pos_mask = inputs.eq(self.i_pad)
        positions.masked_fill_(pos_mask, 0)

        assert torch.all(inputs < self.word_embeddings.num_embeddings), \
            f"Indices in inputs exceed embedding size: max={inputs.max()}, num_embeddings={self.word_embeddings.num_embeddings}"
        assert torch.all(positions < self.position_embeddings.num_embeddings), \
            f"Indices in positions exceed embedding size: max={positions.max()}, num_embeddings={self.position_embeddings.num_embeddings}"
        assert torch.all(segments < self.token_type_embeddings.num_embeddings), \
            f"Indices in segments exceed embedding size: max={segments.max()}, num_embeddings={self.token_type_embeddings.num_embeddings}"

        # (bs, ENCODER_LEN, hid_dim)
        outputs = self.word_embeddings(inputs) + self.position_embeddings(positions)  + self.token_type_embeddings(segments)

        # (bs, ENCODER_LEN, ENCODER_LEN)
        attn_mask = self.create_padding_mask(inputs, inputs, self.i_pad)

        attn_probs = []
        for l in self.layer:
            # (bs, ENCODER_LEN, hid_dim), (bs, n_head, ENCODER_LEN, ENCODER_LEN)
            outputs, attn_prob = l(outputs, attn_mask)
            #attn_probs.append(attn_prob)

        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)

        assert inputs.max() < self.vocab_size
        
        logits_cls = self.projection_cls(outputs_cls) # (bs, 2)
        logits_lm = self.projection_lm(outputs) # (bs, seq_len, vocab_size)
        
        if labels is not None and is_next is not None:
            loss_lm = F.cross_entropy(logits_lm.view(-1, logits_lm.size(-1)), labels.view(-1), ignore_index=self.i_pad)
            loss_cls = F.cross_entropy(logits_cls, is_next)

        else:
            loss_lm = None
            loss_cls = None

        return logits_lm, logits_cls, loss_lm, loss_cls
    
    # def configure_optimizers(self, learning_rate):
    #     param_dict = {pn: p for pn, p in self.named_parameters()}
    #     param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    #     params = list(param_dict.values())
    #     return torch.optim.Adam(self.parameters(), lr=learning_rate)


    def configure_optimizers(self, base_lr, cls_lr_factor=1, lm_lr_factor=1):
        base_params = list(self.word_embeddings.parameters()) \
                    + list(self.position_embeddings.parameters()) \
                    + list(self.token_type_embeddings.parameters()) \
                    + list(self.linear.parameters()) \
                    + [p for l in self.layer for p in l.parameters()]
        
        cls_params = list(self.projection_cls.parameters())
        lm_params = list(self.projection_lm.parameters())

        # Retirer les doublons
        base_params = [p for p in base_params if id(p) not in {id(cp) for cp in cls_params + lm_params}]

        param_groups = [
            {'params': base_params, 'lr': base_lr},
            {'params': cls_params, 'lr': base_lr * cls_lr_factor},
            {'params': lm_params, 'lr': base_lr * lm_lr_factor},
        ]

        return torch.optim.Adam(param_groups)
