import os
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from .common import positional_encodings_like, INF, EPSILON, TransformerEncoder, TransformerDecoder, PackedLSTM, LSTMDecoderAttention, LSTMDecoder, Embedding, Feedforward, mask


class PointerGenerator(nn.Module):
  
    def __init__(self, field, args):
        super().__init__()
        self.field = field
        self.args = args
        self.pad_idx = self.field.vocab.stoi[self.field.pad_token]

        self.encoder_embeddings = Embedding(field, args.dimension, 
            dropout=args.dropout_ratio)
        self.decoder_embeddings = Embedding(field, args.dimension, 
            dropout=args.dropout_ratio)

     
        self.bilstm_before_coattention = PackedLSTM(args.dimension, args.dimension,
            batch_first=True, dropout=args.dropout_ratio, bidirectional=True, num_layers=args.rnn_layers)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(args.dimension, args.dimension,
            dropout=args.dropout_ratio, num_layers=args.rnn_layers)

        self.generative_vocab_size = min(len(field.vocab), args.max_generative_vocab)
        self.out = nn.Linear(args.dimension, self.generative_vocab_size)

        self.dropout = nn.Dropout(0.4)

    def set_embeddings(self, embeddings):
        self.encoder_embeddings.set_embeddings(embeddings)
        self.decoder_embeddings.set_embeddings(embeddings)


    def forward(self, batch):
        context, context_lengths, context_limited    = batch.context_question,  batch.context_question_lengths,  batch.context_question_limited
        answer, answer_lengths, answer_limited       = batch.answer,   batch.answer_lengths,   batch.answer_limited
        oov_to_limited_idx, limited_idx_to_full_idx  = batch.oov_to_limited_idx, batch.limited_idx_to_full_idx

        def map_to_full(x):
            return limited_idx_to_full_idx[x]
        self.map_to_full = map_to_full

        context_embedded = self.encoder_embeddings(context)

        context_encoded, (context_rnn_h, context_rnn_c) = self.bilstm_before_coattention(context_embedded, context_lengths)
        context_rnn_state = [self.reshape_rnn_state(x) for x in (context_rnn_h, context_rnn_c)]

        context_padding = context.data == self.pad_idx
        context_indices = context_limited if context_limited is not None else context
        answer_indices = answer_limited if answer_limited is not None else answer

        pad_idx = self.field.decoder_stoi[self.field.pad_token]
        context_padding = context_indices.data == pad_idx
        self.dual_ptr_rnn_decoder.applyMasks(context_padding)

        if self.training:
            answer_embedded = self.decoder_embeddings(answer)
            decoder_outputs = self.dual_ptr_rnn_decoder(answer_embedded[:, :-1].contiguous(),
                context_encoded, hidden=context_rnn_state)
            rnn_output, context_attention, context_alignment, vocab_pointer_switch, rnn_state = decoder_outputs

            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, 
                context_attention, 
                context_indices, 
                oov_to_limited_idx)

            probs, targets = mask(answer_indices[:, 1:].contiguous(), probs.contiguous(), pad_idx=pad_idx)
            loss = F.nll_loss(probs.log(), targets)
            return loss, None
        else:
            return None, self.greedy(context_encoded, 
                context_indices,
                oov_to_limited_idx, rnn_state=context_rnn_state).data
 
    def reshape_rnn_state(self, h):
        return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2).contiguous()

    def probs(self, generator, outputs, vocab_pointer_switches, 
        context_attention, 
        context_indices, 
        oov_to_limited_idx):


        size = list(outputs.size())

        size[-1] = self.generative_vocab_size
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.generative_vocab_size + len(oov_to_limited_idx)
        if self.generative_vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        p_context_ptr = scaled_p_vocab.new_full(scaled_p_vocab.size(), EPSILON)
        p_context_ptr.scatter_add_(p_context_ptr.dim()-1, context_indices.unsqueeze(1).expand_as(context_attention), context_attention)
        scaled_p_context_ptr = (1 - vocab_pointer_switches).expand_as(p_context_ptr) * p_context_ptr
        probs = scaled_p_vocab + scaled_p_context_ptr
        return probs


    def greedy(self, context, context_indices, oov_to_limited_idx, rnn_state=None):
        B, TC, C = context.size()
        T = self.args.max_output_length
        outs = context.new_full((B, T), self.field.decoder_stoi['<pad>'], dtype=torch.long)
        eos_yet = context.data.new(B).byte().zero_()

        rnn_output, context_alignment = None, None
        for t in range(T):
            if t == 0:
                embedding = self.decoder_embeddings(
                    context[-1].new_full((B, 1), self.field.vocab.stoi['<init>'], dtype=torch.long), [1]*B)

            else:
                embedding = self.decoder_embeddings(outs[:, t - 1].unsqueeze(1), [1]*B)
            decoder_outputs = self.dual_ptr_rnn_decoder(embedding, #hiddens[-1][:, t].unsqueeze(1),
                context, 
                context_alignment=context_alignment,
                hidden=rnn_state, output=rnn_output)

            rnn_output, context_attention, context_alignment, vocab_pointer_switch, rnn_state = decoder_outputs
            probs = self.probs(self.out, rnn_output, vocab_pointer_switch, 
                context_attention, 
                context_indices, 
                oov_to_limited_idx)
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            eos_yet = eos_yet | (preds == self.field.decoder_stoi['<eos>'])
            outs[:, t] = preds.cpu().apply_(self.map_to_full)
            if eos_yet.all():
                break
        return outs


class DualPtrRNNDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.input_feed = True
        if self.input_feed:
            d_in += 1 * d_hid

        self.rnn = LSTMDecoder(self.num_layers, d_in, d_hid, dropout)
        self.context_attn = LSTMDecoderAttention(d_hid, dot=True)

        self.vocab_pointer_switch = nn.Sequential(Feedforward(2 * self.d_hid + d_in, 1), nn.Sigmoid())

    def forward(self, input, context, output=None, hidden=None, context_alignment=None):
        context_output = output.squeeze(1) if output is not None else self.make_init_output(context)
        context_alignment = context_alignment if context_alignment is not None else self.make_init_output(context)

        context_outputs, vocab_pointer_switches, context_attentions, context_alignments = [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_output = self.dropout(context_output)
            if self.input_feed:
                emb_t = torch.cat([emb_t, context_output], 1)
            dec_state, hidden = self.rnn(emb_t, hidden)
            context_output, context_attention, context_alignment = self.context_attn(dec_state, context)
            vocab_pointer_switch = self.vocab_pointer_switch(torch.cat([dec_state, context_output, emb_t], -1))
            context_output = self.dropout(context_output)
            context_outputs.append(context_output)
            vocab_pointer_switches.append(vocab_pointer_switch)
            context_attentions.append(context_attention)
            context_alignments.append(context_alignment)
        context_outputs, vocab_pointer_switches, context_attention = [self.package_outputs(x) for x in [context_outputs, vocab_pointer_switches, context_attentions]]
        return context_outputs, context_attention, context_alignment, vocab_pointer_switches, hidden


    def applyMasks(self, context_mask):
        self.context_attn.applyMasks(context_mask)

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.d_hid)
        return context.new_zeros(h_size)

    def package_outputs(self, outputs):
        outputs = torch.stack(outputs, dim=1)
        return outputs
