#!/usr/bin/env python3
"""Forward, padding, beam-search, and checkpoint checks for attention modes."""

import torch
import torch.nn.functional as F

from nmt_model import NMT
from vocab import Vocab


def _tiny_model(attention_type: str) -> NMT:
    vocab = Vocab.load('vocab.json')
    model = NMT(
        embed_size=256,
        hidden_size=256,
        dropout_rate=0.3,
        vocab=vocab,
        attention_type=attention_type,
    )
    model.eval()
    return model


def check_attention_type(attention_type: str) -> None:
    model = _tiny_model(attention_type)
    batch_size = 2
    src_len = 5
    hidden = 256

    enc_hiddens = torch.randn(batch_size, src_len, 2 * hidden)
    enc_hiddens_proj = model._prepare_enc_hiddens_for_attention(enc_hiddens)
    Ybar_t = torch.randn(batch_size, 256 + hidden)
    dec_state = (torch.randn(batch_size, hidden), torch.randn(batch_size, hidden))
    enc_masks = torch.zeros(batch_size, src_len)
    enc_masks[0, 3:] = 1

    _, o_t, e_t_step = model.step(
        Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks
    )
    alpha_t = F.softmax(e_t_step, dim=-1)

    assert e_t_step.shape == (batch_size, src_len), f'{attention_type}: bad e_t shape {e_t_step.shape}'
    assert o_t.shape == (batch_size, hidden), f'{attention_type}: bad o_t shape {o_t.shape}'
    assert not torch.isnan(e_t_step).any(), f'{attention_type}: NaN in e_t'
    assert not torch.isnan(o_t).any(), f'{attention_type}: NaN in o_t'
    assert torch.all(alpha_t[0, 3:] == 0), f'{attention_type}: pad softmax not zero: {alpha_t[0, 3:]}'
    assert torch.isfinite(alpha_t).all(), f'{attention_type}: non-finite softmax'

    src_sent = [model.vocab.src.id2word[i] for i in range(4, 8)]
    hyps = model.beam_search(src_sent, beam_size=2, max_decoding_time_step=5)
    assert len(hyps) >= 1, f'{attention_type}: beam_search returned no hypotheses'
    print(f'{attention_type}: e_t {tuple(e_t_step.shape)}, o_t {tuple(o_t.shape)}, pad softmax 0, beam ok')


def check_checkpoint_compat() -> None:
    mul = _tiny_model('multiplicative')
    add = _tiny_model('additive')
    assert 'att_enc_proj.weight' not in mul.state_dict()
    assert 'att_enc_proj.weight' in add.state_dict()

    old_args = dict(
        embed_size=mul.model_embeddings.embed_size,
        hidden_size=mul.hidden_size,
        dropout_rate=mul.dropout_rate,
    )
    old_args.setdefault('attention_type', 'multiplicative')
    loaded = NMT(vocab=mul.vocab, **old_args)
    loaded.load_state_dict(mul.state_dict())
    assert loaded.attention_type == 'multiplicative'
    print('old multiplicative checkpoint loads without additive params')


if __name__ == '__main__':
    for mode in ('multiplicative', 'dot', 'additive'):
        check_attention_type(mode)
    check_checkpoint_compat()
    print('Forward checks passed.')
