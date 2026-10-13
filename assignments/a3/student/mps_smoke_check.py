#!/usr/bin/env python3
"""Exercise the NMT forward, backward, checkpoint, and beam-search paths on Apple MPS."""

import tempfile
from pathlib import Path

import torch

from nmt_model import NMT
from utils import read_corpus
from vocab import Vocab


def main():
    mps_backend = getattr(torch.backends, 'mps', None)
    if mps_backend is None or not mps_backend.is_built():
        raise RuntimeError('This PyTorch build does not include MPS support.')
    if not mps_backend.is_available():
        raise RuntimeError(
            'PyTorch was built with MPS, but no MPS device is available in this process. '
            'Check the macOS/PyTorch installation before running this smoke check.'
        )

    device = torch.device('mps')
    torch.manual_seed(0)
    torch.mps.manual_seed(0)

    vocab = Vocab.load('vocab.json')
    source = [read_corpus('./zh_en_data/dev.zh', source='src', vocab_size=3000)[3]]
    target = [read_corpus('./zh_en_data/dev.en', source='tgt', vocab_size=2000)[3]]

    model = NMT(embed_size=32, hidden_size=32, dropout_rate=0.0, vocab=vocab).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    loss = -model(source, target).mean()
    if not torch.isfinite(loss).item():
        raise RuntimeError('MPS forward pass produced a non-finite loss.')
    loss.backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('MPS backward pass produced no parameter gradients.')
    optimizer.step()

    model.eval()
    hypotheses = model.beam_search(source[0], beam_size=2, max_decoding_time_step=5)
    if not hypotheses:
        raise RuntimeError('MPS beam search returned no hypotheses.')

    with tempfile.TemporaryDirectory(prefix='nmt-mps-smoke-') as temp_dir:
        checkpoint = Path(temp_dir) / 'model.bin'
        model.save(str(checkpoint))
        restored = NMT.load(str(checkpoint)).to(device)
        restored_hypotheses = restored.beam_search(
            source[0], beam_size=2, max_decoding_time_step=5
        )
        if not restored_hypotheses:
            raise RuntimeError('Reloaded MPS model returned no beam-search hypotheses.')

    print('MPS smoke check passed; loss=%.4f, top hypothesis=%s' % (
        loss.detach().cpu().item(), hypotheses[0].value
    ))


if __name__ == '__main__':
    main()
