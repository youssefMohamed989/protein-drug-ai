import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.data.protein_features import PAD_IDX, VOCAB, encode_sequence, encode_ss8


def test_encode_sequence_length_and_padding():
    seq = "ACDEFG"
    encoded = encode_sequence(seq, max_len=10)
    assert encoded.shape == (10,)
    assert (encoded[6:] == PAD_IDX).all()


def test_encode_sequence_truncation():
    seq = "A" * 20
    encoded = encode_sequence(seq, max_len=5)
    assert encoded.shape == (5,)


def test_encode_ss8_ignore_index_on_pad():
    labels = "HHHEEE"
    encoded = encode_ss8(labels, max_len=10)
    assert encoded.shape == (10,)
    assert (encoded[6:] == -100).all()


def test_vocab_has_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))
