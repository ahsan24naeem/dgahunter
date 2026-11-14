import numpy as np
import torch

from src.models.charlm import CharGPT, CharLMConfig, batchify, char_surprise, score_texts


def test_charlm_shapes_and_scores():
    torch.manual_seed(0)
    m = CharGPT(CharLMConfig(d_model=32, n_layers=1, n_heads=2))
    x = batchify(["google", "abc"], 64)
    assert m(x).shape == (2, x.shape[1], m.cfg.vocab_size)
    s = score_texts(m, ["google", "abc"], "cpu")
    assert s.shape == (2,) and np.isfinite(s).all()
    assert len(char_surprise(m, "abc")) == 4  # 3 chars + <eos>


def test_charlm_is_causal():
    torch.manual_seed(0)
    m = CharGPT(CharLMConfig(d_model=32, n_layers=1, n_heads=2)).eval()
    a, b = batchify(["abcdef"], 64), batchify(["abcxyz"], 64)
    with torch.no_grad():
        assert torch.allclose(m(a)[0, :4], m(b)[0, :4], atol=1e-5)  # future chars must not leak


def test_canine_lora_tiny_forward():
    from src.models.canine_lora import TINY, _base_model, _peft, _tokenizer, predict_proba

    p = {"lora_r": 4, "lora_alpha": 8, "lora_dropout": 0.0, "target_modules": ["query", "value"]}
    model = _peft(_base_model(TINY), p)
    probs = predict_proba(model, _tokenizer(), ["google", "xkq9z2"], 64, "cpu")
    assert probs.shape == (2,) and ((probs >= 0) & (probs <= 1)).all()
