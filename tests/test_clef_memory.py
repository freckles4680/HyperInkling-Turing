"""The memory savings of scripts/clef_server.py change nothing they should not (CPU)."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import clef_server  # noqa: E402


def test_int8_rows():
    torch.manual_seed(0)
    weight = (torch.randn(5000, 256) * 0.02).half()
    rows = clef_server.Int8Rows(weight, chunk=1000)
    ids = torch.tensor([0, 17, 4999, 17])
    got, want = rows[ids].float(), weight[ids].float()
    assert rows[ids].dtype == torch.float16
    assert ((got - want).abs().amax(1) / want.abs().amax(1)).max() < 0.005
    assert rows.nbytes() < weight.numel() * 2 * 0.51


def test_delta_rule_split_is_bitwise():
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule

    layer = Layer()
    assert clef_server.split_delta_rule_heads(layer, 4) == 1
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 200, 12, 16) for _ in range(3))
    g, beta = -torch.rand(1, 200, 12), torch.rand(1, 200, 12)
    kwargs = dict(g=g, beta=beta, initial_state=None, output_final_state=True,
                  use_qk_l2norm_in_kernel=True, cu_seqlens=None)
    want = qwen.torch_chunk_gated_delta_rule(q, k, v, **kwargs)
    got = layer.chunk_gated_delta_rule(q, k, v, **kwargs)
    assert torch.equal(want[0], got[0]) and torch.equal(want[1], got[1])


def _tiny_text_model():
    qwen = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers import Qwen3_5TextConfig

    layers = 4
    config = Qwen3_5TextConfig(
        hidden_size=64, intermediate_size=96, num_hidden_layers=layers, num_attention_heads=6,
        num_key_value_heads=2, head_dim=16, vocab_size=101, full_attention_interval=4,
        layer_types=["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                     for i in range(layers)],
        attn_output_gate=True, linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
        partial_rotary_factor=0.25,
        rope_parameters={"rope_type": "default", "rope_theta": 1e4, "partial_rotary_factor": 0.25,
                         "mrope_section": [1, 1, 0], "mrope_interleaved": True})
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = qwen.Qwen3_5TextModel(config).eval()
    for p in model.parameters():  # random weights, not the zero/one inits
        p.data.normal_(0, 0.2)
    return model


def test_chunked_layers_match_one_pass():
    model = _tiny_text_model()
    clef_server.no_gqa_in_sdpa()
    ids = torch.randint(0, 101, (1, 300))
    calls = {"attention": 0, "delta": 0}
    grouped, state = clef_server._grouped_causal_sdpa, clef_server._OneLayerState

    def count_attention(*args):
        calls["attention"] += 1
        return grouped(*args)

    def count_delta(*args):
        calls["delta"] += 1
        return state(*args)

    clef_server._grouped_causal_sdpa, clef_server._OneLayerState = count_attention, count_delta
    with torch.inference_mode():
        want = model(input_ids=ids, use_cache=False).last_hidden_state
        counts = clef_server.chunk_layers(model, 64)
        clef_server.split_delta_rule_heads(model, 4)
        got = model(input_ids=ids, use_cache=False).last_hidden_state
    clef_server._grouped_causal_sdpa, clef_server._OneLayerState = grouped, state
    assert counts == {"mlp": 4, "delta": 3, "attention": 1}
    assert calls == {"attention": 5, "delta": 3}  # the chunked paths ran: 300 = 4x64 + 44
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()
