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
