import torch

from kiji_inspector.experiments.ablation import make_ablation_hook


class _FakeSAE:
    """Minimal SAE test double: encode is identity, decode doubles.

    This isolates the hook's *mechanics* (pre-hook vs. post-hook, args vs.
    kwargs, decision-token slicing, feature zeroing) from real SAE math —
    the doubling makes it trivial to tell whether a tensor went through the
    encode/decode round trip at all.
    """

    def __init__(self):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.rms_scale = None

    def parameters(self):
        return iter([self._param])

    def encode(self, x):
        return x.clone()

    def decode(self, features):
        return features * 2


class _FakeLayer(torch.nn.Module):
    """Records exactly what it receives, then passes it straight through."""

    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, hidden_states=None):
        self.received = hidden_states
        return hidden_states


def test_ablation_hook_modifies_layer_input_not_output():
    """Regression test: the ablation hook must be a pre-hook.

    The SAE is trained on activations captured entering a layer (see
    activation_extractor.py's pre-hook fix), so the ablation intervention
    must patch the same tensor — the layer's *input* — not its output.
    A forward hook on the layer's output would intervene one layer
    downstream of where the SAE's features actually live.
    """
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])  # (batch=1, seq=1, d_model=2)
    output = layer(x)
    handle.remove()

    # The layer must receive the SAE round-tripped tensor (encode/decode -> *2),
    # not the raw input — proving the hook patched the layer's *input*.
    assert torch.equal(layer.received, x * 2)
    assert torch.equal(output, x * 2)
    assert not torch.equal(layer.received, x)


def test_ablation_hook_handles_hidden_states_kwarg():
    """Some HF decoder layers are invoked with hidden_states as a kwarg."""
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])
    layer(hidden_states=x)
    handle.remove()

    assert torch.equal(layer.received, x * 2)


def test_ablation_hook_decision_token_only_leaves_earlier_tokens_untouched():
    sae = _FakeSAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=None, decision_token_only=True),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (batch=1, seq=2, d_model=2)
    layer(x)
    handle.remove()

    received = layer.received
    assert torch.equal(received[:, 0, :], x[:, 0, :])
    assert torch.equal(received[:, 1, :], x[:, 1, :] * 2)


def test_ablation_hook_zeros_specified_features():
    class _IdentitySAE(_FakeSAE):
        def decode(self, features):
            return features  # no doubling — isolates feature zeroing

    sae = _IdentitySAE()
    layer = _FakeLayer()
    handle = layer.register_forward_pre_hook(
        make_ablation_hook(sae, feature_indices=[0], decision_token_only=False),
        with_kwargs=True,
    )

    x = torch.tensor([[[1.0, 2.0]]])
    layer(x)
    handle.remove()

    received = layer.received
    assert received[0, 0, 0].item() == 0.0
    assert received[0, 0, 1].item() == 2.0
