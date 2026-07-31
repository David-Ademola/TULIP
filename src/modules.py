import torch
from torch.nn import Linear, Module, Parameter
from torch.nn.functional import softplus


class CoralHead(Module):
    """
    Rank-consistent ordinal regression head (CORAL).

    For `n_classes` ordered levels it emits `n_classes - 1` cutpoint logits,
    where cutpoint k scores P(y > k). A single weight vector is shared across
    every cutpoint and only the biases differ, so the ordering of the cutpoint
    logits is identical for every input (rank consistency).

    The biases are additionally parameterised as a strictly decreasing sequence
      b_0,  b_0 - s_1,  b_0 - s_1 - s_2,  ...      with s_i = softplus(d_i) > 0
    which guarantees P(y > 1) >= P(y > 2) >= ... for *every* input, so the
    implied CDF can never be non-monotone — the property TULIP's urgency
    ranking depends on.
    """

    def __init__(self, in_features: int, n_classes: int) -> None:
        super().__init__()

        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")

        self.n_classes = n_classes
        self.n_cutpoints = n_classes - 1

        # Shared projection — no bias, the cutpoint biases play that role
        self.projection = Linear(in_features, 1, bias=False)

        # softplus(0.5413) ~= 1.0, so cutpoints start ~1 logit apart
        self.first_bias = Parameter(torch.zeros(1))
        self.bias_deltas = Parameter(torch.full((self.n_cutpoints - 1,), 0.5413))

    @property
    def biases(self) -> torch.Tensor:
        """
        Strictly decreasing cutpoint biases, shape (n_cutpoints,).
        """
        # pylint: disable = E1102
        steps = softplus(self.bias_deltas)
        offsets = torch.cat(
            [torch.zeros(1, device=steps.device, dtype=steps.dtype), steps.cumsum(0)]
        )
        return self.first_bias - offsets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_features)

        Returns:
            (B, n_cutpoints) logits; sigmoid(logits[:, k]) = P(y > k)
        """
        return self.projection(x) + self.biases
