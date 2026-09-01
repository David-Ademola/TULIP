"""
TULIP — MAMMO CNN Phase 1 (Single-View MTL Network)
Based on: "MAMMO: A Deep Learning Solution for Facilitating
Radiologist-Machine Collaboration in Breast Cancer Diagnosis"
Kyono, Gilbert, van der Schaar (2018)

Deviations from paper:
  - No conspicuity head (not in dataset)
  - Findings head is multilabel (sigmoid) not multiclass (softmax)
    because a single mammogram can have multiple co-occurring findings
  - Density treated as 4-class categorical (BI-RADS A-D) based on
    dataset structure
  - Suspicion is an ORDINAL head (CORAL), not a softmax over 5 classes.
    BI-RADS is ordered, so predicting 2 for a true 5 must cost more than
    predicting 4. CORAL also yields P(BI-RADS > k) directly, which gives
    TULIP a monotone, ranking-ready score for the Abnormality Index.
    Reference: Cao, Mirjalili, Raschka (2020), "Rank consistent ordinal
    regression for neural network with application to age estimation"
"""

import torch
from timm import create_model
from torch.nn import Dropout, Linear, Module, ReLU, Sequential

from src.modules import CoralHead


class MammoCNN(Module):
    """
    Multi-task CNN for single-view mammogram classification.

    Primary task: Diagnosis (malignant vs benign)
    Auxiliary tasks: Findings, Suspicion (ordinal), Density, Age
    """

    def __init__(
        self,
        n_findings: int = 10,
        n_suspicion: int = 5,
        n_density: int = 4,
        pretrained: bool = True,
        birads_min: int = 1,
        malignant_birads: int = 4,
    ) -> None:
        super().__init__()

        # diagnosis == (BI-RADS >= malignant_birads), so on the 0-based ordinal
        # scale it is exactly the cutpoint P(y > malignant_birads - birads_min - 1).
        # Exposed as "diagnosis_ordinal" so the dedicated diagnosis head and the
        # ordinal head can be compared on the same quantity.
        self._diagnosis_cutpoint = malignant_birads - birads_min - 1
        if not 0 <= self._diagnosis_cutpoint < n_suspicion - 1:
            raise ValueError(
                f"malignant_birads={malignant_birads} is outside the ordinal scale "
                f"[{birads_min}, {birads_min + n_suspicion - 1}]"
            )

        # ── Backbone ──────────────────────────────────────────────────
        # InceptionResNetV2: 1536D features after global average pooling
        self.backbone = create_model(
            model_name="inception_resnet_v2",
            pretrained=pretrained,
            num_classes=0,  # No classification head
            global_pool="avg",  # Global average pooling to get 1536D features
        )
        feat_dim = self.backbone.num_features  # 1536

        # ── Shared dense trunk ────────────────────────────────────────
        self.shared_trunk = Sequential(
            Dropout(p=0.2),
            Linear(feat_dim, 1024),  # type: ignore
            ReLU(),
        )

        # ── Task heads ────────────────────────────────────────────────
        # All heads output raw logits; activations are applied in loss functions

        # Primary task
        self.diagnosis_head = Linear(1024, 1)  # malignant or benign

        # Auxiliary tasks
        self.findings_head = Linear(1024, n_findings)  # multilabel sigmoid
        self.suspicion_head = CoralHead(1024, n_suspicion)  # ordinal, 4 cutpoints
        self.density_head = Linear(1024, n_density)  # multiclass softmax
        self.age_head = Linear(1024, 1)  # regression mean squared error

        # Store config for unfreezing helpers
        self._backbone_layers = list(self.backbone.children())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) standardized mammogram tensor

        Returns:
            dict of raw logits / scalars per head. "suspicion" holds ordinal
            cutpoint logits where sigmoid(suspicion[:, k]) = P(y > k) on the
            0-based BI-RADS scale. "diagnosis_ordinal" is the cutpoint that
            corresponds to the diagnosis label — no extra parameters, just a
            slice, so it can be compared against the dedicated diagnosis head.
        """
        features = self.backbone(x)  # (B, 1536)
        shared_rep = self.shared_trunk(features)  # (B, 1024)

        diagnosis_logits = self.diagnosis_head(shared_rep)  # (B, 1)
        findings_logits = self.findings_head(shared_rep)  # (B, n_findings)
        suspicion_logits = self.suspicion_head(shared_rep)  # (B, n_suspicion - 1)
        density_logits = self.density_head(shared_rep)  # (B, n_density)
        age_pred = self.age_head(shared_rep).squeeze(1)  # (B,)

        cutpoint = self._diagnosis_cutpoint
        return {
            "diagnosis": diagnosis_logits,
            "findings": findings_logits,
            "suspicion": suspicion_logits,
            "density": density_logits,
            "age": age_pred,
            "diagnosis_ordinal": suspicion_logits[:, cutpoint : cutpoint + 1],  # (B, 1)
        }

    @staticmethod
    def suspicion_to_level(suspicion_logits: torch.Tensor) -> torch.Tensor:
        """
        Decode ordinal cutpoint logits to a 0-based level index.

        Args:
            suspicion_logits: (B, n_cutpoints) raw cutpoint logits

        Returns:
            (B,) long tensor of predicted levels in [0, n_cutpoints]
        """
        return (suspicion_logits > 0).sum(dim=1)

    # ── Iterative freezing helpers ──────────────────────────────────
    def freeze_backbone(self) -> None:
        """
        Freeze the entire backbone, train only the head layers.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def freeze_head(self) -> None:
        """
        Freeze the head layers, train only the backbone.
        """
        for param in self.shared_trunk.parameters():
            param.requires_grad = False

        for head in [
            self.diagnosis_head,
            self.findings_head,
            self.suspicion_head,
            self.density_head,
            self.age_head,
        ]:
            for param in head.parameters():
                param.requires_grad = False

    def unfreeze_all(self) -> None:
        """
        Unfreeze the entire model
        """
        for param in self.parameters():
            param.requires_grad = True

    def get_parameter_counts(self) -> dict[str, int]:
        """
        Get the number of total and trainable parameters in the model.

        Returns:
            dict with keys 'total' and 'trainable' containing parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total_params, "trainable": trainable_params}


if __name__ == "__main__":
    # Quick sanity check
    model = MammoCNN(pretrained=True).eval()
    dummy_input = torch.randn(1, 3, 720, 1280)  # (B, C, H, W)

    with torch.no_grad():
        outputs = model(dummy_input)

    for head_name, output in outputs.items():
        print(f"{head_name}: {output.shape}")

    # Number of parameters
    param_count = model.get_parameter_counts()
    num_params = param_count["total"]
    num_trainable_params = param_count["trainable"]

    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable_params:,}")

    input("Press Enter to exit...")
