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
  - Suspicion treated as 5-class categorical (one-hot float32 targets)
"""

import torch
from timm import create_model
from torch.nn import Dropout, Linear, Module, ReLU, Sequential


class MammoCNN(Module):
    def __init__(
        self,
        n_findings: int = 10,
        n_suspicion: int = 5,
        n_density: int = 4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

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
            Linear(feat_dim, 1024),
            ReLU(),
        )

        # ── Task heads ────────────────────────────────────────────────
        # All heads output raw logits; activations are applied in loss functions

        # Primary task
        self.diagnosis_head = Linear(1024, 1)  # malignant or benign

        # Auxiliary tasks
        self.findings_head = Linear(1024, n_findings)  # multilabel sigmoid
        self.suspicion_head = Linear(1024, n_suspicion)  # multiclass softmax
        self.density_head = Linear(1024, n_density)  # multiclass softmax
        self.age_head = Linear(1024, 1)  # regression mean squared error

        # Store config for unfreezing helpers
        self._backbone_layers = list(self.backbone.children())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) standardized mammogram tensor

        Returns:
            dict of raw logits / scalars per head
        """
        features = self.backbone(x)  # (B, 1536)
        shared_rep = self.shared_trunk(features)  # (B, 1024)

        diagnosis_logits = self.diagnosis_head(shared_rep).squeeze(1)  # (B,)
        findings_logits = self.findings_head(shared_rep)  # (B, n_findings)
        suspicion_logits = self.suspicion_head(shared_rep)  # (B, n_suspicion)
        density_logits = self.density_head(shared_rep)  # (B, n_density)
        age_pred = self.age_head(shared_rep).squeeze(1)  # (B,)

        return {
            "diagnosis": diagnosis_logits,
            "findings": findings_logits,
            "suspicion": suspicion_logits,
            "density": density_logits,
            "age": age_pred,
        }

    # ── Iterative unfreezing helpers ──────────────────────────────────
    def freeze_backbone(self) -> None:
        """
        Freeze the entire backbone, train only the head layers.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone_top_hald(self) -> None:
        """
        Unfreeze the top half of the backbone layers.
        """
        layers = self._backbone_layers
        midpoint = len(layers) // 2

        for layer in layers[midpoint:]:
            for param in layer.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """
        Unfreeze the entire backbone for fine-tuning.
        """
        for param in self.backbone.parameters():
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
    dummy_input = torch.randn(1, 3, 320, 416)  # (B, C, H, W)

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
