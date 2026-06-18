import torch
import torch.nn as nn

from models.pointnet_backbone import PointNetBackbone, LocalKNNBackbone
from models.temporal_snn import TemporalSNN, BidirectionalTemporalSNN


class PointNetSNN(nn.Module):
    """
    Extended SNN-PointNet model with novel features from SPM paper + notes:

    Novel additions vs original:
      1. learnable_lif   — trainable tau (leak) and V_th (threshold) per neuron
      2. local_knn       — KNN neighbourhood embedding in backbone (like SPM SEL)
      3. bidirectional   — bidirectional temporal processing (like SPM Time Flip)
      4. num_classes     — now properly parameterised (was hardcoded to 10);
                           set to 40 for ModelNet40.

    Architecture flow:
        Point cloud slice [B, n, 3]
             |
        [Backbone]  PointNetBackbone  OR  LocalKNNBackbone
             |        per-point spiking MLP      KNN + spiking MLP
             v
        Mean pooling -> slice embedding [B, D]
             |
        [Temporal] TemporalSNN  OR  BidirectionalTemporalSNN
             v
        logits [B, num_classes]
    """

    def __init__(self,
                 point_dims=[128, 256, 512],
                 temporal_dim=512,
                 num_classes=10,
                 learnable_lif=False,
                 local_knn=False,
                 knn_k=16,
                 bidirectional=False,
                 use_bn=False):
        super().__init__()

        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # Backbone
        if local_knn:
            self.backbone = LocalKNNBackbone(
                hidden_dims=point_dims, k=knn_k, learnable_lif=learnable_lif, use_bn=use_bn
            )
        else:
            self.backbone = PointNetBackbone(
                hidden_dims=point_dims, learnable_lif=learnable_lif, use_bn=use_bn
            )

        # Temporal
        if bidirectional:
            self.temporal = BidirectionalTemporalSNN(
                dim=temporal_dim, num_classes=num_classes, learnable_lif=learnable_lif, use_bn=use_bn
            )
        else:
            self.temporal = TemporalSNN(
                dim=temporal_dim, num_classes=num_classes, learnable_lif=learnable_lif, use_bn=use_bn
            )

    def reset_state(self, batch_size, device=None):
        self.backbone.reset_state(batch_size, device)
        self.temporal.reset_state(batch_size, device)

    def forward_step(self, pts_slice):
        """
        Process one slice of points.
        pts_slice : [B, n_points, 3]
        Returns   : logits_t [B, num_classes]
        """
        per_point_feat = self.backbone(pts_slice)       # [B, n, D]
        slice_feat = per_point_feat.mean(dim=1)         # [B, D]

        if self.bidirectional:
            # Store feature and return forward-only logits for aux loss
            return self.temporal.forward_step(slice_feat)
        else:
            return self.temporal(slice_feat)

    def forward_step_feat(self, feat):
        """
        Process a precomputed backbone embedding through the temporal head.
        Used by ASPWrapper, which precomputes all backbone embeddings upfront
        and feeds them in SSP-selected order.

        Args:
            feat : [B, temporal_dim]  precomputed backbone mean-pool embedding

        Returns:
            logits : [B, num_classes]
        """
        if self.bidirectional:
            return self.temporal.forward_step(feat)
        else:
            return self.temporal(feat)

    def finalize(self):
        """
        For bidirectional mode: call after all forward_step() calls to run
        the backward pass and return fused final logits.
        """
        if self.bidirectional:
            return self.temporal.classify()
        raise RuntimeError("finalize() only for bidirectional mode")

    def forward_full(self, pts):
        """Process the full point cloud as a single time step."""
        per_point_feat = self.backbone(pts)
        slice_feat = per_point_feat.mean(dim=1)
        if self.bidirectional:
            # For full mode, use single forward pass
            return self.temporal.forward(slice_feat)
        return self.temporal(slice_feat)

    def get_firing_rates(self):
        """
        Collect firing rates from all LearnableLIF layers.
        Returns a dict {layer_name: rate} for efficiency analysis.
        (From your notes: 'total spike rate = all spikes / total no. of neurons')
        """
        rates = {}
        if hasattr(self.backbone, 'firing_rates'):
            rates.update(self.backbone.firing_rates())
        if hasattr(self.temporal, 'firing_rates'):
            rates.update(self.temporal.firing_rates())
        return rates
