from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import torch
from torch import nn
from algorithm.reference_dtqn import ReferenceDTQN


@dataclass(frozen=True)
#图结构
class GraphLayout:
    slot_counts: Tuple[int, ...]

    @property
    def type_count(self) -> int:
        return len(self.slot_counts)

    @property
    def total_slots(self) -> int:
        return sum(self.slot_counts)

    @property
    def node_count(self) -> int:
        return 2 + self.type_count + 2 * self.total_slots

    @property
    def slot_type_indices(self) -> Tuple[int, ...]:
        return tuple(
            type_index
            for type_index, count in enumerate(self.slot_counts)
            for _ in range(count)
        )

    @classmethod
    def resolve(
        cls,
        node_count: int,
        action_count: int,
        slot_counts: Optional[Sequence[int]] = None,
    ) -> "GraphLayout":
        if action_count < 3 or (action_count - 1) % 2:
            raise ValueError("action_count must follow 1 + 2 * instance_type_count")
        type_count = (action_count - 1) // 2
        if slot_counts is None:
            lifecycle_nodes = int(node_count) - 2 - type_count
            divisor = 2 * type_count
            if lifecycle_nodes <= 0 or lifecycle_nodes % divisor:
                raise ValueError(
                    "cannot infer equal instance slots from node_count/action_count"
                )
            per_type = lifecycle_nodes // divisor
            resolved = (per_type,) * type_count
        else:
            resolved = tuple(int(value) for value in slot_counts)
        if len(resolved) != type_count or any(value <= 0 for value in resolved):
            raise ValueError(
                "instance_slot_counts must contain one positive value per type"
            )
        layout = cls(resolved)
        if layout.node_count != int(node_count):
            raise ValueError(
                f"graph layout expects {layout.node_count} nodes, got {node_count}"
            )
        return layout

#节点特征预处理
class GraphFeatureScaling(nn.Module):
    def __init__(self, layout: GraphLayout, feature_dim: int):
        super().__init__()
        scales = torch.ones(layout.node_count, feature_dim)
        logarithmic = torch.zeros_like(scales, dtype=torch.bool)
        if feature_dim == 2:
            scales[0] = torch.tensor([900., 1.])
            scales[1] = torch.tensor([16384., 1024.])
            logarithmic[1] = True
            for i, count in enumerate(layout.slot_counts):
                scales[2 + i] = float(count)
            cold_start = 2 + layout.type_count
            running_start = cold_start + layout.total_slots
            scales[cold_start:running_start, 0] = 900.
            scales[running_start:] = 4096.
            logarithmic[running_start:] = True
        self.register_buffer("scales", scales)
        self.register_buffer("logarithmic", logarithmic)

    def forward(self, nodes: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
                                                                         
        values = torch.where(node_mask.bool().unsqueeze(-1), nodes,
                             torch.zeros_like(nodes)) / self.scales
                                                                           
        logged = torch.log1p(values.clamp_min(0))
        return torch.where(self.logarithmic, logged, values)

#节点
class MaskedRelationAttention(nn.Module):
    def __init__(self, hidden_dim: int, heads: int):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by attention heads")
        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = self.hidden_dim // self.heads
        self.scale = self.head_dim**-0.5
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def _heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = values.shape
        return values.reshape(
            batch, nodes, self.heads, self.head_dim
        ).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        allowed: torch.Tensor,
        key_valid: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        q = self._heads(self.query(self.query_norm(queries)))
        k = self._heads(self.key(self.key_norm(keys)))
        v = self._heads(self.value(keys))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        relation = allowed.to(device=scores.device, dtype=torch.bool)
        valid = relation[None, None] & key_valid[:, None, None, :].bool()
        safe_scores = scores.masked_fill(~valid, -1e9)
        row_max = safe_scores.max(dim=-1, keepdim=True).values
        has_neighbor = valid.any(dim=-1, keepdim=True)
        row_max = torch.where(has_neighbor, row_max, torch.zeros_like(row_max))
        weights = torch.exp(safe_scores - row_max) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        message = torch.matmul(weights, v).transpose(1, 2).reshape(
            queries.shape[0], queries.shape[1], self.hidden_dim
        )
        message = self.output(message)
        valid_output = query_valid & valid.any(dim=-1).squeeze(1)
        return message * valid_output.unsqueeze(-1).to(message.dtype)
#语义
class HeterogeneousGraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, layout: GraphLayout):
        super().__init__()
        self.layout = layout
        self.load_center_attention = MaskedRelationAttention(hidden_dim, heads)
        self.center_type_attention = MaskedRelationAttention(hidden_dim, heads)
        self.type_cold_attention = MaskedRelationAttention(hidden_dim, heads)
        self.type_running_attention = MaskedRelationAttention(hidden_dim, heads)
        self.center_semantic_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.type_semantic_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.center_norm = nn.LayerNorm(hidden_dim)
        self.load_norm = nn.LayerNorm(hidden_dim)
        self.type_norm = nn.LayerNorm(hidden_dim)
        self.cold_norm = nn.LayerNorm(hidden_dim)
        self.running_norm = nn.LayerNorm(hidden_dim)
        slot_types = torch.as_tensor(layout.slot_type_indices, dtype=torch.long)
        self.register_buffer(
            "type_slot_allowed",
            torch.nn.functional.one_hot(
                slot_types, num_classes=layout.type_count
            ).transpose(0, 1).bool(),
            persistent=False,
        )

    @staticmethod
    def _fuse_semantics(
        views: torch.Tensor,
        valid: torch.Tensor,
        scorer: nn.Module,
    ) -> torch.Tensor:
        scores = scorer(views).squeeze(-1).masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1) * valid.to(scores.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return (weights.unsqueeze(-1) * views).sum(dim=2)

    def forward(
        self,
        center: torch.Tensor,
        load: torch.Tensor,
        types: torch.Tensor,
        cold: torch.Tensor,
        running: torch.Tensor,
        cold_valid: torch.Tensor,
        running_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = center.shape[0]
        center_valid = torch.ones(
            batch, 1, dtype=torch.bool, device=center.device
        )
        type_valid = torch.ones(
            batch, self.layout.type_count, dtype=torch.bool, device=center.device
        )
        one_to_one = torch.ones(1, 1, dtype=torch.bool, device=center.device)
        center_to_types = torch.ones(
            1, self.layout.type_count, dtype=torch.bool, device=center.device
        )
        type_to_center = center_to_types.transpose(0, 1)
        type_to_slots = self.type_slot_allowed.to(center.device)

        center_from_load = self.load_center_attention(
            center, load, one_to_one, center_valid, center_valid
        )
        load_from_center = self.load_center_attention(
            load, center, one_to_one, center_valid, center_valid
        )
        center_from_types = self.center_type_attention(
            center, types, center_to_types, type_valid, center_valid
        )

        types_from_center = self.center_type_attention(
            types, center, type_to_center, center_valid, type_valid
        )
        types_from_cold = self.type_cold_attention(
            types, cold, type_to_slots, cold_valid, type_valid
        )
        cold_from_types = self.type_cold_attention(
            cold,
            types,
            type_to_slots.transpose(0, 1),
            type_valid,
            cold_valid,
        )
        types_from_running = self.type_running_attention(
            types, running, type_to_slots, running_valid, type_valid
        )
        running_from_types = self.type_running_attention(
            running,
            types,
            type_to_slots.transpose(0, 1),
            type_valid,
            running_valid,
        )

        center_views = torch.stack((center_from_load, center_from_types), dim=2)
        center_view_valid = torch.ones(
            batch, 1, 2, dtype=torch.bool, device=center.device
        )
        center_message = self._fuse_semantics(
            center_views, center_view_valid, self.center_semantic_score
        )

        cold_by_type = torch.einsum(
            "ns,bs->bn", type_to_slots.float(), cold_valid.float()
        ).bool()
        running_by_type = torch.einsum(
            "ns,bs->bn", type_to_slots.float(), running_valid.float()
        ).bool()
        type_views = torch.stack(
            (types_from_center, types_from_cold, types_from_running), dim=2
        )
        type_view_valid = torch.stack(
            (type_valid, cold_by_type, running_by_type), dim=2
        )
        type_message = self._fuse_semantics(
            type_views, type_view_valid, self.type_semantic_score
        )

        cold_mask = cold_valid.unsqueeze(-1).to(cold.dtype)
        running_mask = running_valid.unsqueeze(-1).to(running.dtype)
        return (
            self.center_norm(center + center_message),
            self.load_norm(load + load_from_center),
            self.type_norm(types + type_message),
            self.cold_norm(cold + cold_from_types) * cold_mask,
            self.running_norm(running + running_from_types) * running_mask,
        )

class HANEncoder(nn.Module):
    def __init__(
        self,
        node_count: int,
        feature_dim: int,
        hidden_dim: int,
        heads: int,
        action_count: int,
        instance_slot_counts: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.layout = GraphLayout.resolve(
            node_count, action_count, instance_slot_counts
        )
        self.node_count = int(node_count)
        self.feature_scaling = GraphFeatureScaling(self.layout, feature_dim)
        self.center_projection = nn.Linear(feature_dim, hidden_dim)
        self.load_projection = nn.Linear(feature_dim, hidden_dim)
        self.type_projection = nn.Linear(feature_dim, hidden_dim)
        self.cold_projection = nn.Linear(feature_dim, hidden_dim)
        self.running_projection = nn.Linear(feature_dim, hidden_dim)
        self.instance_type_embedding = nn.Parameter(
            torch.zeros(1, self.layout.type_count, hidden_dim)
        )
        self.layers = nn.ModuleList(
            [
                HeterogeneousGraphAttentionLayer(hidden_dim, heads, self.layout)
                for _ in range(2)
            ]
        )
        self.register_buffer(
            "slot_type_indices",
            torch.as_tensor(self.layout.slot_type_indices, dtype=torch.long),
            persistent=False,
        )
        nn.init.normal_(self.instance_type_embedding, std=0.02)

    def forward(
        self, nodes: torch.Tensor, node_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if nodes.ndim != 3 or nodes.shape[1] != self.node_count:
            raise ValueError(
                f"expected [batch,{self.node_count},features], got {tuple(nodes.shape)}"
            )
        if node_mask is None:
            node_mask = torch.ones(
                nodes.shape[:2], dtype=torch.bool, device=nodes.device
            )
        if node_mask.shape != nodes.shape[:2]:
            raise ValueError("node_mask shape must match [batch,node_count]")

        nodes = self.feature_scaling(nodes, node_mask)

        count = self.layout.type_count
        slots = self.layout.total_slots
        type_start = 2
        cold_start = type_start + count
        running_start = cold_start + slots
        center = self.center_projection(nodes[:, :1])
        load = self.load_projection(nodes[:, 1:2])
        types = self.type_projection(nodes[:, type_start:cold_start])
        types = types + self.instance_type_embedding
        slot_embeddings = self.instance_type_embedding[:, self.slot_type_indices]
        cold_valid = node_mask[:, cold_start:running_start].bool()
        running_valid = node_mask[:, running_start:].bool()
        cold = self.cold_projection(nodes[:, cold_start:running_start])
        cold = (cold + slot_embeddings) * cold_valid.unsqueeze(-1).to(cold.dtype)
        running = self.running_projection(nodes[:, running_start:])
        running = (
            running + slot_embeddings
        ) * running_valid.unsqueeze(-1).to(running.dtype)

        for layer in self.layers:
            center, load, types, cold, running = layer(
                center,
                load,
                types,
                cold,
                running,
                cold_valid,
                running_valid,
            )

        return center.squeeze(1)

class GraphObservationEmbedding(nn.Module):
    def __init__(
        self,
        graph_encoder: nn.Module,
        node_count: int,
        feature_dim: int,
        action_count: int,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.node_count = int(node_count)
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.graph_observation_dim = self.node_count * self.feature_dim
        self.node_mask_offset = self.graph_observation_dim
        self.action_mask_offset = self.node_mask_offset + self.node_count
        self.observation_dim = self.action_mask_offset + self.action_count

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "reference DTQN observations must have shape "
                f"[batch,time,{self.observation_dim}]"
            )
        batch, steps, _ = observations.shape
        nodes = observations[..., : self.graph_observation_dim].reshape(
            batch * steps, self.node_count, self.feature_dim
        )
        node_mask = observations[
            ..., self.node_mask_offset : self.action_mask_offset
        ].reshape(batch * steps, self.node_count).bool()
        center_embedding = self.graph_encoder(nodes, node_mask)
        return center_embedding.reshape(batch, steps, -1)

class TemporalMaskedDTQN(ReferenceDTQN):
    def __init__(
        self,
        graph_encoder: nn.Module,
        node_count: int,
        feature_dim: int,
        action_count: int,
        instance_slot_counts: Optional[Sequence[int]] = None,
        hidden_dim: int = 128,
        attention_heads: int = 8,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.0,
        max_context: int = 50,
    ):
        self.layout = GraphLayout.resolve(
            node_count, action_count, instance_slot_counts
        )
        self.node_count = int(node_count)
        self.feature_dim = int(feature_dim)
        self.action_count = int(action_count)
        self.graph_observation_dim = self.node_count * self.feature_dim
        self.node_mask_offset = self.graph_observation_dim
        self.action_mask_offset = self.node_mask_offset + self.node_count
        self.combined_observation_dim = self.action_mask_offset + self.action_count
        super().__init__(
            obs_dim=self.combined_observation_dim,
            num_actions=self.action_count,
            embed_per_obs_dim=1,
            action_dim=0,
            inner_embed_size=hidden_dim,
            num_heads=attention_heads,
            num_layers=transformer_layers,
            history_len=max_context,
            dropout=transformer_dropout,
            gate="res",
            identity=False,
            pos="learned",
            discrete=False,
            bag_size=0,
        )
        self.obs_embedding = GraphObservationEmbedding(
            graph_encoder=graph_encoder,
            node_count=self.node_count,
            feature_dim=self.feature_dim,
            action_count=self.action_count,
        )

    @property
    def graph_encoder(self) -> nn.Module:
        return self.obs_embedding.graph_encoder

    def _flatten_graph_context(
        self,
        context: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        node_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if context.ndim != 4:
            return context
        batch, steps, nodes, features = context.shape
        if (nodes, features) != (self.node_count, self.feature_dim):
            raise ValueError(
                f"expected graph context [batch,time,{self.node_count},{self.feature_dim}]"
            )
        node_masks = torch.ones(
            batch,
            steps,
            self.node_count,
            dtype=context.dtype,
            device=context.device,
        )
        if node_mask is not None:
            if node_mask.ndim == 2:
                node_masks = node_mask[:, None].expand(-1, steps, -1).to(context.dtype)
            elif node_mask.ndim == 3:
                node_masks = node_mask.to(context.dtype)
            else:
                raise ValueError("node_mask must have shape [B,N] or [B,T,N]")
        action_masks = torch.ones(
            batch,
            steps,
            self.action_count,
            dtype=context.dtype,
            device=context.device,
        )
        if action_mask is not None:
            if action_mask.ndim == 2:
                action_masks[:, -1] = action_mask.to(dtype=context.dtype)
            elif action_mask.ndim == 3:
                action_masks = action_mask.to(dtype=context.dtype)
            else:
                raise ValueError("action_mask must have shape [B,A] or [B,T,A]")
        return torch.cat(
            (context.flatten(start_dim=2), node_masks, action_masks), dim=-1
        )

    def forward(
        self,
        obss: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        bag_obss: Optional[torch.Tensor] = None,
        bag_actions: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if obss.ndim == 4:
            if (
                action_mask is None
                and actions is not None
                and actions.shape[-1] == self.action_count
            ):
                action_mask, actions = actions, None
            obss = self._flatten_graph_context(obss, action_mask, node_mask)
        q_values = super().forward(obss, actions, bag_obss, bag_actions)
        masks = obss[..., self.action_mask_offset :].bool()
        return q_values.masked_fill(~masks, torch.finfo(q_values.dtype).min)

class HANMaskedDTQN(TemporalMaskedDTQN):
    def __init__(
        self,
        node_count: int,
        feature_dim: int,
        action_count: int,
        instance_slot_counts: Optional[Sequence[int]] = None,
        hidden_dim: int = 128,
        attention_heads: int = 8,
        transformer_layers: int = 2,
        transformer_dropout: float = 0.0,
        graph_attention_heads: int = 2,
        max_context: int = 50,
    ):

        with torch.random.fork_rng(devices=[]):
            graph_encoder = HANEncoder(
                node_count,
                feature_dim,
                hidden_dim,
                graph_attention_heads,
                action_count,
                instance_slot_counts,
            )
        super().__init__(
            graph_encoder=graph_encoder,
            node_count=node_count,
            feature_dim=feature_dim,
            action_count=action_count,
            instance_slot_counts=graph_encoder.layout.slot_counts,
            hidden_dim=hidden_dim,
            attention_heads=attention_heads,
            transformer_layers=transformer_layers,
            transformer_dropout=transformer_dropout,
            max_context=max_context,
        )

    @property
    def han(self) -> HANEncoder:
        return self.graph_encoder

def build_masked_dtqn(encoder_type: str, **network_args) -> TemporalMaskedDTQN:
    if encoder_type == "han":
        return HANMaskedDTQN(**network_args)
    raise ValueError(f"unknown graph encoder: {encoder_type}")
