# Inherited from https://github.com/HazyResearch/KGEmb
"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch

from torch import nn
from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "CP", "MurE", "RotE", "RefE", "AttE"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[..., 0])
        rel_e = self.rel(queries[..., 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


class CP(BaseE):
    """Canonical tensor decomposition https://arxiv.org/pdf/1806.07297.pdf"""

    def __init__(self, args):
        super(CP, self).__init__(args)
        self.sim = "dot"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.entity(queries[..., 0]) * self.rel(queries[..., 1])
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


class MurE(BaseE):
    """Diagonal scaling https://arxiv.org/pdf/1905.09791.pdf"""

    def __init__(self, args):
        super(MurE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.sim = "dist"

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = self.rel_diag(queries[..., 1]) * self.entity(queries[..., 0]) + self.rel(queries[..., 1])
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


class RotE(BaseE):
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(RotE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.sim = "dist"

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[..., 1]), self.entity(queries[..., 0])) + self.rel(queries[..., 1])
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases

class RefE(BaseE):
    """Euclidean 2x2 Givens reflections"""

    def __init__(self, args):
        super(RefE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.sim = "dist"

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs = givens_reflection(self.rel_diag(queries[..., 1]), self.entity(queries[..., 0]))
        rel = self.rel(queries[..., 1])
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs + rel, lhs_biases


class AttE(BaseE):
    """Euclidean attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttE, self).__init__(args)
        self.sim = "dist"

        # reflection
        self.ref = nn.Embedding(self.sizes[1], self.rank)

        # rotation
        self.rot = nn.Embedding(self.sizes[1], self.rank)

        # attention
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=-2)
        self.scale = 1. / np.sqrt(self.rank)

        with torch.no_grad():
            nn.init.uniform_(self.rot.weight, -1.0, 1.0)
            nn.init.uniform_(self.ref.weight, -1.0, 1.0)

    def get_reflection_queries(self, queries):
        lhs_ref_e = givens_reflection(
            self.ref(queries[..., 1]), self.entity(queries[..., 0])
        )
        return lhs_ref_e

    def get_rotation_queries(self, queries):
        lhs_rot_e = givens_rotations(
            self.rot(queries[..., 1]), self.entity(queries[..., 0])
        )
        return lhs_rot_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        lhs_ref_e = self.get_reflection_queries(queries).unsqueeze(-2)
        lhs_rot_e = self.get_rotation_queries(queries).unsqueeze(-2)

        # self-attention mechanism
        cands = torch.cat([lhs_ref_e, lhs_rot_e], dim=-2)
        context_vec = self.context_vec(queries[..., 1]).unsqueeze(-2)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        lhs_e = torch.sum(att_weights * cands, dim=-2) + self.rel(queries[..., 1])
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases
