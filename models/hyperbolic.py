# Inherited from https://github.com/HazyResearch/KGEmb
"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

HYP_MODELS = ["RotH", "RefH", "AttH"]


class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        """
        rank: dim
        entity: nn.Embedding, size = (n_entities, dim)
        rel: nn.Embedding, size = (n_relations, dim)
        rel_daig: nn.Embedding, size = (n_relations, dim), what is this???
        multi_c: bool
        c_init: tensor, size = dim
        """
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.multi_c = args.multi_c
        if self.multi_c:
            self.c = nn.Embedding(self.sizes[1], 1)
        else:
            self.c = nn.Embedding(1, 1)

        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.c.weight)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = expmap0(self.entity(queries[..., 0]), c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(mobius_add(head, rel1, c), c)   # hyperbolic
        res1 = givens_rotations(self.rel_diag(queries[..., 1]), lhs)   # givens_rotation(Euclidean, hyperbolic)
        res2 = mobius_add(res1, rel2, c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res2.dim() < 3:
            res2 = res2.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res2, c), lhs_biases


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        rel, _ = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel = expmap0(rel, c)   # hyperbolic
        lhs = givens_reflection(self.rel_diag(queries[..., 1]), self.entity(queries[..., 0]))   # givens_reflection(Euclidean, Euclidean)
        lhs = expmap0(lhs, c)   # hyperbolic
        res = project(mobius_add(lhs, rel, c), c)   # hyperbolic
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.act = nn.Softmax(dim=-2)
        self.scale = 1. / np.sqrt(self.rank)

        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.normal_(self.context_vec.weight, 0, self.init_size)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = self.entity(queries[..., 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[..., 1]), 2, dim=-1)
        rot_q = givens_rotations(rot_mat, head).unsqueeze(-2)
        ref_q = givens_reflection(ref_mat, head).unsqueeze(-2)
        cands = torch.cat([ref_q, rot_q], dim=-2)
        context_vec = self.context_vec(queries[..., 1]).unsqueeze(-2)
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=-2)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases
