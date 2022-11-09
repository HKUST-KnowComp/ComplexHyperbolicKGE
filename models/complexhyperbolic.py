# Copyright (c) 2022-present, Huiru Xiao, HKUST-KnowComp.
# All rights reserved.
"""Complex Hyperbolic Knowledge Graph embedding models."""
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from torch import nn
from abc import ABC, abstractmethod
from models.base import KGModel
from utils.complexhyperbolic import chyp_distance, expmap0, logmap0, project, mobius_add, Distance, real_mobius_add
from utils.euclidean import givens_rotations, givens_reflection

CHYP_MODELS = ["FFTRotH", "FFTRefH", "FFTAttH"]


class FFTUnitBall(KGModel):
    """
    rel_diag: nn.Embedding, size = (n_relations, dim/2)
    multi_c: bool
    c_init: tensor, size = #relations
    """
    def __init__(self, args):
        super(FFTUnitBall, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                       args.init_size)
        self.dim = 2 * (self.rank - 1)
        del self.entity
        self.entity = nn.Embedding(self.sizes[0], 2 * self.rank)
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.dim)
        self.rel_diag = nn.Embedding(self.sizes[1], self.dim)
        self.multi_c = args.multi_c
        if self.multi_c:
            self.c = nn.Embedding(self.sizes[1], 1)
        else:
            self.c = nn.Embedding(1, 1)

        with torch.no_grad():
            nn.init.normal_(self.entity.weight, 0.0, self.init_size)
            nn.init.normal_(self.rel.weight, 0.0, self.init_size)
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.ones_(self.c.weight)

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space.

        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        lhs_e, c = lhs_e
        # lhs_e = lhs_e[..., :self.rank] + 1j * lhs_e[..., self.rank:]
        # rhs_e = rhs_e[..., :self.rank] + 1j * rhs_e[..., self.rank:]
        # return - chyp_distance(lhs_e, rhs_e, c, eval_mode) ** 2
        # re_lhs = lhs_e[..., :self.rank]
        # im_lhs = lhs_e[..., self.rank:]
        # re_rhs = rhs_e[..., :self.rank]
        # im_rhs = rhs_e[..., self.rank:]
        return - Distance.apply(lhs_e, rhs_e) ** 2


class FFTRotH(FFTUnitBall):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = self.entity(queries[..., 0])
        head = head[..., :self.rank] + 1j * head[..., self.rank:]
        head = torch.fft.irfft(head, norm="ortho")
        head = expmap0(head, c)   # hyperbolic
        rel1, rel2 = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel1 = expmap0(rel1, c)   # hyperbolic
        rel2 = expmap0(rel2, c)   # hyperbolic
        lhs = project(real_mobius_add(head, rel1, c), c)   # hyperbolic
        res1 = givens_rotations(self.rel_diag(queries[..., 1]), lhs)   # givens_rotation(Euclidean, hyperbolic)
        res2 = real_mobius_add(res1, rel2, c)   # hyperbolic
        res2 = torch.fft.rfft(res2, norm="ortho")
        res = torch.cat((res2.real, res2.imag), -1)
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases


class FFTRefH(FFTUnitBall):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        rel, _ = torch.chunk(self.rel(queries[..., 1]), 2, dim=-1)   # Euclidean
        rel = expmap0(rel, c)   # hyperbolic
        head = self.entity(queries[..., 0])
        head = head[..., :self.rank] + 1j * head[..., self.rank:]
        head = torch.fft.irfft(head, norm="ortho")
        lhs = givens_reflection(self.rel_diag(queries[..., 1]), head)   # givens_reflection(Euclidean, Euclidean)
        lhs = expmap0(lhs, c)   # hyperbolic
        res = project(real_mobius_add(lhs, rel, c), c)   # hyperbolic
        res = torch.fft.rfft(res, norm="ortho")
        res = torch.cat((res.real, res.imag), -1)
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases


class FFTAttH(FFTUnitBall):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(FFTAttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.dim)
        self.context_vec = nn.Embedding(self.sizes[1], self.dim)
        self.act = nn.Softmax(dim=-2)
        self.scale = 1. / np.sqrt(self.rank)
        
        with torch.no_grad():
            nn.init.uniform_(self.rel_diag.weight, -1.0, 1.0)
            nn.init.normal_(self.context_vec.weight, 0.0, self.init_size)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c(queries[..., 1]))
        head = self.entity(queries[..., 0])
        head = head[..., :self.rank] + 1j * head[..., self.rank:]
        head = torch.fft.irfft(head, norm="ortho")
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
        res = project(real_mobius_add(lhs, rel, c), c)
        res = torch.fft.rfft(res, norm="ortho")
        res = torch.cat((res.real, res.imag), -1)
        lhs_biases = self.bh(queries[..., 0])
        while res.dim() < 3:
            res = res.unsqueeze(1)
        while c.dim() < 3:
            c = c.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return (res, c), lhs_biases

