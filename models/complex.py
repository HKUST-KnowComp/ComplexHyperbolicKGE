# Inherited from https://github.com/HazyResearch/KGEmb
"""Euclidean Knowledge Graph embedding models where embeddings are in complex space."""
import torch
from torch import nn

from models.base import KGModel

COMPLEX_MODELS = ["ComplEx", "RotatE", "Fourier"]


class BaseC(KGModel):
    """Complex Knowledge Graph Embedding models.

    Attributes:
        embeddings: complex embeddings for entities and relations
    """

    def __init__(self, args):
        """Initialize a Complex KGModel.
        rank: dim
        entity: nn.Embedding, size = (n_entities, 2 * dim)
        rel: nn.Embedding, size = (n_relations, 2 * dim)
        embeddings: a ModuleList with two nn.Embedding, [0]: entity, [1]: rel
        """
        super(BaseC, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        assert self.rank % 2 == 0, "Complex models require even embedding dimension"
        self.rank = self.rank // 2

    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e = lhs_e[..., :self.rank], lhs_e[..., self.rank:]
        rhs_e = rhs_e[..., :self.rank], rhs_e[..., self.rank:]
        return torch.sum(
            lhs_e[0] * rhs_e[0] + lhs_e[1] * rhs_e[1],
            -1, keepdim=True
        )

    def get_complex_embeddings(self, queries, tails=None):
        """Get complex embeddings of queries."""
        head_e = self.entity(queries[..., 0])
        rel_e = self.rel(queries[..., 1])
        if tails is None:
            rhs_e = self.entity.weight
        else:
            rhs_e = self.entity(tails)
        head_e = (head_e[..., :self.rank], head_e[..., self.rank:])
        rel_e = (rel_e[..., :self.rank], rel_e[..., self.rank:])
        rhs_e = (rhs_e[..., :self.rank], rhs_e[..., self.rank:])
        return head_e, rel_e, rhs_e

    def get_factors(self, queries, tails=None):
        """Compute factors for embeddings' regularization."""
        head_e, rel_e, rhs_e = self.get_complex_embeddings(queries)
        head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2)
        rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2)
        return head_f, rel_f, rhs_f


class ComplEx(BaseC):
    """Simple complex model http://proceedings.mlr.press/v48/trouillon16.pdf"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        lhs_e = torch.cat([
            head_e[0] * rel_e[0] - head_e[1] * rel_e[1],
            head_e[0] * rel_e[1] + head_e[1] * rel_e[0]
        ], -1)
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


class RotatE(BaseC):
    """Rotations in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        rel_norm = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2)
        cos = rel_e[0] / rel_norm
        sin = rel_e[1] / rel_norm
        lhs_e = torch.cat([
            head_e[0] * cos - head_e[1] * sin,
            head_e[0] * sin + head_e[1] * cos
        ], -1)
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


class Fourier(BaseC):
    """Fourier transformes in complex space https://openreview.net/pdf?id=HkgEQnRqYQ"""
    def __init__(self, args):
        super(Fourier, self).__init__(args)
        self.dim = 2 * (self.rank - 1)
        # self.dim = int(1.5 * self.rank) // 2 * 2
        del self.rel
        self.rel = nn.Embedding(self.sizes[1], 2 * self.dim)
        with torch.no_grad():
            nn.init.normal_(self.rel.weight, 0.0, self.init_size)
            self.rel.weight[..., :self.dim] += 3*self.init_size
            self.rel.weight[..., self.dim:] -= 3*self.init_size

    def get_complex_embeddings(self, queries, tails=None):
        """Get complex embeddings of queries."""
        head_e = self.entity(queries[..., 0])
        rel_e = self.rel(queries[..., 1])
        if tails is None:
            rhs_e = self.entity.weight
        else:
            rhs_e = self.entity(tails)
        head_e = (head_e[..., :self.rank], head_e[..., self.rank:])
        rel_e = (rel_e[..., :self.dim], rel_e[..., self.dim:])
        rhs_e = (rhs_e[..., :self.rank], rhs_e[..., self.rank:])
        return head_e, rel_e, rhs_e

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        head_e, rel_e, _ = self.get_complex_embeddings(queries)
        # IFFT
        head = head_e[0] + 1j * head_e[1]
        head = torch.fft.irfft(head, norm="ortho", n=self.dim)
        # high-pass & low-pass
        hpf = rel_e[0]
        lpf = rel_e[1]
        res = 0.5 * (torch.min(head, hpf) + torch.max(head, lpf))
        # FFT
        res = torch.fft.rfft(res, norm="ortho", n=2*self.rank-1)
        lhs_e = torch.cat((res.real, res.imag), -1)
        lhs_biases = self.bh(queries[..., 0])
        while lhs_e.dim() < 3:
            lhs_e = lhs_e.unsqueeze(1)
        while lhs_biases.dim() < 3:
            lhs_biases = lhs_biases.unsqueeze(1)
        return lhs_e, lhs_biases


