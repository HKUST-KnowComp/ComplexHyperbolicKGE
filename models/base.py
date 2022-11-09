# Inherited from https://github.com/HazyResearch/KGEmb
"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod
from numpy import isin

import torch
from torch import nn
from torch import overrides
import gc


class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
            # self.bias_type = torch.double
        elif data_type == 'float':
            self.data_type = torch.float
            # self.bias_type = torch.float
        # elif data_type == 'cfloat':
        #     self.data_type = torch.cfloat
        #     self.bias_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = gamma
        self.entity = nn.Embedding(sizes[0], self.rank)
        self.rel = nn.Embedding(sizes[1], self.rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bt = nn.Embedding(sizes[0], 1)

        with torch.no_grad():
            nn.init.normal_(self.entity.weight, 0.0, self.init_size)
            nn.init.normal_(self.rel.weight, 0.0, self.init_size)
            nn.init.zeros_(self.bh.weight)
            nn.init.zeros_(self.bt.weight)

    def register_buffer(self, name, tensor, persistent=True):
        with torch.no_grad():
            tensor = tensor.to(self.data_type)
        super().register_buffer(name, tensor, persistent)
    
    def register_parameter(self, name, param):
        with torch.no_grad():
            param = param.to(self.data_type)
        super().register_parameter(name, param)
    
    def add_module(self, name, module):
        with torch.no_grad():
            if isinstance(module, nn.Embedding):
                module.weight.data = module.weight.data.to(self.data_type)
        super().add_module(name, module)

    def __setattr__(self, name, value):
        with torch.no_grad():
            if isinstance(value, nn.Embedding):
                value.weight.data = value.weight.data.to(self.data_type)
            elif isinstance(value, nn.Parameter):
                value = value.to(self.data_type)
            elif isinstance(value, torch.Tensor):
                value = value.to(self.data_type)
            elif isinstance(value, nn.Container):
                value = value.to(self.data_type)
        super().__setattr__(name, value)

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    def get_rhs(self, tails=None):
        """Get embeddings and biases of target entities.

        Args:
            tails: torch.LongTensor with tails
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if tails are given, then returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if tails are given, then returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        if tails is None:
            rhs_e, rhs_biases = self.entity.weight, self.bt.weight
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(0)
            while rhs_biases.dim() < 3:
                rhs_biases = rhs_biases.unsqueeze(0)
        else:
            rhs_e, rhs_biases = self.entity(tails), self.bt(tails)
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(1)
            while rhs_biases.dim() < 3:
                rhs_biases = rhs_biases.unsqueeze(1)
        return rhs_e, rhs_biases

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e)
        if self.bias == 'constant':
            return self.gamma + score
        elif self.bias == 'learn':
            return lhs_biases + rhs_biases + score
        else:
            return score

    def get_factors(self, queries, tails=None):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation)
            queries: torch.LongTensor with query triples (head, relation)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[..., 0])
        rel_e = self.rel(queries[..., 1])
        while head_e.dim() < 3:
            head_e = head_e.unsqueeze(1)
        while rel_e.dim() < 3:
            rel_e = rel_e.unsqueeze(1)
        if tails is None:
            rhs_e = self.entity.weight
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(0)
        else:
            rhs_e = self.entity(tails)
            while rhs_e.dim() < 3:
                rhs_e = rhs_e.unsqueeze(1)
        return head_e, rel_e, rhs_e

    def forward(self, queries, tails=None):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation)
            tails: torch.LongTensor with tails
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        while queries.dim() < 3:
            queries = queries.unsqueeze(1)
        if tails is not None:
            while tails.dim() < 2:
                tails = tails.unsqueeze(0)
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(tails)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases))

        # get factors for regularization
        factors = self.get_factors(queries, tails)
        return predictions, factors

    def get_ranking(self, queries, filters, batch_size=500):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries), 1)
        device = self.entity.weight.device
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(None)
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                # mask = torch.zeros((these_queries.size(0), self.entity.weight.size(0)), dtype=torch.bool)
                # for i, query in enumerate(these_queries.numpy()):
                #     mask[i, query[2]] = 1
                #     mask[i, filters[tuple(query[:2])]] = 1
                # mask = mask.to(device)
                these_queries = these_queries.to(device)

                q = self.get_queries(these_queries[..., :2])
                rhs = self.get_rhs(these_queries[..., 2])
                scores = self.score(q, candidates)
                targets = self.score(q, rhs)
                # scores.masked_fill_(mask.unsqueeze(-1), -1e6)
                
                # set filtered and true scores to -1e6 to be ignored
                # for i, query in enumerate(these_queries.cpu().numpy()):
                these_queries = these_queries.cpu().numpy()
                for i, query in enumerate(these_queries):
                    filter_out = filters[tuple(query[:2])]
                    filter_out += [query[2].item()]
                    scores[i, filter_out] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
                del these_queries
                del q
                del rhs
                del scores
                del targets
        del candidates
        gc.collect()
        return ranks.squeeze(1)

    def compute_metrics(self, examples, filters, batch_size=10):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        # rhs
        q = examples
        ranks = self.get_ranking(q, filters["rhs"], batch_size=batch_size)
        mean_rank["rhs"] = torch.mean(ranks).item()
        mean_reciprocal_rank["rhs"] = torch.mean(1. / ranks).item()
        hits_at["rhs"] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))

        # lhs
        q = torch.stack([examples[..., 2], examples[..., 1] + self.sizes[1] // 2, examples[..., 0]], dim=-1)
        ranks = self.get_ranking(q, filters["lhs"], batch_size=batch_size)
        mean_rank["lhs"] = torch.mean(ranks).item()
        mean_reciprocal_rank["lhs"] = torch.mean(1. / ranks).item()
        hits_at["lhs"] = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))
        
        return mean_rank, mean_reciprocal_rank, hits_at

