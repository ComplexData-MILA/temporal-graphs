import copy
from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell
from torch_scatter import scatter

from torch_geometric.nn.inits import zeros
from torch_geometric.nn.models.tgn import (
    TimeEncoder, 
    IdentityMessage
)

from src.layers.tgn import TGNMemory

import torchsnooper

def dot_product_attention(matrix, vector):
    w = matrix.bmm(vector.unsqueeze(-1)).squeeze(-1).softmax(-1)
    return w.unsqueeze(1).bmm(matrix).squeeze(1)

class TSAM(TGNMemory):
    def __get_updated_memory__(self, n_id):
        self.__assoc__[n_id] = torch.arange(n_id.size(0), device=n_id.device)

        # Compute messages (src -> dst).
        msg_s, t_s, src_s, dst_s = self.__compute_msg__(n_id, self.msg_s_store, self.msg_s_module)

        # Compute messages (dst -> src).
        msg_d, t_d, src_d, dst_d = self.__compute_msg__(n_id, self.msg_d_store, self.msg_d_module)

        # Aggregate messages.
        idx = torch.cat([src_s, src_d], dim=0)
        msg = torch.cat([msg_s, msg_d], dim=0)
        t = torch.cat([t_s, t_d], dim=0)
        aggr = self.aggr_module(msg, self.__assoc__[idx], t, n_id.size(0))

        # Get local copy of updated memory.
        retrieved = dot_product_attention(aggr, self.memory[n_id])
        memory = self.gru(retrieved, self.memory[n_id])

        # Get local copy of updated `last_update`.
        last_update = self.last_update.scatter(0, idx, t)[n_id]

        return memory, last_update


class TSAMMessage(IdentityMessage):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__(raw_msg_dim, memory_dim, time_dim)
        self.to_qk = torch.nn.Linear(self.out_channels, memory_dim*2)
        self.out_channels = memory_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        msg = super().forward(z_src, z_dst, raw_msg, t_enc)
        q, k = self.to_qk(msg).chunk(2, dim=-1)
        return torch.einsum('ni,nj->nij', q, k)

class TSAMAggregator(torch.nn.Module):
    def forward(self, msg, index, t, dim_size):
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')