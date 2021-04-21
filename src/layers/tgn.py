import copy
from typing import Callable, Tuple
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell
from torch_scatter import scatter

from torch_geometric.nn.inits import zeros
from torch_geometric.utils import softmax
from torch_geometric.nn.models.tgn import (
    TGNMemory as _TGN,
    TimeEncoder, 
    IdentityMessage,
    LastAggregator
)

import torchsnooper

_MemoryTuple = namedtuple('Memory', ['src', 'dst', 't', 'raw_msg'])
class Memory(_MemoryTuple):
    template = 'Memory(src={src}, dst={dst}, t={t}, raw_msg={raw_msg})'
    
    def to(self, device, **kwargs):
        return Memory(*(t.to(device, **kwargs) for t in self))
    
    def __repr__(self):
        return self.template.format(**{k: v.size() for k, v in self._asdict().items()})


class ExpireSpan(torch.nn.Module):
    def __init__(self, dim: int, max_time: int = 1000, ramp_length: int = 500):
        super().__init__()
        self.max_time = max_time
        self.to_expiration = Linear(dim, 1)
        self.ramp_length = ramp_length

    def forward(self, msg, t_rel):
        e = self.to_expiration(msg).sigmoid() * self.max_time
        r = e - t_rel.unsqueeze(-1)

        return torch.clamp((r / self.ramp_length) + 1, min = 0., max = 1.)

class TGNMemory(_TGN):
    def __init__(self, *args, expire_span: ExpireSpan = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.expire_span = expire_span

    def __reset_message_store__(self):
        i = self.memory.new_empty((0, ), dtype=torch.long)
        msg = self.memory.new_empty((0, self.raw_msg_dim))
        # Message store format: (src, dst, t, msg)
        self.msg_s_store = {j: Memory(i, i, i, msg) for j in range(self.num_nodes)}
        self.msg_d_store = {j: Memory(i, i, i, msg) for j in range(self.num_nodes)}

    def __update_msg_store__(self, src, dst, t, raw_msg, msg_store):
        n_id, perm = src.sort()
        n_id, count = n_id.unique_consecutive(return_counts=True)
        for i, idx in zip(n_id.tolist(), perm.split(count.tolist())):
            msg_store[i] = Memory(src[idx], dst[idx], t[idx], raw_msg[idx])

    def __compute_msg__(self, n_id, msg_store, msg_module):
        data = [msg_store[i].to(n_id.device) for i in n_id.tolist()]
        src, dst, t, raw_msg = (torch.cat(x, dim=0) for x in zip(*data))
        t_rel = t - self.last_update[src]
        t_enc = self.time_enc(t_rel.type_as(raw_msg))
        
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc) # concat
        if self.expire_span:
            msg = msg * self.expire_span(msg, t_rel)
        return msg, t, src, dst


class AttentionAggregator(torch.nn.Module):
    def __init__(self, memory_dim: int):
        super().__init__()
        self.attn = torch.nn.Sequential(
            Linear(memory_dim, memory_dim // 2),
            torch.nn.Tanh(),
            Linear(memory_dim // 2, 1)
        )

    def forward(self, msg, index, t, dim_size):
        w = softmax(self.attn(msg), index, num_nodes=dim_size)
        msg = w * msg
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce='sum')