import copy
from typing import Callable, Tuple
from collections import namedtuple

import torch
from torch import Tensor
from torch.nn import Linear, GRUCell
from torch_scatter import scatter

from torch_geometric.nn.inits import zeros
from torch_geometric.nn.models.tgn import (
    TGNMemory as _TGN,
    TimeEncoder, 
    IdentityMessage,
    LastAggregator
)


_MemoryTuple = namedtuple('Memory', ['src', 'dst', 't', 'raw_msg'])
class Memory(_MemoryTuple):
    template = 'Memory(src={src}, dst={dst}, t={t}, raw_msg={raw_msg})'
    
    def to(self, device, **kwargs):
        return Memory(*(t.to(device, **kwargs) for t in self))
    
    def __repr__(self):
        return self.template.format(**{k: v.size() for k, v in self._asdict().items()})

class TGNMemory(_TGN):        
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
        
        msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
        return msg, t, src, dst