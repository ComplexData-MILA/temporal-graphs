import argparse
import os

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm, trange

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator
)
from src.layers.tgn import TGNMemory, AttentionAggregator, ExpireSpan
from src.layers.gnn import GraphAttentionEmbedding, LinkPredictor
from src.layers.tsam import TSAM, TSAMAggregator, TSAMMessage
from src.trainer import Trainer

def launch_wandb(args):
    if not args.use_wandb:
        return

    import wandb
    from datetime import datetime
    
    config = dict(vars(args))
    config.pop('wandb_group')
    config.pop('use_wandb')
    config.pop('rank')

    name = f'%s_%s' % (os.environ['USER'], datetime.now().strftime('%F_%T'))
    if args.wandb_group:
        name = name + f'_{args.rank:01}'

    return wandb.init(
        name=name,
        project='temporal-graphs',
        entity=os.environ.get('WANDB_USER', 'cdlab-mila'),
        config=config,
        group=args.wandb_group,
        dir=os.environ.get('SLURM_TMPDIR', '.')
    )

def main(
    memory_type: str,
    path: str,
    dataset: str,
    embedding_dim: int,
    learning_rate: float = 0.0001,
    epochs: int = 25,
    run: 'wandb.Run' = None,
    regularize: bool = False,
    **kwargs
):
    dataset = JODIEDataset(path, name=dataset)
    data = dataset[0]

    memory_dim = time_dim = embedding_dim
    raw_msg_dim = data.msg.size(-1)

    if memory_type == 'tgn':
        msg_module = IdentityMessage(data.msg.size(-1), memory_dim, time_dim)
        expire_span = (args.expire_span > 0) and ExpireSpan(dim=msg_module.out_channels, max_time=args.expire_span)
        memory = TGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=msg_module,
            aggregator_module=AttentionAggregator(msg_module.out_channels), # LastAggregator(),
            expire_span=expire_span
        )
    elif memory_type == 'tsam':
        memory = TSAM(
            data.num_nodes, 
            raw_msg_dim, 
            memory_dim, 
            time_dim,
            message_module=TSAMMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=AttentionAggregator(memory_dim) #TSAMAggregator(),
        )
    else:
        raise ValueError(f'Invalid memory_type {memory_type}.')

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    )

    link_pred = LinkPredictor(in_channels=embedding_dim)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | 
        set(gnn.parameters()) | 
        set(link_pred.parameters()), 
    lr=learning_rate)
    
    trainer = Trainer(
        dataset,
        memory,
        gnn,
        link_pred,
        optimizer,
        run=run,
        epochs=epochs,
        regularize=regularize
    )
    trainer.trial()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_type', default='tgn', choices=['tgn', 'tsam'])
    parser.add_argument('-e', '--expire_span', type=int, default=-1)
    parser.add_argument('--path', default='./data/JODIE')
    parser.add_argument('--dataset', default='mooc', choices=['mooc', 'wikipedia', 'lastfm', 'reddit'])
    parser.add_argument('-r', '--regularize', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_group', type=str, required=False)
    parser.add_argument('--rank', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', 0)))

    args = parser.parse_args()
    run = launch_wandb(args)

    main(**vars(args), run=run)