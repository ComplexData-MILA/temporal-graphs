import json
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm, trange
from dataclasses import dataclass

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn.models.tgn import LastNeighborLoader

# https://burhan-mudassar.netlify.app/post/power-of-hooks-in-pytorch/
class ExpireSpanHook:
    def __init__(self):
        self.handle = None
        self.reset()

    def register_hook(self, module):
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.expired += (output > 0).sum().item()
        self.num_events += output.size(0)
        return output

    def reset(self):
        self.expired, self.num_events = 0., 0

    def compute(self):
        if self.num_events == 0:
            return 1
        return self.expired / self.num_events

    def close(self):
        if self.handle:
            self.handle.remove()

@dataclass
class Trainer:
    dataset: JODIEDataset
    memory: torch.nn.Module
    gnn: torch.nn.Module
    link_pred: torch.nn.Module
    optimizer: torch.optim.Optimizer

    criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    regularize: bool = False
    
    neighbor_size: int = 10
    batch_size: int = 200
    epochs: int = 50
    seed: int = 12345

    run: 'wandb.Run' = None

    def __post_init__(self):
        data = self.dataset[0].to(self.device)

        # Ensure to only sample actual destination nodes as negatives.
        self.min_dst_idx, self.max_dst_idx = int(data.dst.min()), int(data.dst.max())
        self.train_data, self.val_data, self.test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

        self.neighbor_loader = LastNeighborLoader(data.num_nodes, size=self.neighbor_size, device=self.device)
        self.assoc = torch.empty(data.num_nodes, dtype=torch.long, device=self.device) # Helper vector to map global node indices to local ones.
        self.data = data

        self.memory.to(self.device)
        self.gnn.to(self.device)
        self.link_pred.to(self.device)

        self._hook = ExpireSpanHook()
        if self.memory.expire_span:
            self._hook.register_hook(self.memory.expire_span)

    def _forward_on_batch(self, batch):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(self.min_dst_idx, self.max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=self.device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        # Get updated memory of all nodes involved in the computation.
        prev_mem = self.memory.memory[n_id]
        cur_mem, last_update = self.memory(n_id)
        z = self.gnn(cur_mem, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id])

        pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])
        neg_out = self.link_pred(z[self.assoc[src]], z[self.assoc[neg_dst]])

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(src, pos_dst, t, msg)
        self.neighbor_loader.insert(src, pos_dst)

        if self.memory.training:
            reg = 0.
            if self.regularize:
                reg = (1 - torch.nn.functional.cosine_similarity(prev_mem, cur_mem)).mean()
            return pos_out, neg_out, reg
        return pos_out, neg_out

    def train(self):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory.
        self.neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        total_events = 0
        with tqdm(self.train_data.seq_batches(batch_size=self.batch_size), leave=False, total=self.train_data.num_events // self.batch_size) as pbar:
            for i, batch in enumerate(pbar, 1):
                self.optimizer.zero_grad()

                pos_out, neg_out, reg = self._forward_on_batch(batch)

                loss = self.criterion(pos_out, torch.ones_like(pos_out))
                loss += self.criterion(neg_out, torch.zeros_like(neg_out))
                loss += reg

                loss.backward()
                self.optimizer.step()
                self.memory.detach()

                total_loss += loss.item() * batch.num_events
                total_events += batch.num_events
                
                span = self._hook.compute()

                pbar.set_description(f'loss: {total_loss / total_events:0.4f}, span: {span:0.4f}') # / total_events
                if self.run is not None:
                    metrics = {'loss': loss.item(), 'span': span}
                    if self.regularize:
                        metrics['reg'] = reg
                    self.run.log(metrics)

        self._hook.reset()
        return total_loss / self.train_data.num_events

    @torch.no_grad()
    def test(self, inference_data):
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        torch.manual_seed(self.seed)  # Ensure deterministic sampling across epochs.

        aps, aucs = [], []
        for batch in tqdm(inference_data.seq_batches(batch_size=self.batch_size), leave=False, total=inference_data.num_events // self.batch_size):
            pos_out, neg_out = self._forward_on_batch(batch)

            y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
            y_true = torch.cat(
                [torch.ones(pos_out.size(0)),
                torch.zeros(neg_out.size(0))], dim=0)

            aps.append(average_precision_score(y_true, y_pred))
            aucs.append(roc_auc_score(y_true, y_pred))

        return torch.tensor(aps).mean().item(), torch.tensor(aucs).mean().item()

    def trial(self):
        with trange(self.epochs) as pbar:
            for epoch in pbar:
                loss = self.train()
                val_ap, val_auc = self.test(self.val_data)

                metrics = {
                    'epoch': epoch,
                    'train_loss': loss,
                    'val_ap': val_ap,
                    'val_auc': val_auc
                }
                if epoch == (self.epochs - 1):
                    test_ap, test_auc = self.test(self.test_data)
                    metrics['test_ap'] = test_ap
                    metrics['test_auc'] = test_auc
                
                if self.run is not None:
                    self.run.log(metrics)
                else:
                    print(json.dumps(metrics, indent=2))