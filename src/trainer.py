import json
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm, trange
from dataclasses import dataclass

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn.models.tgn import LastNeighborLoader

@dataclass
class Trainer:
    dataset: JODIEDataset
    memory: torch.nn.Module
    gnn: torch.nn.Module
    link_pred: torch.nn.Module
    optimizer: torch.optim.Optimizer

    criterion: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    neighbor_size: int = 10
    batch_size: int = 100
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

    def _forward_on_batch(self, batch):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(self.min_dst_idx, self.max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=self.device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)
        z = self.gnn(z, last_update, edge_index, self.data.t[e_id], self.data.msg[e_id])

        pos_out = self.link_pred(z[self.assoc[src]], z[self.assoc[pos_dst]])
        neg_out = self.link_pred(z[self.assoc[src]], z[self.assoc[neg_dst]])

        # Update memory and neighbor loader with ground-truth state.
        self.memory.update_state(src, pos_dst, t, msg)
        self.neighbor_loader.insert(src, pos_dst)

        return pos_out, neg_out

    def train(self):
        self.memory.train()
        self.gnn.train()
        self.link_pred.train()

        self.memory.reset_state()  # Start with a fresh memory.
        self.neighbor_loader.reset_state()  # Start with an empty graph.

        total_loss = 0
        with tqdm(self.train_data.seq_batches(batch_size=self.batch_size), leave=False) as pbar:
            for batch in pbar:
                self.optimizer.zero_grad()

                pos_out, neg_out = self._forward_on_batch(batch)

                loss = self.criterion(pos_out, torch.ones_like(pos_out))
                loss += self.criterion(neg_out, torch.zeros_like(neg_out))

                loss.backward()
                self.optimizer.step()
                self.memory.detach()

                total_loss += loss.item() * batch.num_events
                pbar.set_description(f'loss: {loss.item():0.4f}')
                if self.run is not None:
                    self.run.log({'loss': loss.item()})

        return total_loss / self.train_data.num_events

    @torch.no_grad()
    def test(self, inference_data):
        self.memory.eval()
        self.gnn.eval()
        self.link_pred.eval()

        torch.manual_seed(self.seed)  # Ensure deterministic sampling across epochs.

        aps, aucs = [], []
        for batch in tqdm(inference_data.seq_batches(batch_size=self.batch_size), leave=False):
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