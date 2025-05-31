import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_sum
from torch_geometric.nn import global_add_pool
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from abc import ABC, abstractmethod

# per-molecule bidirectional GRU booster
class BatchGRUBooster(nn.Module):
    """Per-molecule bi-GRU booster for PyG batches."""
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.bias = nn.Parameter(torch.empty(hidden).uniform_(
            -1.0 / hidden**0.5, 1.0 / hidden**0.5))

    def forward(self, h, batch):
        # handle empty graph case
        if h.size(0) == 0:
            return h
        h = F.relu(h + self.bias)
        num_graphs = int(batch.max()) + 1
        lengths = torch.bincount(batch.cpu(), minlength=num_graphs).to(batch.device)
        max_len = int(lengths.max().item())
        # build padded tensor [num_graphs, max_len, H]
        padded = h.new_zeros((num_graphs, max_len, h.size(-1)))
        # compute true prefix-sum offsets
        prefix = torch.cumsum(lengths, dim=0) - lengths
        positions = (torch.arange(h.size(0), device=h.device) - prefix[batch]).long()
        padded[batch, positions] = h
        # pack & GRU
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_p, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out_p, batch_first=True, total_length=max_len)
        # clamp GRU activations to avoid overflow/NaN
        out = out.clamp(-10.0, 10.0)
        # flatten and strip padding
        flat = out.reshape(num_graphs * max_len, -1)
        return flat[: h.size(0)]

class CMPNNEncoder(nn.Module):
    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=128, num_steps=5, dropout=0., n_tasks=1, readout='gru', use_booster=True, use_graph_residual:bool = False ):
        super().__init__()
        # store for external queries/tests
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        # control usage of sequence booster: only when order-aware readout
        self.use_booster = use_booster and readout == 'gru'
        # input projections with LayerNorm
        self.lin_node = nn.Sequential(
            nn.Linear(in_node_feats, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.lin_edge = nn.Sequential(
            nn.Linear(in_edge_feats, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # dropout for message passing and booster, from constructor
        self.dropout_mp = nn.Dropout(dropout)
        # FFN head dropout
        self.dropout_head = nn.Dropout(0.10)
        # raw features projection with LayerNorm
        self.lin_x = nn.Sequential(
            nn.Linear(in_node_feats, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # message passing layers with per-step parameters
        self.num_steps = num_steps
        # message update transforms
        self.node_updates = nn.ModuleList([
            nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_steps)
        ])

        self.node_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_steps)
        ])

        self.edge_updates = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_steps)
        ])
        # optional global context booster (order-aware)
        if self.use_booster:
            self.batch_gru = BatchGRUBooster(hidden_dim)
            # final booster projection with LayerNorm
            self.booster_proj = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        # final merge projection with LayerNorm
        self.final_update = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        # graph prediction head, input dim depends on readout method
        if readout == 'gru':
            # permutation-variant readout: bi-GRU then linear
            self.readout_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.graph_pred = nn.Linear(2 * hidden_dim, n_tasks)
        elif readout == 'sum':
            # permutation-invariant readout: sum pooling
            self.graph_pred = nn.Linear(hidden_dim, n_tasks)
        else:
            raise ValueError(f'Unknown readout: {readout}')
        emb_dim = 2 * hidden_dim if readout=='gru' else hidden_dim
        self.graph_norm = nn.BatchNorm1d(emb_dim)
        # save choice for forward/embed
        self.readout = readout

        self.graph_residual = use_graph_residual
        if self.graph_residual:
            self.graph_skip = nn.Linear(emb_dim, hidden_dim)


    def forward(self, x, edge_index, edge_attr, batch):
        # handle empty graph case: return empty predictions
        if x.size(0) == 0:
            return torch.empty((0, self.n_tasks), device=x.device)
        # modular forward
        # 1) init embeddings
        h_v0, h_v, h_e0, h_e = self._init_embeddings(x, edge_attr)
        # 2) directed message passing (use precomputed inv_idx if available)
        inv_idx = getattr(batch, 'inv_idx', None)
        if inv_idx is None:
            inv_idx = self._build_inverse_edges(edge_index, x.size(0))
        h_v, h_e = self._message_passing(h_v, h_e, h_e0, edge_index, inv_idx, batch)
        # 3) final node-edge merging
        h_v = self._final_merge(h_v, h_e, x, edge_index)
        # 4) readout and prediction based on chosen method
        if self.readout == 'gru':
            z =  self._readout(h_v, batch)
            z = self.graph_norm(z)
            z = self.graph_pred(z)
        else:
            # sum pooling (permutation invariant)
            z = global_add_pool(h_v, batch)
            z = self.graph_norm(z)
            z =  self.graph_pred(z)
        if self.graph_residual:
            out = out + self.graph_skip(z)

        else:   
            out = z

        return out

    def embed(self, x, edge_index, edge_attr, batch):
        """Compute graph embeddings without applying the final prediction head."""
        # handle empty graph case: return empty embeddings
        if x.size(0) == 0:
            dim = 2 * self.hidden_dim if self.readout=='gru' else self.hidden_dim
            return torch.empty((0, dim), device=x.device)
        # 1) initial embeddings
        h_v0, h_v, h_e0, h_e = self._init_embeddings(x, edge_attr)
        # 2) message passing
        inv_idx = getattr(batch, 'inv_idx', None)
        if inv_idx is None:
            inv_idx = self._build_inverse_edges(edge_index, x.size(0))
        h_v, h_e = self._message_passing(h_v, h_e, h_e0, edge_index, inv_idx, batch)
        # 3) final merge
        h_v = self._final_merge(h_v, h_e, x, edge_index)
        # 4) readout embedding
        if self.readout == 'gru':
            batch_size = int(batch.max().item()) + 1
            H = h_v.size(1)
            lengths = torch.bincount(batch.cpu(), minlength=batch_size).to(batch.device)
            max_len = int(lengths.max().item())
            prefix = torch.cumsum(lengths, dim=0) - lengths
            padded = h_v.new_zeros((batch_size, max_len, H))
            idx = torch.arange(h_v.size(0), device=h_v.device)
            padded[batch, idx - prefix[batch]] = h_v
            packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_p, _ = self.readout_gru(packed)
            out, _ = pad_packed_sequence(out_p, batch_first=True, total_length=max_len)
            out = out.clamp(-10., 10.) # clamp GRU activations
            mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
            out = out * mask.unsqueeze(-1)
            z = out.sum(dim=1) / lengths.unsqueeze(1).clamp(min=1)
            return z
        else:
            # sum pooling (permutation invariant)
            return global_add_pool(h_v, batch)

    @staticmethod
    def _build_inverse_edges(edge_index, num_nodes):
        # build mapping from each directed edge to its inverse
        src, dst = edge_index  # [E]
        flat = (src * num_nodes + dst).tolist()
        rev_map = {v: i for i, v in enumerate(flat)}
        inv_flat = (dst * num_nodes + src).tolist()
        inv_idx = [rev_map[v] for v in inv_flat]
        return torch.tensor(inv_idx, dtype=torch.long, device=src.device)

    def _init_embeddings(self, x, edge_attr):
        """Project raw node/edge features, apply BN+ReLU."""
        h_v0 = F.relu(self.lin_node(x))
        h_v = h_v0
        h_e0 = F.relu(self.lin_edge(edge_attr))
        h_e = h_e0
        return h_v0, h_v, h_e0, h_e

    def _message_passing(self, h_v, h_e, h_e0, edge_index, inv_idx, batch):
        """Perform K rounds of communicative message passing."""
        # handle empty graph case: no nodes
        if h_v.size(0) == 0:
            return h_v, h_e
        N = h_v.size(0)
        for step in range(self.num_steps):
            # node boost & update for this step
            m_sum = scatter_sum(h_e, edge_index[1], dim=0, dim_size=N)
            m_max, _ = scatter_max(h_e, edge_index[1], dim=0, dim_size=N)
            m_max = F.relu(m_max)  # clamp to avoid negative inf on isolated nodes
            # node update and residual
            h_v_new = F.relu(self.node_updates[step](torch.cat([h_v, m_sum * m_max], dim=-1)))
            h_v = h_v + h_v_new
            h_v = self.node_norms[step](h_v)
            # edge update with dropout
            src, dst = edge_index
            m_e = h_v[dst] - h_e[inv_idx]
            h_e = F.relu(h_e0 + self.edge_updates[step](m_e))
            h_e = self.dropout_mp(h_e)
        # apply global context booster once after message passing (if enabled)
        if self.use_booster:
            h_v = F.relu(self.booster_proj(self.batch_gru(h_v, batch)))
            # saturating tanh after booster
            h_v = torch.tanh(h_v)
            h_v = self.dropout_mp(h_v)
        return h_v, h_e

    def _final_merge(self, h_v, h_e, x, edge_index):
        """Merge node and edge info using final communicative kernel."""
        N = h_v.size(0)
        m_final = scatter_sum(h_e, edge_index[1], dim=0, dim_size=N)
        h_x = self.lin_x(x)  # project raw features into hidden space
        concat = torch.cat([h_v, m_final, h_x], dim=-1)
        return F.relu(self.final_update(concat))

    def _readout(self, h_v, batch):
        """Compute graph embedding and final prediction."""
        # handle empty graph case: no nodes -> no graphs
        if h_v.size(0) == 0:
            return torch.empty((0, self.n_tasks), device=h_v.device)
        batch_size = int(batch.max().item()) + 1
        H = h_v.size(1)
        lengths = torch.bincount(batch.cpu(), minlength=batch_size).to(batch.device)
        max_len = int(lengths.max().item())
        # compute prefix-sum offsets and pad sequences
        prefix = torch.cumsum(lengths, dim=0) - lengths
        padded = h_v.new_zeros((batch_size, max_len, H))
        idx = torch.arange(h_v.size(0), device=h_v.device)
        padded[batch, idx - prefix[batch]] = h_v
        # pack and run bi-GRU
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.readout_gru(packed)
        # h_n: [2, B, H] -> concat both directions and clamp activations
        h_n = torch.cat([h_n[0], h_n[1]], dim=1).clamp(-10., 10.)
        return h_n
class FFN(nn.Module, ABC):
    """
    An abstract base class for a feed-forward network.
    Defines the interface for mapping input features to output predictions.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        pass


class MLP(FFN):
    """
    A multilayer perceptron (MLP) implementing a feed-forward network.
    This network consists of an input layer, configurable hidden layers, and an output layer.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 300,
                 n_layers: int = 1,
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        Initializes the MLP.
        
        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output predictions.
            hidden_dim: Size of the hidden layers.
            n_layers: Number of hidden layers (not counting the output layer).
            dropout: Dropout probability.
            activation: Activation function name.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout

        act = get_activation_fn(activation)
        layers = []

        # First layer: input to hidden.
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Additional hidden layers.
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer.
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        return self.mlp(x)

    def __repr__(self):
        return (f"MLP(input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"hidden_dim={self.hidden_dim}, n_layers={self.n_layers}, dropout={self.dropout})")

    @classmethod
    def build(cls,
              input_dim: int,
              output_dim: int,
              hidden_dim: int = 300,
              n_layers: int = 1,
              dropout: float = 0.0,
              activation: str = 'relu') -> 'MLP':
        """
        Build an MLP model with the specified parameters.
        """
        return cls(input_dim=input_dim,
                   output_dim=output_dim,
                   hidden_dim=hidden_dim,
                   n_layers=n_layers,
                   dropout=dropout,
                   activation=activation)

class RegressionHead(nn.Module):
    """
    Plain regression head for continuous targets.
    
    Features
    --------
    • Configurable MLP backbone (width, depth, dropout, activation)  
    • Optional global residual connection input → output_dim  
    • No post-processing: returns raw real-valued predictions
    """
    def __init__(
        self,
        input_dim:  int,
        output_dim: int = 1,
        hidden_dim: int = 300,
        n_layers:   int = 1,
        dropout:    float = 0.0,
        activation: str   = "relu",
        use_residual: bool = False
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_dim  = hidden_dim
        self.n_layers    = n_layers
        self.dropout     = dropout
        self.activation  = activation
        self.use_residual = use_residual
        
        # core network
        self.net = MLP(input_dim, output_dim,
                       hidden_dim, n_layers,
                       dropout, activation)
        
        # optional linear skip
        if use_residual:
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.use_residual:
            out = out + self.skip(x)
        return out   # raw values; no normalisation

    def __repr__(self) -> str:
        return (f"RegressionHead(net={self.net}, "
                f"use_residual={self.use_residual})")

    def get_parameters(self):
        params = {"net": self.net}
        if self.use_residual:
            params["skip"] = self.skip
        return params

class PeriodicHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 300,
        n_layers: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu',
        use_residual: bool = False,
        normalize: bool = True
    ):
        """
        PeriodicHead with optional global residual connection.

        Args:
            input_dim:    dimensionality of input features
            output_dim:   number of periodic outputs (2 for sin/cos)
            hidden_dim:   width of hidden layers
            n_layers:     number of hidden layers in the MLP
            dropout:      dropout probability
            activation:   activation name (e.g. 'relu')
            use_residual: if True, adds a linear skip from input -> output_dim
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.normalize = normalize
        # build the MLP
        # core MLP
        self.net = MLP(input_dim, output_dim, hidden_dim, n_layers, dropout, activation)
        self.use_residual = use_residual
        if use_residual:
            # project input to match sin/cos output shape
            self.skip = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # main branch
        out = self.net(x)
        # add optional skip connection
        if self.use_residual:
            res = self.skip(x)
            out = out + res
        # normalize to unit circle
        if self.normalize:
            return out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out

    def __repr__(self):
        return f"PeriodicHead(net={self.net}, use_residual={self.use_residual})"

    def get_parameters(self):
        params = {'net': self.net}
        if self.use_residual:
            params['skip'] = self.skip
        return params
    


class RawPeriodicHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, normalize=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.normalize = normalize

    def forward(self, x):
        out = self.net(x)
        if self.normalize:
            out = out / (out.norm(dim=-1, keepdim=True) + 1e-8)
        return out

def get_activation_fn(activation: str) -> nn.Module:
    """
    Returns an instance of the activation function based on the provided string.
    """
    if activation.lower() == 'relu':
        return nn.ReLU()
    elif activation.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif activation.lower() == 'tanh':
        return nn.Tanh()
    elif activation.lower() == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unsupported activation function: {activation}")


# ----------------------------------------------------------------------------
def initialize_weights(model: nn.Module):
    """
    Initialize model weights in-place: xavier normals for matrices, zeros for biases.
    """
    for name, param in model.named_parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

class Identity(nn.Module):
    """Identity layer for PyTorch."""
    def forward(self, x):
        return x

def initialize_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
