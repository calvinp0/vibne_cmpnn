from torch_geometric.data import Batch

from torch_geometric.data import Batch

def collate_pairs(pair_list):
    """
    pair_list : list[(Data, Data)]
        The `DataLoader` gives us a list of *pairs* for one minibatch.
        We have to flatten it into 2 × N individual graphs and then
        hand it to `Batch.from_data_list`, so the result has all the usual
        PyG attributes (x, edge_index, batch, num_graphs, …).
    """
    graphs = [g for pair in pair_list for g in pair]          # flatten
    return Batch.from_data_list(graphs)                       # -> Batch
