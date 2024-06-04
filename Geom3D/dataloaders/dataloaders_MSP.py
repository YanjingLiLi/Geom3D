import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import numpy as np


class BatchMultiPro(Data):
    def __init__(self, **kwargs):
        super(BatchMultiPro, self).__init__(**kwargs)
        return

    @staticmethod
    def from_data_list(data_list):
        batch = BatchMultiPro()

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))

        for key in keys:
            batch[key] = []

        batch.batch_protein_1 = []
        batch.batch_protein_2 = []

        for i, data in enumerate(data_list):
            num_nodes_protein_1 = data.num_nodes_1
            num_nodes_protein_2 = data.num_nodes_2
            batch.batch_protein_1.append(torch.full((num_nodes_protein_1,), i, dtype=torch.long))
            batch.batch_protein_2.append(torch.full((num_nodes_protein_2,), i, dtype=torch.long))
            
            for key in data.keys:
                item = data[key]
                batch[key].append(item)

        
        for key in keys:
            if key not in ["x_1", "x_2", "id"]:
                batch[key] = torch.cat(batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
            else:
                batch[key] = np.array(batch[key]).flatten().tolist()

        batch.batch_protein_1 = torch.cat(batch.batch_protein_1, dim=-1)
        batch.batch_protein_2 = torch.cat(batch.batch_protein_2, dim=-1)

        return batch.contiguous()
    
    @property
    def num_graphs(self):
        '''Returns the number of graphs in the batch.'''
        return self.batch[-1].item() + 1


class DataLoaderMultiPro(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMultiPro, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: BatchMultiPro.from_data_list(data_list),
            **kwargs)