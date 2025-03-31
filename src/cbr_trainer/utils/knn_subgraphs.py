import torch
from torch.nn import Module

import numpy as np
class Nneighbors(Module):
    def __init__(self, dataset_obj, device):
        super().__init__()
        self.dataset_obj = dataset_obj
        self.ent2id = dataset_obj.ent2id
        self.rel2id = dataset_obj.rel2id
        self.id2ent = dataset_obj.id2ent
        self.triples = {}
        for s, r2o in dataset_obj.full_adj_map.items():
            for r, o in r2o.items():
                self.triples[(s, r)] = o
        self.device = device
        self.entity_vectors = self.get_entity_vectors(dataset_obj).to(self.device)

    def get_entity_vectors(self, dataset_obj):
        # Create the relation based representation of the entities
        entity_vectors = np.zeros((len(self.ent2id), len(self.rel2id)))
        for e_ctr, e1 in enumerate(dataset_obj.full_edge_index[0]):
            entity_vectors[e1, dataset_obj.full_edge_attr[e_ctr]] = 1
        adj_mat = np.sqrt(entity_vectors)
        l2norm = np.linalg.norm(adj_mat, axis=-1)
        l2norm += np.finfo(np.float32).eps 
        entity_vectors = entity_vectors / l2norm.reshape(l2norm.shape[0], 1)
        return torch.Tensor(entity_vectors)

    def calc_sim(self, query_entities):
        """
        :param adj_mat: N X R
        :param query_entities: b is a batch of indices of query entities
        :return:
        """
        query_entities_vec = torch.index_select(self.entity_vectors, dim=0, index=query_entities)
        sim = torch.matmul(query_entities_vec, self.entity_vectors.T)
        return sim
    
    def forward(self, query_list, query_id, split, k=None):

        neighbor_list, neighbor_slices = [], [0]
        batch_e1s = []
        batch_query_nn = []
        for query_ctr, query in enumerate(query_list):
            e1, r = query[0], query[1]
            batch_e1s.append(self.ent2id[e1])
        batch_e1s = torch.LongTensor(batch_e1s).to(self.device)
        sim = self.calc_sim(batch_e1s)  # n X N (n== size of batch, N: size of all entities)
        nearest_neighbor_1_hop = np.argsort(-sim.cpu(), axis=-1)


        all_knn_ids = []

        for i in range(len(query_list)):
            knn_ids_with_r = []
            knn_idxs = nearest_neighbor_1_hop[i]
            

            e1, r = query_list[i][0], query_list[i][1]

           
            for idx in knn_idxs:
                
                if self.id2ent[idx.item()] != e1:
                    query_rel = [i for i in self.dataset_obj.train_idmap if i[0] == (self.id2ent[idx.item()])]

                    if len(query_rel) != 0:
                        knn_ids_with_r.append(self.dataset_obj.train_idmap[(self.id2ent[idx.item()], r)])

                        if len(knn_ids_with_r) >= (k + 5):
                            break

            all_knn_ids.append(knn_ids_with_r)
        
        for query_ctr, query in enumerate(query_list):
            # choose top-K

            if split == "train":
                neighbor_list.extend([self.dataset_obj.train_dataset[query_id]] + [self.dataset_obj.train_dataset[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])
                neighbor_slices.append(len(neighbor_list))
                batch_query_nn.extend([self.dataset_obj.raw_train_data[query_id]] + [self.dataset_obj.raw_train_data[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])

            if split == "test":
                neighbor_list.extend([self.dataset_obj.test_dataset[query_id]] + [self.dataset_obj.train_dataset[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])
                neighbor_slices.append(len(neighbor_list))
                batch_query_nn.extend([self.dataset_obj.raw_test_data[query_id]] + [self.dataset_obj.raw_train_data[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])
                
            if split == "dev":
                neighbor_list.extend([self.dataset_obj.dev_dataset[query_id]] + [self.dataset_obj.train_dataset[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])
                neighbor_slices.append(len(neighbor_list))
                batch_query_nn.extend([self.dataset_obj.raw_dev_data[query_id]] + [self.dataset_obj.raw_train_data[knn_id] for knn_id in all_knn_ids[query_ctr][:k]])
   
        assert neighbor_slices[-1] == len(neighbor_list)
        return neighbor_list, neighbor_slices, batch_query_nn