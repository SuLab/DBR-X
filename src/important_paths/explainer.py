import torch
import torch.nn as nn
import dgl
import numpy as np
from torch.nn import functional as F
from collections import defaultdict
import pandas as pd
from heapq import heappush, heappop
from itertools import count
from src.cbr_trainer.utils.knn_subgraphs import Nneighbors
from src.cbr_trainer.utils.dist_scr import L2Dist
from tqdm import tqdm
from .utils import *

class MaskExplainer:
    def __init__(self, model, device, args, split):
        self.model = model
        self.device = device
        self.args = args
        self.dist_fn = L2Dist
        self.mrn_nodes = None  # To be set by data_loader
        
        self.dist_fn = self.dist_fn("mean", "mean")
    
    
    def explain(self, drug_id, dis_id, data_loader, validation = False, split = "train" ):
        self.mrn_nodes = pd.DataFrame(data_loader.node_labels.items(), columns=['id', 'label'])  # Mock mrn_nodes
        self.nodes_mapping = data_loader.nodes_mapping

        self.nodes_mapping_inv = data_loader.nodes_mapping_inv

        
        
        # Initialize neighbors
        neighbors = Nneighbors(data_loader, self.device)
        
        
        
        disease_label = self.mrn_nodes[self.mrn_nodes["id"] == dis_id]["label"].iloc[0]
        drug_label = self.mrn_nodes[self.mrn_nodes["id"] == drug_id]["label"].iloc[0]
        
        batch, nn_graph, nn_slices, nn_batch, nn_idx = self.cbr_format(drug_id, 
                                                                       split, 
                                                                       data_loader, 
                                                                       neighbors)
        
        global_idx_map = self.nodes_mapping[disease_label][0][dis_id]  #global index
        local_idx_map = np.where(nn_graph[0].nodes[disease_label][0]["_ID"] == global_idx_map)#convert to local
        dist_pred, pred = self.model_cbr_dists(nn_graph, nn_idx, local_idx_map, disease_label, self.device)
        ml_edge_mask_dict = self.get_edge_mask_dict(nn_graph[0].to(self.device))
        optimizer = torch.optim.Adam(ml_edge_mask_dict.values(), lr=self.args.lr_, weight_decay=0)
        
        #mask prediction 
        global_drug = self.nodes_mapping[drug_label][0][batch["id"][0]]
        global_dis = self.nodes_mapping[disease_label][0][dis_id]

        local_dis = np.where(nn_graph[0].nodes[disease_label][0]["_ID"] == global_dis)[0].item()
        local_drug = np.where(nn_graph[0].nodes[drug_label][0]["_ID"] == global_drug)[0].item()

        ntype_hetero_nids_to_homo_nids = self.get_ntype_hetero_nids_to_homo_nids(nn_graph[0])    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(drug_label, int(local_drug))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(disease_label, int(local_dis))]
        
        for e in tqdm(range(self.args.num_epochs)):    
            
            # Apply sigmoid to edge_mask to get eweight
            ml_eweight_dict = {etype: ml_edge_mask_dict[etype].sigmoid() for etype in ml_edge_mask_dict}

            #Query + knn rep
            model_rep =[]
            for i, nn_ in enumerate(nn_graph):
                #get representation for query (apply mask)
                nn_ = nn_.to(self.device)
                if i ==0:
                    model_rep.append(self.model(nn_, feat_nids =nn_.ndata[dgl.NID], eweight_dict =ml_eweight_dict))

                #get representation of k-nn
                else: 
                    model_rep.append(self.model(nn_, feat_nids =nn_.ndata[dgl.NID]))

            knn_rep = []
            knn_true_false = []
            label_identifier =[]

            for i,rep  in enumerate(model_rep): 
                if i ==0 : 
                    query_rep, query_true_false_vec = self.concat_rep(rep, nn_idx[i], disease_label)
                    query_rep= F.normalize(query_rep,p =2, dim=1)
                else:
                    nn_all_rep, nn_true_false = self.concat_rep(rep, nn_idx[i], disease_label)
                    nn_all_rep= F.normalize(nn_all_rep,p =2, dim=1)

                    knn_rep.append(nn_all_rep)
                    knn_true_false.append(nn_true_false)
                    label_identifier.append(torch.full(nn_true_false.shape, i))

            knn_all_rep = torch.cat(knn_rep)
            knn_all_true_false = torch.cat(knn_true_false)
            label_identifier = torch.cat(label_identifier)


            #Distance between query node subgraph and KNN answer nodes
            #Distance between query node subgraph and KNN answer nodes
            query_rep = query_rep.to(self.device)
            knn_all_rep = knn_all_rep.to(self.device)
            label_identifier = label_identifier.to(self.device)
            dists = self.dist_fn(query_rep, 
                knn_all_rep[knn_all_true_false.long()], 
                target_identifiers=label_identifier) 

            # #get prediction with mask 
            score = query_rep[local_idx_map]

            pdist = nn.PairwiseDistance(p=2)
            pred_loss_val = pdist(pred, score)


            # #path loss 
            nn_graph[0] = nn_graph[0].to(self.device)
            nn_graph[0].edata['eweight'] = ml_eweight_dict
            ml_ghomo = dgl.to_homogeneous(nn_graph[0], edata= ["eweight"])
            ml_ghomo_eweights = ml_ghomo.edata['eweight']

            path_loss_val= self.path_loss(homo_src_nid, 
                                     homo_tgt_nid, 
                                     ml_ghomo,
                                     ml_ghomo_eweights, 
                                     num_paths=10, 
                                     penalty = self.args.penalty, 
                                     degree_thr= self.args.degree_thr)

            loss = pred_loss_val+ path_loss_val

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()



        comp_g_paths = self.get_paths(local_drug, 
                                      local_dis,
                                      drug_label,
                                      disease_label,
                                      nn_graph[0], 
                                      ml_eweight_dict)
        
        
        
        all_paths_id = []
        for path in comp_g_paths:
            path_id = []
            for element in path:
                src_type, rel, tgt_type, src_dgl, tgt_dgl = element[0][0], element[0][1], element[0][2], element[1], element[2]
                src_id, tgt_id = self.get_node_id(src_type, 
                                                  tgt_type, 
                                                  src_dgl,
                                                  tgt_dgl, 
                                                  nn_graph)
                path_id.append((src_id, rel, tgt_id))
            all_paths_id.append(path_id)
        
        if validation == True: 
            return pred, ml_ghomo, ml_eweight_dict, ml_ghomo_eweights, ntype_hetero_nids_to_homo_nids, nn_graph, nn_idx, all_paths_id
        else:
            return all_paths_id
    
    
    def cbr_format(self, drug_id, split, data_loader, neighbors):

        predictions = defaultdict(list)
        results = {}
        
        # Select the correct dataset based on the split
        if split == "train":
            dataset = data_loader.raw_train_data
        elif split == "test":
            dataset = data_loader.raw_test_data
        elif split == "dev":
            dataset = data_loader.raw_dev_data
        else:
            raise ValueError(f"Split '{split}' is not defined. Choose from 'train', 'test', or 'dev'.")

        
        for n in range(len(dataset)):
            if dataset[n]["id"][0] == drug_id: 
                i = n
            else:
                continue
                
        batch = dataset[i]   
        nn_graph, nn_slices, nn_batch = neighbors([batch["id"]], i, split, k=5) 
        new_batch_len = len(nn_slices) - 1
        nn_idx = self.return_local_index (nn_batch, nn_graph)

        return batch, nn_graph, nn_slices, nn_batch, nn_idx
    

    def return_local_index (self, nn_batch, nn_graph):
        """
        Given list of batches (query + knn)
        Get local index of answer
        """

        local_id_list = []
        for i in range(len(nn_batch)): 
            batch_answers = nn_batch[i]["answer"]

            batch_answers_local_idx = []
            for ans in batch_answers: 
                label_ans = self.mrn_nodes[self.mrn_nodes['id'] == ans]['label'].iloc[0]
                global_idx  = self.nodes_mapping[label_ans][0][ans]
                local_idx =np.where(nn_graph[i].nodes[label_ans][0]["_ID"] == global_idx)[0].item()
                batch_answers_local_idx.append(local_idx)
            local_id_list.append(batch_answers_local_idx)

        return local_id_list
    
    def model_cbr_dists(self, nn_graph,nn_idx, local_idx_map, tgt_ans, device):
    
        self.model.eval()
        #Get query and knn model representation 
        model_rep =[]
        for nn_ in (nn_graph):
            nn_ = nn_.to(self.device)
            model_rep.append(self.model(nn_, feat_nids =nn_.ndata[dgl.NID]))

        #Concatenate representation 
        #Create True/False tensor of query and knn answers
        knn_rep = []
        knn_true_false = []
        label_identifier =[]

        for i,rep  in enumerate(model_rep): 
            if i ==0 : 
                query_rep, query_true_false_vec = self.concat_rep(rep, nn_idx[i], tgt_ans)
                query_rep= F.normalize(query_rep,p =2, dim=1)
            else:
                nn_all_rep, nn_true_false = self.concat_rep(rep, nn_idx[i], tgt_ans)
                nn_all_rep= F.normalize(nn_all_rep,p =2, dim=1)

                knn_rep.append(nn_all_rep)
                knn_true_false.append(nn_true_false)
                label_identifier.append(torch.full(nn_true_false.shape, i))

        knn_all_rep = torch.cat(knn_rep)
        knn_all_true_false = torch.cat(knn_true_false)
        label_identifier = torch.cat(label_identifier)


        #Distance between query node subgraph and KNN answer nodes
        query_rep = query_rep.to(self.device)
        knn_all_rep = knn_all_rep.to(self.device)
        label_identifier = label_identifier.to(self.device)

        dists_pred = self.dist_fn(query_rep, 
            knn_all_rep[knn_all_true_false.long()], 
            target_identifiers=label_identifier) 


        pred_int = query_rep[local_idx_map]

        return dists_pred, pred_int
    
    def concat_rep(self, model_rep, ans_idx, tgt_ans):

        """
        model_rep: model representation. Dictionary, each key corresponds to node type
        ans_idx: index of true answers
        tgt_ans: answer node type of interest
        Return: Node representations and True/False vector indicating asnwer nodes
        """

        for key,val in model_rep.items(): 

            if key == tgt_ans:

                true_false_rep = torch.zeros(model_rep[key].shape[0])
                true_false_rep[ans_idx] =1
                return val, true_false_rep
    
    def get_edge_mask_dict(self, ghetero):
        '''
        Create a dictionary mapping etypes to learnable edge masks 

        Parameters
        ----------
        ghetero : heterogeneous dgl graph.

        Return
        ----------
        edge_mask_dict : dictionary
            key=etype, value=torch.nn.Parameter with size number of etype edges
        '''
        device = ghetero.device
        edge_mask_dict = {}
        for etype in ghetero.canonical_etypes:
            num_edges = ghetero.num_edges(etype)
            num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

            try:
                std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
                edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)

            except: 
                std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * 1))
                edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)

        return edge_mask_dict
    
    
    def get_ntype_hetero_nids_to_homo_nids(self, ghetero):
        '''
        Create a dictionary mapping tuples as (node type, node id) of the input heterogeneous graph
        to the node ids of the homogeneous version of the input graph.

        Parameters
        ----------
        ghetero : heterogeneous dgl graph

        Returns
        ----------
        ntype_hetero_nids_to_homo_nids : dict
        '''
        tmp = get_homo_nids_to_ntype_hetero_nids(ghetero)

        ntype_hetero_nids_to_homo_nids = {v: k for k, v in tmp.items()}
        return ntype_hetero_nids_to_homo_nids


    def get_neg_path_score_func(self, g, weight, exclude_node=[], penalty= 1, degree_thr= 10):
        '''
        Compute the negative path score for the shortest path algorithm.

        Parameters
        ----------
        g : dgl graph

        weight: string
           The edge weights stored in g.edata

        exclude_node : iterable
            Degree of these nodes will be set to 0 when computing the path score, so they will likely be included.

        Returns
        ----------
        neg_path_score_func: callable function
           Takes in two node ids and return the edge weight. 
        '''
        log_eweights = g.edata[weight].log().tolist() #edge weights
        log_in_degrees = g.in_degrees() #node degree
        log_in_degrees[exclude_node] = 0 #assign node degree 0 to tgt and src node
        log_in_degrees = log_in_degrees.tolist() 
        u, v = g.edges()

        neg_path_score_map = {edge : (penalty+(abs(log_in_degrees[edge[1]]-degree_thr))) - log_eweights[i] for i, edge in enumerate(zip(u.tolist(), v.tolist()))}


        def neg_path_score_func(u, v):
            return neg_path_score_map[(u, v)]
        return neg_path_score_func

    
    def k_shortest_paths_with_max_length(self, g, 
                                         src_nid, 
                                         tgt_nid, 
                                         weight=None, 
                                         k=5, 
                                         max_length=None,
                                         ignore_nodes=None,
                                         ignore_edges=None):

        """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
           each with maximum lenghth `max_length`, return starting from the shortest ones. 
           If a weighted shortest path search is to be used, no negative weights are allowed.

        Parameters
        ----------
           See function `k_shortest_paths_generator`

        Return
        -------
        paths: list of lists
           Each list is a path containing node ids
        """
        path_generator = self.k_shortest_paths_generator(g, 
                                                    src_nid, 
                                                    tgt_nid, 
                                                    weight=weight,
                                                    k=k, 
                                                    ignore_nodes_init=ignore_nodes,
                                                    ignore_edges_init=ignore_edges)

        try:
            if max_length:
                paths = [path for path in path_generator if len(path) <= max_length + 1]
            else:
                paths = list(path_generator)

        except ValueError:
            paths = [[]]

        return paths


    def k_shortest_paths_generator(self, g, 
                                   src_nid, 
                                   tgt_nid, 
                                   weight=None, 
                                   k=5, 
                                   ignore_nodes_init=None,
                                   ignore_edges_init=None):
        """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
           each with maximum lenghth `max_length`, return starting from the shortest ones. 
           If a weighted shortest path search is to be used, no negative weights are allowed.

        Adapted from NetworkX shortest_simple_paths
        https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

        Parameters
        ----------
        g : dgl graph

        src_nid : int
            source node id

        tgt_nid : int
            target node id

        weight: callable function, optional 
           Takes in two node ids and return the edge weight. 

        k: int
           number of paths

        ignore_nodes_init : set of nodes
           nodes to ignore, optional

        ignore_edges_init : set of edges
           edges to ignore, optional

        Returns
        -------
        path_generator: generator
           A generator that produces lists of tuples (path score, path), in order from
           shortest to longest. Each path is a list of node ids

        """
        if not weight:
            weight = lambda u, v: 1

        def length_func(path):

            return sum(weight(u, v) for (u, v) in zip(path, path[1:]))

        listA = list()
        listB = PathBuffer()
        prev_path = None
        while not prev_path or len(listA) < k:
            if not prev_path:
                length, path = bidirectional_dijkstra(g, src_nid, tgt_nid, weight, ignore_nodes_init, ignore_edges_init)

                listB.push(length, path)
            else:
                ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
                ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
                for i in range(1, len(prev_path)):
                    root = prev_path[:i]
                    root_length = length_func(root)
                    for path in listA:
                        if path[:i] == root:
                            ignore_edges.add((path[i - 1], path[i]))
                    try:
                        length, spur = bidirectional_dijkstra(g,
                                                              root[-1],
                                                              tgt_nid,
                                                              ignore_nodes=ignore_nodes,
                                                              ignore_edges=ignore_edges,
                                                              weight=weight)
                        path = root[:-1] + spur
                        listB.push(root_length + length, path)
                    except ValueError:
                        pass
                    ignore_nodes.add(root[-1])

            if listB:
                path = listB.pop()
                yield path
                listA.append(path)
                prev_path = path
            else:
                break
    def get_eids_on_paths(self, paths, ghomo):
        '''
        Collect all edge ids on the paths

        Note: The current version is a list version. An edge may be collected multiple times
        A different version is a set version where an edge can only contribute one time 
        even it appears in multiple paths

        Parameters
        ----------
        ghomo : dgl homogeneous graph

        Returns
        -------
        paths: list of lists
            Each list contains (source node ids, target node ids)

        '''
        row, col = ghomo.edges()
        eids = []
        for path in paths:
            for i in range(len(path)-1):
                eids += [((row == path[i]) & (col == path[i+1])).nonzero()][0]

        return torch.LongTensor(eids)


    def path_loss(self, src_nid, tgt_nid, g, eweights, num_paths=5, penalty=1, degree_thr=10):
        
 
        
            """Compute the path loss.

            Parameters
            ----------
            src_nid : int
                source node id

            tgt_nid : int
                target node id

            g : dgl graph

            eweights : Tensor
                Edge weights with shape equals the number of edges.

            num_paths : int
                Number of paths to compute path loss on

            Returns
            -------
            loss : Tensor
                The path loss
            """
            neg_path_score_func = self.get_neg_path_score_func(g, 'eweight', [src_nid, tgt_nid], penalty, degree_thr)
            paths = self.k_shortest_paths_with_max_length(g, 
                                                     src_nid, 
                                                     tgt_nid, 
                                                     weight=neg_path_score_func, 
                                                     k=num_paths)

            eids_on_path = self.get_eids_on_paths(paths, g)

            if eids_on_path.nelement() > 0:
                loss_on_path =  -eweights[eids_on_path].mean()
            else:
                loss_on_path = 0

            eids_off_path_mask = ~torch.isin(torch.arange(eweights.shape[0]), eids_on_path)
            if eids_off_path_mask.any():
                loss_off_path = eweights[eids_off_path_mask].mean()
            else:
                loss_off_path = 0

            loss = self.args.alpha * loss_on_path + self.args.beta * loss_off_path 

            return loss   

        

    def get_paths(self,
                  src_nid, 
                  tgt_nid, 
                  src_ntype,
                  tgt_ntype,
                  ghetero,
                  edge_mask_dict,
                  num_paths=200, 
                  max_path_length=3):

        """A postprocessing step that turns the `edge_mask_dict` into actual paths.

        Parameters
        ----------
        edge_mask_dict : dict
            key=`etype`, value=torch.nn.Parameter with size being the number of `etype` edges

        Others: see the `explain` method.

        Returns
        -------
        paths: list of lists
            each list contains (cannonical edge type, source node ids, target node ids)
        """
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        eweight_dict = {etype: edge_mask_dict[etype].sigmoid() for etype in edge_mask_dict}
        ghetero.edata['eweight'] = edge_mask_dict

        # convert ghetero to ghomo and find paths
        ghomo = dgl.to_homogeneous(ghetero, edata=['eweight'])
        ntype_hetero_nids_to_homo_nids = self.get_ntype_hetero_nids_to_homo_nids(ghetero)    
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(tgt_ntype, int(tgt_nid))]

        neg_path_score_func = self.get_neg_path_score_func(ghomo, 'eweight', [src_nid, tgt_nid])
        homo_paths = self.k_shortest_paths_with_max_length(ghomo, 
                                                       homo_src_nid, 
                                                       homo_tgt_nid,
                                                       weight=neg_path_score_func,
                                                       k=num_paths,
                                                       max_length=max_path_length)

        paths = []
        homo_nids_to_ntype_hetero_nids = get_homo_nids_to_ntype_hetero_nids(ghetero)

        if len(homo_paths) > 0:
            for homo_path in homo_paths:

                hetero_path = []
                for i in range(1, len(homo_path)):
                    homo_u, homo_v = homo_path[i-1], homo_path[i]
                    hetero_u_ntype, hetero_u_nid = homo_nids_to_ntype_hetero_nids[homo_u] 
                    hetero_v_ntype, hetero_v_nid = homo_nids_to_ntype_hetero_nids[homo_v] 

                    can_etype = ntype_pairs_to_cannonical_etypes[(hetero_u_ntype, hetero_v_ntype)] 
                    hetero_path += [(can_etype, hetero_u_nid, hetero_v_nid)]
                paths += [hetero_path]

        else:
            # A rare case, no paths found, take the top edges
            cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
            M = len(cat_edge_mask)
            k = min(num_paths * max_path_length, M)
            threshold = cat_edge_mask.topk(k)[0][-1].item()
            path = []
            for etype in edge_mask_dict:
                u, v = ghetero.edges(etype=etype)  
                topk_edge_mask = edge_mask_dict[etype] >= threshold
                path += list(zip([etype] * topk_edge_mask.sum().item(), u[topk_edge_mask].tolist(), v[topk_edge_mask].tolist()))                
            paths = [path]

        return paths

    
    
    def get_ntype_pairs_to_cannonical_etypes(self, ghetero, pred_etype='likes'):
        '''
        Create a dictionary mapping tuples as (source node type, target node type) to 
        cannonical edge types. Edges wity type `pred_etype` will be excluded.
        A helper function for path finding.
        Only works if there is only one edge type between any pair of node types.

        Parameters
        ----------
        ghetero : heterogeneous dgl graph

        pred_etype : string
            The edge type for prediction

        Returns
        ----------
        ntype_pairs_to_cannonical_etypes : dict
        '''
        ntype_pairs_to_cannonical_etypes = {}
        for src_ntype, etype, tgt_ntype in ghetero.canonical_etypes:
            if etype != pred_etype:
                ntype_pairs_to_cannonical_etypes[(src_ntype, tgt_ntype)] = (src_ntype, etype, tgt_ntype)
                
        return ntype_pairs_to_cannonical_etypes
    
    
    def get_node_id(self, src_type, tgt_type, src_idx, tgt_idx, nn_graph):
    
        global_src_idx = nn_graph[0].nodes[src_type][0]["_ID"][src_idx].item()
        global_tgt_idx = nn_graph[0].nodes[tgt_type][0]["_ID"][tgt_idx].item()



        return self.nodes_mapping_inv[src_type][0][global_src_idx], self.nodes_mapping_inv[tgt_type][0][global_tgt_idx]
class PathBuffer:
    """For shortest paths finding
    
    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    """
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path