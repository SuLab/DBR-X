import os
import pickle
import pandas as pd
import torch
import dgl
from collections import defaultdict
from tqdm import tqdm
import numpy as np


import dgl
import torch
import random
import textwrap
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dgl.subgraph import khop_in_subgraph
from itertools import count
from heapq import heappop, heappush
from sklearn.metrics import roc_auc_score
import logging 
logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_args(args):
    print("Arguments:", vars(args))

def load_data_paths(data_dir, data_name, paths_file_dir):
    """
    Load paths and node data from files efficiently.
    
    Args:
        data_dir (str): Directory containing the data.
        data_name (str): Name of the data.
        paths_file_dir (str): Directory containing paths file.
        
    Returns:
        tuple: Loaded graph paths and node labels dictionary.
    """
    graph_pkl = os.path.join(data_dir, "subgraphs", paths_file_dir)
    data_nodes = os.path.join(data_dir, f"{data_name}_nodes", "nodes.csv")
    
    nodes_df = pd.read_csv(data_nodes, usecols=['id', 'label'])
    node_labels = nodes_df.set_index('id')['label'].to_dict()
    
    with open(graph_pkl, "rb") as fin:
        all_paths = pickle.load(fin)
        
    return all_paths, node_labels

def convert_list(data):
    """
    Converts a list of relationships to a list of triples more efficiently.
    
    Args:
        data (list): List of path elements.
        
    Returns:
        list: List of triples.
    """
    return [(data[i], data[i+1], data[i+2]) for i in range(0, len(data)-2)]

def extract_path_triples(all_paths):
    """
    Extract triples from paths with optimized data structures.
    
    Args:
        all_paths (dict): Dictionary containing path information.
        
    Returns:
        DataFrame: DataFrame containing source, relation, target information.
    """
    data = []
    
    for drug in all_paths.keys():
        for path in all_paths[drug][0]:
            one_path = [drug]
            for trip in path:
                one_path.extend([trip[0], trip[1]])
            triples = convert_list(one_path)
            for i in range(0, len(triples), 2):
                data.append((triples[i][0], triples[i][1], triples[i][2]))
    
    mrn_paths = pd.DataFrame(data, columns=["str", "rel", "tgt"])
    return mrn_paths.drop_duplicates()

def add_node_types(mrn_graph, node_labels):
    """
    Add node types to the graph using vectorized operations.
    
    Args:
        mrn_graph (DataFrame): Graph DataFrame.
        node_labels (dict): Dictionary mapping node IDs to labels.
        
    Returns:
        DataFrame: Graph with node types added.
    """
    mrn_graph["src_type"] = mrn_graph['str'].map(node_labels)
    mrn_graph["tgt_type"] = mrn_graph['tgt'].map(node_labels)
    return mrn_graph

def create_node_mappings(node_labels):
    """
    Create node mappings by label with optimized data structures.
    
    Args:
        node_labels (dict): Dictionary mapping node IDs to labels.
        
    Returns:
        tuple: Set of nodes by label and node ID mappings.
    """
    set_nodes_label = defaultdict(list)
    for node, label in node_labels.items():
        set_nodes_label[label].append(node)
    
    set_nodes_label_id = {}
    set_nodes_label_id_rev = {}
    
    for label, nodes in set_nodes_label.items():
        dict_label = {node: i for i, node in enumerate(nodes)}
        dict_label_rev = {i: node for i, node in enumerate(nodes)}
        set_nodes_label_id[label] = [dict_label]
        set_nodes_label_id_rev[label] = [dict_label_rev]
    
    return set_nodes_label, set_nodes_label_id, set_nodes_label_id_rev

def create_dgl_graph(mrn_graph, node_labels, set_nodes_label_id):
    """
    Create a DGL heterograph efficiently.
    
    Args:
        mrn_graph (DataFrame): Graph DataFrame.
        node_labels (dict): Dictionary mapping node IDs to labels.
        set_nodes_label_id (dict): Node mappings by label.
        
    Returns:
        DGLHeteroGraph: Created DGL heterograph.
    """
    grouped_data = mrn_graph.groupby(['src_type', 'rel', 'tgt_type'])
    graph_data = {}
    
    for (src_label, rel, tgt_label), group in grouped_data:
        src_ids = []
        tgt_ids = []
        src_mapping = set_nodes_label_id[src_label][0]
        tgt_mapping = set_nodes_label_id[tgt_label][0]
        
        for _, row in group.iterrows():
            src = row['str']
            tgt = row['tgt']
            src_ids.append(src_mapping[src])
            tgt_ids.append(tgt_mapping[tgt])
        
        graph_data[(src_label, rel, tgt_label)] = (
            torch.tensor(src_ids, dtype=torch.int64),
            torch.tensor(tgt_ids, dtype=torch.int64)
        )
    
    return dgl.heterograph(graph_data)

def process_graph_data(data_dir, data_name, paths_file_dir):
    """
    Main function that processes graph data efficiently.
    
    Args:
        data_dir (str): Directory containing the data.
        data_name (str): Name of the data.
        paths_file_dir (str): Directory containing paths file.
        
    Returns:
        tuple: DGL graph and set of nodes by label.
    """
    all_paths, node_labels = load_data_paths(data_dir, data_name, paths_file_dir)
    mrn_graph = extract_path_triples(all_paths)
    mrn_graph = add_node_types(mrn_graph, node_labels)
    set_nodes_label, set_nodes_label_id, _ = create_node_mappings(node_labels)
    graph_mrn_dgl = create_dgl_graph(mrn_graph, node_labels, set_nodes_label_id)
    return graph_mrn_dgl, set_nodes_label




    
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_args(args):
    for k, v in vars(args).items():
        print(f'{k:25} {v}')
        
def set_config_args(args, config_path, dataset_name, model_name=''):
    with open(config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)[dataset_name]
        if model_name:
            config = config[model_name]

    for key, value in config.items():
        setattr(args, key, value)
    return args
    
'''
Model training utils
'''
def idx_split(idx, ratio, seed=0):
    """
    Randomly split `idx` into idx1 and idx2, where idx1 : idx2 = `ratio` : 1 - `ratio`
    
    Parameters
    ----------
    idx : tensor
        
    ratio: float
 
    Returns
    ----------
        Two index (tensor) after split
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def eids_split(eids, val_ratio, test_ratio, seed=0):
    """
    Split `eids` into three parts: train, valid, and test,
    where train : valid : test = (1 - `val_ratio` - `test_ratio`) : `val_ratio` : `test_ratio`
    
    Parameters
    ----------
    eid : tensor
        edge id
        
    val_ratio : float
    
    test_ratio : float

    seed : int

    Returns
    ----------
        Three edge ids (tensor) after split
    """
    train_ratio = (1 - val_ratio - test_ratio)
    train_eids, pred_eids = idx_split(eids, train_ratio, seed)
    val_eids, test_eids = idx_split(pred_eids, val_ratio / (1 - train_ratio), seed)
    return train_eids, val_eids, test_eids

def negative_sampling(graph, pred_etype=None, num_neg_samples=None):
    '''
    Adapted from PyG negative_sampling function
    https://pytorch-geometric.readthedocs.io/en/1.7.2/_modules/torch_geometric/utils/
    negative_sampling.html#negative_sampling

    Parameters
    ----------
    graph : dgl graph
    
    pred_etype : string
        The edge type for prediction

    num_neg_samples : int
    
    Returns
    ----------
        Two negative nids. Nids for src and tgt nodes of the `pred_etype`
    '''
    # src_N: total number of src nodes
    # N (tgt_N): total number of tgt nodes
    # M: total number of possible edges, square of src_N * tgt_N
    # pos_M: number of positive samples (observed edges)
    # neg_M: number of negative samples
    pos_src_nids, pos_tgt_nids = graph.edges(etype=pred_etype)
    if pred_etype is None:
        N = graph.num_nodes()
        M = N * N
    else:
        src_ntype, _, tgt_ntype = graph.to_canonical_etype(pred_etype) 
        src_N, N = graph.num_nodes(src_ntype), graph.num_nodes(tgt_ntype)
        M = src_N * N

    pos_M = pos_src_nids.shape[0]
    neg_M = num_neg_samples or pos_M
    neg_M = min(neg_M, M - pos_M) # incase M - pos_M < neg_M

    # Percentage of edges to opos_tgt_nidsersample, so only need to sample once in most cases
    alpha = abs(1 / (1 - 1.1 * (pos_M / M)))
    size = min(M, int(alpha * neg_M))
    perm = torch.tensor(random.sample(range(M), size))
    
    idx = pos_src_nids * N + pos_tgt_nids
    # mask = torch.from_npos_src_nidsmpy(np.isin(perm, idx.to('cppos_src_nids'))).to(torch.bool)
    mask = torch.isin(perm, idx.to('cpu')).to(torch.bool)
    perm = perm[~mask][:neg_M].to(pos_src_nids.device)

    neg_src_nids = torch.div(perm, N, rounding_mode='floor')
    neg_tgt_nids = perm % N

    return neg_src_nids, neg_tgt_nids

'''
DGL graph manipulation utils
'''
def get_homo_nids_to_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to the node ids of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_hetero_nids : dict
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    homo_nids_to_hetero_nids = dict(zip(homo_nids, hetero_nids))
    print(homo_nids_to_hetero_nids)
    return homo_nids_to_hetero_nids

def get_homo_nids_to_ntype_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to tuples as (node type, node id) of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_ntype_hetero_nids : dict
    '''

    # print(ghetero)
    # logger.info(ghetero)
    ghomo = dgl.to_homogeneous(ghetero)
    
    homo_nids = range(ghomo.num_nodes())
    ntypes = ghetero.ntypes
    # This line relies on the default order of ntype_ids is the order in ghetero.ntypes
    ntypes = [ntypes[i] for i in ghomo.ndata[dgl.NTYPE]] 
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    ntypes_hetero_nids = list(zip(ntypes, hetero_nids))
    homo_nids_to_ntype_hetero_nids = dict(zip(homo_nids, ntypes_hetero_nids))
    return homo_nids_to_ntype_hetero_nids

def get_ntype_hetero_nids_to_homo_nids(ghetero):
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

def get_ntype_pairs_to_cannonical_etypes(ghetero, pred_etype='likes'):
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

def get_num_nodes_dict(ghetero):
    '''
    Create a dictionary containing number of nodes of all ntypes in a heterogeneous graph
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    Returns 
    ----------
    num_nodes_dict : dict
        key=node type, value=number of nodes
    '''
    num_nodes_dict = {}
    for ntype in ghetero.ntypes:
        num_nodes_dict[ntype] = ghetero.num_nodes(ntype)    
    return num_nodes_dict

def remove_all_edges_of_etype(ghetero, etype):
    '''
    Remove all edges with type `etype` from `ghetero`. If `etype` is not in `ghetero`, do nothing.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    etype : string or triple of strings
        Edge type in simple form (string) or cannonical form (triple of strings)
    
    Returns 
    ----------
    removed_ghetero : heterogeneous dgl graph
        
    '''
    etype = ghetero.to_canonical_etype(etype)
    if etype in ghetero.canonical_etypes:
        eids = ghetero.edges('eid', etype=etype)
        removed_ghetero = dgl.remove_edges(ghetero, eids, etype=etype)
    else:
        removed_ghetero = ghetero
    return removed_ghetero

def hetero_src_tgt_khop_in_subgraph(src_ntype, src_nid, tgt_ntype, tgt_nid, ghetero, k):
    '''
    Find the `k`-hop subgraph around the src node and tgt node in `ghetero`
    The output will be the union of two subgraphs.
    See the dgl `khop_in_subgraph` function as a referrence
    https://docs.dgl.ai/en/0.9.x/generated/dgl.khop_in_subgraph.html
    
    Parameters
    ----------
    src_ntype: string
        source node type
    
    src_nid : int
        source node id

    tgt_ntype: string
        target node type

    tgt_nid : int
        target node id

    ghetero : heterogeneous dgl graph

    k: int
        Number of hops

    Return
    ----------
    sghetero_src_nid: int
        id of the source node in the subgraph

    sghetero_tgt_nid: int
        id of the target node in the subgraph

    sghetero : heterogeneous dgl graph
        Union of two k-hop subgraphs

    sghetero_feat_nid: Tensor
        The original `ghetero` node ids of subgraph nodes, for feature identification
    
    '''
    # Extract k-hop subgraph centered at the (src, tgt) pair
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid
    
    if src_ntype == tgt_ntype:
        pred_dict = {src_ntype: torch.tensor([src_nid, tgt_nid])}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype][0]
        sghetero_tgt_nid = inv_dict[tgt_ntype][1]
    else:
        pred_dict = {src_ntype: src_nid, tgt_ntype: tgt_nid}
        sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
        sghetero_src_nid = inv_dict[src_ntype]
        sghetero_tgt_nid = inv_dict[tgt_ntype]

    sghetero_feat_nid = sghetero.ndata[dgl.NID]
    
    return sghetero_src_nid, sghetero_tgt_nid, sghetero, sghetero_feat_nid


'''
Path finding utils
'''

def bidirectional_dijkstra(g, src_nid, tgt_nid, weight=None, ignore_nodes=None, ignore_edges=None):
    """Dijkstra's algorithm for shortest paths using bidirectional search.
    
    Adapted from NetworkX _bidirectional_dijkstra
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

    ignore_nodes : container of nodes
       nodes to ignore, optional

    ignore_edges : container of edges
       edges to ignore, optional

    Returns
    -------
    length : number
        Shortest path length.

    """
    if src_nid == tgt_nid:
        return (0, [src_nid])

    src, tgt = g.edges()
    Gpred = lambda i: src[tgt == i].tolist()
    Gsucc = lambda i: tgt[src == i].tolist()
    
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)
    
    if ignore_edges:
        def filter_pred_iter(pred_iter):
            def iterate(v):
                for w in pred_iter(v):
                    if (w, v) not in ignore_edges:
                        yield w

            return iterate

        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w

            return iterate

        Gpred = filter_pred_iter(Gpred)
        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{src_nid: [src_nid]}, {tgt_nid: [tgt_nid]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{src_nid: 0}, {tgt_nid: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), src_nid))
    push(fringe[1], (0, next(c), tgt_nid))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    if not weight:
        weight = lambda u, v: 1
            
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = weight(v, w)
                vwLength = dists[dir][v] + minweight
            else:  # back, must remember to change v,w->w,v
                minweight = weight(w, v)
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise ValueError("No paths found")




