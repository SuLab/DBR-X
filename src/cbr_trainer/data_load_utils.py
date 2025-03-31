import os
import pickle
import pandas as pd
import torch
import dgl
from collections import defaultdict
from tqdm import tqdm

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
    
    # Load nodes with optimized pandas reading
    nodes_df = pd.read_csv(data_nodes, usecols=['id', 'label'])
    node_labels = nodes_df.set_index('id')['label'].to_dict()
    
    # Load paths
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
    # More efficient list comprehension
    return [(data[i], data[i+1], data[i+2]) for i in range(0, len(data)-2)]

def extract_path_triples(all_paths):
    """
    Extract triples from paths with optimized data structures.
    
    Args:
        all_paths (dict): Dictionary containing path information.
        
    Returns:
        DataFrame: DataFrame containing source, relation, target information.
    """
    # Preallocate lists for better memory management
    data = []
    
    # Process drugs in batches
    for drug in all_paths.keys():
        # Process paths for current drug
        for path in all_paths[drug][0]:
            # Build path
            one_path = [drug]
            for trip in path:
                one_path.extend([trip[0], trip[1]])  # Extend is faster than multiple appends
            
            # Convert to triples and extract even indices
            triples = convert_list(one_path)
            for i in range(0, len(triples), 2):
                data.append((triples[i][0], triples[i][1], triples[i][2]))
    
    # Create DataFrame directly from data
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
    # Use pandas apply for better performance
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
    # Group nodes by label
    set_nodes_label = defaultdict(list)
    for node, label in node_labels.items():
        set_nodes_label[label].append(node)
    
    # Create mappings more efficiently
    set_nodes_label_id = {}
    set_nodes_label_id_rev = {}
    
    for label, nodes in set_nodes_label.items():
        # Create mappings in one pass
        dict_label = {node: i for i, node in enumerate(nodes)}
        dict_label_rev = {i: node for i, node in enumerate(nodes)}
        
        # Store as dict instead of list for direct access
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
    # Group by edge type for better performance
    grouped_data = mrn_graph.groupby(['src_type', 'rel', 'tgt_type'])
    
    graph_data = {}
    
    # Process each edge type
    for (src_label, rel, tgt_label), group in grouped_data:
        src_ids = []
        tgt_ids = []
        
        # Get ID mappings for current types
        src_mapping = set_nodes_label_id[src_label][0]
        tgt_mapping = set_nodes_label_id[tgt_label][0]
        
        # Map node IDs to indices
        for _, row in group.iterrows():
            src = row['str']
            tgt = row['tgt']
            src_ids.append(src_mapping[src])
            tgt_ids.append(tgt_mapping[tgt])
        
        # Create tensors directly
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
    # Load data
    all_paths, node_labels = load_data_paths(data_dir, data_name, paths_file_dir)
    
    # Extract triples from paths
    mrn_graph = extract_path_triples(all_paths)
    
    # Add node types
    mrn_graph = add_node_types(mrn_graph, node_labels)
    
    # Create node mappings
    set_nodes_label, set_nodes_label_id, _ = create_node_mappings(node_labels)
    
    # Create DGL graph
    graph_mrn_dgl = create_dgl_graph(mrn_graph, node_labels, set_nodes_label_id)
    
    return graph_mrn_dgl, set_nodes_label