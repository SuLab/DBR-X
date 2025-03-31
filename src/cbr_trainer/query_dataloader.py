import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import torch
import pickle
import dgl 
import os
import logging
from .data_load_utils import process_graph_data, convert_list

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class DataLoader:
    def __init__(self, data_dir, data_name, paths_file_dir):
        self.data_name = data_name
        self.data_dir = data_dir
        
#         # Load DGL graph and node mappings
        logger.info("Loading and processing graph data...")
        self.g_mrn, self.set_nodes_label = process_graph_data(data_dir, data_name, paths_file_dir)

        # Create node mappings efficiently
        logger.info("Creating node mappings...")
        self.nodes_mapping = self._create_node_mappings(self.set_nodes_label)
        
#         # Create inverse node mappings
        self.nodes_mapping_inv = self._create_inverse_mappings(self.nodes_mapping)
       
        
        
        #Inverse node mapping 
        self.nodes_mapping_inv = defaultdict(list)
        for nd_type in self.nodes_mapping: 

            inverted_dict = {v: k for k, v in self.nodes_mapping[nd_type][0].items()}
            self.nodes_mapping_inv[nd_type].append(inverted_dict)

        
        # Load paths data once
        logger.info("Loading paths data...")
        self.all_paths = self.load_paths(data_dir, data_name, paths_file_dir)
        
        # Initialize dictionaries for entities and relations
        self.ent2id = {}
        self.rel2id = {}
        self.id2ent = {}
        self.id2rel = {}
        
        # Initialize graph structures
        self.full_adj_map = {}
        self.full_edge_index = [[], []]
        self.full_edge_attr = []
        
        # Process all CSV data at once
        logger.info("Processing CSV data...")
        self._process_csv_data()
        
        # Process path data
        logger.info("Processing path data...")
        self._process_path_data()
        
        # Build node labels
        logger.info("Building node labels...")
        self.node_labels = self._get_node_labels()
        
        # Build datasets
        logger.info("Building train dataset...")
        self.train_dataset = self._build_dataset(self.raw_train_data)
        
        logger.info("Building test dataset...")
        self.test_dataset = self._build_dataset(self.raw_test_data)
        
        logger.info("Building dev dataset...")
        self.dev_dataset = self._build_dataset(self.raw_dev_data)
        
        # Create entity-to-index mapping for training data
        self.train_idmap = self._create_id_map(self.raw_train_data)

    def _create_node_mappings(self, set_nodes_label):
        """Create node mappings more efficiently"""
        nodes_mapping = {}
        for node_type, node_ids in set_nodes_label.items():
            nodes_mapping[node_type] = [{node_id: i for i, node_id in enumerate(node_ids)}]
        return nodes_mapping
    
    def _create_inverse_mappings(self, nodes_mapping):
        """Create inverse node mappings more efficiently"""
        nodes_mapping_inv = {}
        for nd_type, mapping_list in nodes_mapping.items():
            nodes_mapping_inv[nd_type] = [{v: k for k, v in mapping_list[0].items()}]
        return nodes_mapping_inv
    
    def _process_csv_data(self):
        """Process all CSV data at once"""
        split_data = {}

        # Load all CSV files at once
        for split_name in ['train00', 'test', 'dev']:
            split_path = os.path.join(self.data_dir, f'{split_name}.csv')
            split_data[split_name] = pd.read_csv(split_path)

        # Initialize data structures
        self.raw_train_data = []
        self.raw_train_data_map = defaultdict(list)
        self.raw_dev_data = []
        self.raw_dev_data_map = defaultdict(list)
        self.raw_test_data = []
        self.raw_test_data_map = defaultdict(list)

        full_etype_nodes = defaultdict(set)  # Use sets to avoid duplicates
        full_etype_edges = defaultdict(list)

        # Process each split
        for split_name, df in split_data.items():
            raw_data_dict = {}  # Use a dictionary to aggregate answers with the same id
            raw_data_map = defaultdict(list)

            for _, row in df.iterrows():
                e1, r, e2 = row['start_id'], row['type'], row['end_id']

                # Update entity and relation dictionaries
                if e1 not in self.ent2id:
                    self.ent2id[e1] = len(self.ent2id)
                if e2 not in self.ent2id:
                    self.ent2id[e2] = len(self.ent2id)
                if r not in self.rel2id:
                    self.rel2id[r] = len(self.rel2id)

                # Update edge type nodes and edges
                full_etype_nodes[r].add(e1)
                full_etype_nodes[r].add(e2)
                full_etype_edges[r].append((e1, e2))

                # Update adjacency map and edge indices
                self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
                self.full_edge_index[0].append(self.ent2id[e1])
                self.full_edge_index[1].append(self.ent2id[e2])
                self.full_edge_attr.append(self.rel2id[r])

                # Update raw data map
                raw_data_map[(e1, r)].append(e2)

                # Update raw data dict
                id_key = (e1, r)
                if id_key not in raw_data_dict:
                    raw_data_dict[id_key] = {
                        "id": id_key,
                        "seed_entities": [e1],
                        "question": r,
                        "answer": [e2]
                    }
                else:
                    # Add e2 to existing answers if not already present
                    if e2 not in raw_data_dict[id_key]["answer"]:
                        raw_data_dict[id_key]["answer"].append(e2)

            # Convert dictionary to list
            raw_data = list(raw_data_dict.values())

            # Assign to appropriate data structures
            if split_name == 'train00':
#                 print(raw_data)
                self.raw_train_data = raw_data
                self.raw_train_data_map = raw_data_map
            elif split_name == 'dev':
                self.raw_dev_data = raw_data
                self.raw_dev_data_map = raw_data_map
            elif split_name == 'test':
                self.raw_test_data = raw_data
                self.raw_test_data_map = raw_data_map
    
    def _process_path_data(self):
        """Process path data efficiently"""
        # Use DataFrame to efficiently process path data
        str_list, rel_list, tgt_list = [], [], []
        
        # Process paths in batches
        for drug in tqdm(self.all_paths.keys()):
            if drug not in self.all_paths:
                continue
                
            # Extract paths for current drug
            for path in self.all_paths[drug][0]:
                one_path = [drug]
                
                for trip in path:
                    edge, node = trip[0], trip[1]
                    one_path.append(edge)
                    one_path.append(node)
                
                # Convert path to triples
                triples = convert_list(one_path)
                
                for i in range(0, len(triples), 2):
                    src, rel, tgt = triples[i]
                    str_list.append(src)
                    rel_list.append(rel)
                    tgt_list.append(tgt)
        
        # Create dataframe and drop duplicates
        graph = pd.DataFrame({
            "e1": str_list,
            "r": rel_list,
            "e2": tgt_list
        }).drop_duplicates()
        
        # Process the graph data
        for e1, r, e2 in zip(graph["e1"], graph['r'], graph['e2']):
            # Skip processing if entity already exists in dictionaries
            if e1 not in self.ent2id:
                self.ent2id[e1] = len(self.ent2id)
            if e2 not in self.ent2id:
                self.ent2id[e2] = len(self.ent2id)
            if r not in self.rel2id:
                self.rel2id[r] = len(self.rel2id)
                r_inv = r + "_inv"
                self.rel2id[r_inv] = len(self.rel2id)
            
            # Update adjacency map and edge indices
            self.full_adj_map.setdefault(e1, {}).setdefault(r, []).append(e2)
            self.full_edge_index[0].append(self.ent2id[e1])
            self.full_edge_index[1].append(self.ent2id[e2])
            self.full_edge_attr.append(self.rel2id[r])
        
        # Create inverse mappings
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        
        # Convert to tensors
        self.full_edge_index = torch.LongTensor(self.full_edge_index)
        self.full_edge_attr = torch.LongTensor(self.full_edge_attr)
    
    def _build_dataset(self, raw_data):
        """Build dataset efficiently"""
        dataset = []
        
        for item in tqdm(raw_data):
            # Filter answers
            item['answer'] = [i for i in item["answer"] if i in self.nodes_mapping.get("Disease", [{}])[0]]
            
            # Get drug
            drug = item['seed_entities'][0]
            
            # Build subgraph
            subgraph = self.build_query_subgraph_dgl(self.g_mrn, drug, self.all_paths, self.nodes_mapping, self.node_labels)
            dataset.append(subgraph)
        
        return dataset
    
    def load_paths(self, data_dir, data_name, paths_file_dir):
        """Load paths from pickle file"""
        graph_pkl = os.path.join(data_dir, "subgraphs", paths_file_dir)
        with open(graph_pkl, "rb") as f:
            all_paths = pickle.load(f)
        return all_paths
    
    def _get_node_labels(self):
        """Get node labels from mapping"""
        node_labels = {}
        for node_type, mappings in self.nodes_mapping.items():
            for node_id in mappings[0].keys():
                node_labels[node_id] = node_type
        return node_labels
    
    def _create_id_map(self, raw_data):
        """Create ID mapping for raw data"""
        id_map = {}
        for idx, item in enumerate(raw_data):
            e1, r = item["id"]  # Unpack the tuple directly
            id_map[(e1, r)] = idx  # Make sure to use a tuple as the key
        return id_map
    
    def query_paths(self, drug_identifier, all_paths): 
        """Return list of CBR paths for a given drug query"""
        list_paths_query_ = []
        
        # Handle case where drug identifier is not in all_paths
        if drug_identifier not in all_paths:
            return list_paths_query_
            
        for path in all_paths[drug_identifier][0]:
            path_query = [drug_identifier]

            for triplet in path: 
                edge, node = triplet[0], triplet[1]
                path_query.append(edge)
                path_query.append(node)

            # Form triplets more efficiently
            result_list = [(path_query[i], path_query[i+1], path_query[i+2]) for i in range(0, len(path_query)-2, 2)]
            list_paths_query_.append(result_list)

        return list_paths_query_

    def get_unique_triplets(self, list_paths):
        """
        Extract unique triplets from paths
        """
        # Use set comprehension for better performance
        return {triplet for sublist in list_paths for triplet in sublist}

    def build_query_subgraph_dgl(self, g_mrn, drug_identifier, all_paths, nodes_mapping, node_labels):
        """
        Build DGL subgraph based on CBR paths
        """
        edge_index_dict = defaultdict(list)

        # Get list of paths and unique triplets
        list_paths_query_ = self.query_paths(drug_identifier, all_paths)
        unique_triplets = self.get_unique_triplets(list_paths_query_)

        # Process triplets in batches
        for triplet in unique_triplets: 
            src_name, rel, tgt_name = triplet
            
            # Quick validation of node existence
            if (src_name not in node_labels or tgt_name not in node_labels):
                continue
                
            src_type, tgt_type = node_labels[src_name], node_labels[tgt_name]
            
            # Skip if source or target type not in nodes_mapping
            if (src_type not in nodes_mapping or tgt_type not in nodes_mapping):
                continue
                
            # Skip if source or target name not in their respective mappings
            if (src_name not in nodes_mapping[src_type][0] or 
                tgt_name not in nodes_mapping[tgt_type][0]):
                continue
                
            src_id = nodes_mapping[src_type][0][src_name]
            tgt_id = nodes_mapping[tgt_type][0][tgt_name]
            etype_query = (src_type, rel, tgt_type)
            
            # Check edge existence and get ID
            if g_mrn.has_edges_between(src_id, tgt_id, etype=etype_query):
                edge_idx = g_mrn.edge_ids(src_id, tgt_id, etype=etype_query)
                edge_index_dict[etype_query].append(edge_idx)

        # Return subgraph
        return dgl.edge_subgraph(g_mrn, edge_index_dict)
    
    
    
    def get_data(self):
        """Return all data structures"""
        return (
            self.ent2id,
            self.rel2id,
            self.id2ent,
            self.id2rel,
            self.full_adj_map,
            self.full_edge_index,
            self.full_edge_attr,
            self.train_idmap,
            self.raw_train_data,
            self.raw_test_data,
            self.raw_dev_data,
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
            self.g_mrn
        )