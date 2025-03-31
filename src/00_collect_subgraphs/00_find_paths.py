#import libraries
import pandas as pd
from collections import *
import networkx as nx
import tqdm
import pickle as pkl
import random
from numpy.random import default_rng
import numpy as np
from collections import Counter
import os
import argparse
import logging

logger = logging.getLogger()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


if __name__ == '__main__':
    """
    Create file with chains around train drug queries, joining the drug query entitiy to the disease answer.
    """
    parser = argparse.ArgumentParser(description="Collect subgraphs around training entities")
    parser.add_argument("--data_name", default = "MIND", help = "Name of dataset. Will use it to create output file name")
    parser.add_argument("--data_dir_name", default = "../../data/MIND/", help = 'Path to data directory (contains train, test, dev)')     
    parser.add_argument("--output_dir", default = "../../data/MIND/subgraphs/", help = 'Path to data directory to save collected chains')   
    parser.add_argument("--cutoff", default = 3, help ="Number of hops to consider when collecting chains")
    parser.add_argument("--paths_to_collect", default =1000 , help ="Number of total paths to collect for each query")
    args = parser.parse_args()
    
    #import data
    train = pd.read_csv(os.path.join(args.data_dir_name,"train00.txt"), names= ["start_id", "type", "end_id"], sep ="\t")
    mrn = pd.read_csv(os.path.join(args.data_dir_name, "graph.txt"), names= ["start_id", "type", "end_id"], sep ="\t")
    
    #collect drug-disease pairs of trainig set
    drug_disease = defaultdict(list)
    drug_disease_pairs = []
    for start,end in zip(train["start_id"], train["end_id"]):
        drug_disease[start].append(end)
        drug_disease_pairs.append((start,end))
    drugs_list = drug_disease.keys()#diseases
    
    #Build graph using networkx
    logger.info("=====Loading KG=====")
    graph_mrn = nx.DiGraph()
    for source,edge,target in zip(mrn["start_id"], mrn["type"], mrn["end_id"]): 
            graph_mrn.add_edge(source,target, weight =edge)
    edge_labels = nx.get_edge_attributes(graph_mrn,'weight') #set edge labels
    
    #collect subgraphs
    all_metapath_edges = defaultdict(list)

    for drug in tqdm.tqdm(drug_disease):
        number_paths_to_collect_for_disease = args.paths_to_collect / len(drug_disease[drug])

        for dis in drug_disease[drug]:
            count = 0
            collected_paths = []

            for mp in nx.all_simple_paths(graph_mrn, drug, dis, cutoff=args.cutoff):
                path = []

                if count >= number_paths_to_collect_for_disease:
                    path.append(dis)
                    count = 0
                    break

                for n in range(len(mp) - 1):
                    node = mp[n]
                    edge = edge_labels.get((mp[n], mp[n + 1]))

                    if node != drug:
                        path.extend([node, edge])
                    else:
                        path.append(edge)

                path.append(dis)
                count += 1

                # Generate triples
                triples = [(path[p], path[p + 1]) for p in range(0, len(path), 2) if len(path) >= 2]

                all_metapath_edges[drug].append(triples)
           
    #save paths 
    pkl_dir = os.path.join(args.output_dir, f'{args.data_name}_chains_collection_{args.cutoff}hops_{args.paths_to_collect}_paths.pkl')
    
    handle = open(pkl_dir, "wb")
    pkl.dump(all_metapath_edges, handle)
    handle.close()