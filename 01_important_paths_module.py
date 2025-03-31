from src.cbr_trainer.query_dataloader import DataLoader
from src.important_paths.explainer import MaskExplainer
from src.important_paths.utils import set_seed
from src.cbr_trainer.rgcn_model import HeteroRGCN

import argparse
import torch
import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Mask Explanation")
    
    # General arguments
    parser.add_argument("--data_dir", type=str, default="data/MIND/", help="Data directory")
    parser.add_argument("--data_name", type=str, default="MIND", help="Dataset name")
    parser.add_argument("--paths_file_dir", type=str, default="MIND_cbr_subgraph_knn-15_branch-1000.pkl", help="Paths file")
    parser.add_argument("--model_path", type=str, default="link_prediction_results/MIND/model/", help="Model checkpoint path")
    parser.add_argument("--model_name", type=str, default="best_model_MIND_20250330_075232.pt", help="Model name for saving")
    parser.add_argument("--output_dir", type=str, default="important_paths_results/MIND/", help="Dir to save important paths")
    
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--split", type=str, default="train", help="Split to evaluate")

    # Explainer-specific hyperparameters
    parser.add_argument("--lr_", type=float, default=0.1, help="Learning rate for explainer")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha weight for loss")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta weight for loss")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for explainer")
    parser.add_argument("--num_paths", type=int, default=5, help="Number of paths to extract")
    parser.add_argument("--max_path_length", type=int, default=4, help="Max path length")
    parser.add_argument("--degree_thr", type=int, default=10, help="Degree threshold")
    parser.add_argument("--penalty", type=float, default=1.0, help="Penalty for path loss")

    # Model configuration arguments
    parser.add_argument_group("Arguments related to model configuration")
    parser.add_argument("--emb_dim", type=int, default=128, help="Node embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Initial GCN layer dimensionality")
    parser.add_argument("--out_dim", type=int, default=128, help="Hidden GCN layer dimensionality")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Initialize data loader
    dataset_obj = DataLoader(args.data_dir, 
                             args.data_name, 
                             args.paths_file_dir)

    # Load model
    device = torch.device(args.device)
    model = HeteroRGCN(
                        dataset_obj.g_mrn,
                        args.emb_dim,
                        args.hidden_dim,
                        args.out_dim,
        
                    ).to(device)
    
    
    
    model_dir = os.path.join(args.model_path, args.model_name)
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


#     # Initialize explainer
    explainer = MaskExplainer(model, device, args, args.split)

#     # Process queries
    important_paths_query = defaultdict(list)
    if args.split == "train":
        drug_dis_train = [(item["seed_entities"][0], ans) for item in dataset_obj.raw_train_data for ans in item["answer"]]                    
    if args.split == "test":
        drug_dis_train = [(item["seed_entities"][0], ans) for item in dataset_obj.raw_test_data for ans in item["answer"]]
    
    if args.split == "dev":
        drug_dis_train = [(item["seed_entities"][0], ans) for item in dataset_obj.raw_dev_data for ans in item["answer"]]    
                        
    for drug_id, dis_id in tqdm(drug_dis_train):
        paths = explainer.explain(drug_id, dis_id, dataset_obj)
        important_paths_query[(drug_id, dis_id)].append(paths)

    # Save results
    model_name_clean = args.model_name.replace(".pt", "")
    output_file = os.path.join(args.output_dir, f"important_paths_{args.split}_{model_name_clean}.json")

    important_paths_query_json = {str(k): v for k, v in important_paths_query.items()} 
    with open(output_file, "w") as f:
        json.dump(important_paths_query_json, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()