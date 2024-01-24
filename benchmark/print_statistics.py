# Import libraries
import argparse
import torch
from Code.datasetClass import InOrgMatDatasets

def main(args):
    
    # Create dataset
    dataset = InOrgMatDatasets(dataset=args['dataset_name'], root=args['dataset_root'])
    try:
        dataset.load_data_split(split_strategy=args['split_strategy'], 
                                stratify_on=args['stratify_on'], 
                                stratify_distribution=args['stratify_distribution'])
    except FileNotFoundError:
        dataset.create_data_split(split_strategy=args['split_strategy'], 
                                stratify_on=args['stratify_on'], 
                                stratify_distribution=args['stratify_distribution'])
        dataset.load_data_split(split_strategy=args['split_strategy'], 
                                stratify_on=args['stratify_on'], 
                                stratify_distribution=args['stratify_distribution'])

    df = dataset.get_statistics(return_dataframe=True)

    selected_columns = ['# of nodes', '# of edges', '# of elements', 'NP size (Å)']
    print(df[selected_columns].describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--dataset_name', required=True, type=str)
    parser.add_argument('--split_strategy', default='stratified', type=str)
    parser.add_argument('--stratify_on', default='Crystal system (Number)', type=str)
    parser.add_argument('--stratify_distribution', default='equal', type=str)
    args = vars(parser.parse_args())

    main(args)
    
