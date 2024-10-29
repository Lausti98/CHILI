#!/usr/bin/env python
# coding: utf-8

# %% Distance Regression with CHILI-3K using GCN model

# %% Imports

import warnings
import yaml
import json
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN, SchNet, DimeNet
from benchmark.dataset_class import CHILI
from benchmark.modules import MLP, Secondary
import itertools

### GLOBAL VARIABLES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_dicts(dict1, dict2):
    for key in dict2:
        if key in dict1:
            # Merge values if the key exists in both dictionaries
            dict1[key] = dict2[key]
    return dict1


### Training and Model Selection ###
def find_optimal_params(config, model_class, config_dict, dataset, param_grid) -> dict:
    best_val_loss = 100000000
    for values in itertools.product(*param_grid.values()):
        
        params = dict(zip(param_grid.keys(), values))
        print(f'searching parameter-set: {params}')
        
        loader_dict = load_to_dataloaders(dataset, params['batch_size'])
        config_dict['Model_config'] = merge_dicts(config_dict['Model_config'], params)
        config_dict['Train_config'] = merge_dicts(config_dict['Train_config'], params)
        model = model_class(**config_dict['Model_config']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = params['learning_rate'])
        train_loss, val_loss = train(config, model, None, optimizer, loader_dict)
        
        with open(f'parameter_tuning_{config_dict["task"]}_{config_dict["model"]}.txt', 'a+') as f:
            f.write(f'{json.dumps(params)}: train_loss={train_loss:.4f}, validation_error={val_loss:.4f}\n')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_combination = params

    print(f'Best hyperparameters: {best_combination}')
    print(f'Best validation loss: {best_val_loss:.4f}')
    
    return {'best_params': best_combination, 'val_loss': best_val_loss}


def train(config, model, secondary, optimizer, loader_dict) -> None:
    print('Started training')
    improved_function = config['improved_function']
    loss_function = config['loss_function']
    task_function = config['task_function']
    model_config = config['model_config']
    params = config['params']
    
    train_loader = loader_dict['train_loader']
    val_loader = loader_dict['val_loader']
    
    patience = 0
    best_error = None
    best_train_error = None
    for epoch in range(params['epochs']):
        
        # Patience
        if patience >= params['max_patience']:
            print("Max Patience reached, quitting...", flush=True)
            break

        # Training loop
        model.train()
        train_loss = 0
        for data in train_loader:

            # Send to device
            data = data.to(device)

            # Perform forward pass
            pred, truth = task_function(data, model, secondary, model_config['kwargs'], device, config_dict=None)
            
            loss = loss_function(pred, truth)

            # Back prop. loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        # Training loss
        train_loss = train_loss / len(train_loader)

        val_error = evaluate(config, model, secondary, val_loader)

        val_error = val_error / len(val_loader)

        if improved_function(best_error, val_error):
            best_error = val_error
            best_train_error = train_loss
            patience = 0
        else:
            patience += 1

        # Print checkpoint
        print(f'Epoch: {epoch+1}/{params["epochs"]}, Train Loss: {train_loss:.4f}, Val MSE: {val_error:.4f}')

    return best_train_error, best_error


@torch.no_grad()
def evaluate(config, model, secondary, loader) -> None:
    # Validation loop
    task_function = config['task_function']
    model_config = config['model_config']
    metric_function = config['metric_function']

    model.eval()
    error = 0
    for data in loader:
        
        # Send to device
        data = data.to(device)

        # Perform forward pass
        with torch.no_grad():
            pred, truth = task_function(data, model, secondary, model_config['kwargs'], device, config_dict=None)
            
            metric = metric_function(pred, truth)

        # Aggregate errors
        error += metric.item()
    
    return error / len(loader)


### Data loading ### 
def get_chili_dataset(root, dataset_name) -> CHILI:
    return CHILI(root, dataset_name)

def split_data(dataset, strategy='random', test_size=0.1):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      dataset.create_data_split(split_strategy=strategy, test_size=test_size)
      dataset.load_data_split(split_strategy=strategy)
    
    return dataset

def load_to_dataloaders(dataset, batch_size) -> dict[DataLoader]:
    train_loader = DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset.validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset.test_set, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader
    }


### Task definitions and configs ###
def load_config(fp):
    with open(fp, "r") as file:
      config_dict = yaml.safe_load(file)
    return config_dict

def edge_attr_regression(data, model, secondary, model_kwargs, device, config_dict):
    evaluated_kwargs = {}
    for key, value in model_kwargs.items():
        evaluated_kwargs[key] = eval(value)
    if isinstance(model, DimeNet):
        evaluated_kwargs['z'] = torch.cat((data.x, data.pos_abs), dim=1)
    else:
        evaluated_kwargs['x'] = torch.cat((data.x, data.pos_abs), dim=1)
    
    if 'edge_attr' in evaluated_kwargs.keys():
        evaluated_kwargs['edge_attr'] = None
    if 'edge_weight' in evaluated_kwargs.keys():
        evaluated_kwargs['edge_weight'] = None
    pred = model.forward(**evaluated_kwargs)
    pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
    truth = data.edge_attr
    return pred, truth

model_configs = {
    "GCN": {
        "class": 'GCN',
        "kwargs": {"x": "None", "edge_index": "data.edge_index", "edge_attr": "data.edge_attr", "edge_weight": "data.edge_attr", "batch": "data.batch"},
        "skip_training": False,
    },
    "DimeNet": {
        "class": 'DimeNet',
        "kwargs": {"z": "None", "pos": "data.pos_abs", "batch": "data.batch"},
        "skip_training": False,

    }
}
model_configuration = model_configs['GCN']
config_dict = load_config('benchmark/configs/CHILI-3K/DistanceRegression_GCN.yaml')
print(f'{config_dict=}')
model_class = eval(config_dict['model'])# model_configuration['class']
model_kwargs = model_configuration['kwargs']

config = {}
config['model_config'] = model_configuration

model = model_class(**config_dict['Model_config']).to(device=device)
secondary = Secondary(**config_dict['Secondary_config']).to(device=device)

# %% Model Setup

# Hyperparamters
learning_rate = 0.001
batch_size = 16
max_epochs = 10
seeds = 42
max_patience = 50  # Epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Dataset Module

# Create dataset
root = 'benchmark/dataset/'
dataset='CHILI-3K'
dataset = get_chili_dataset(root, dataset)# CHILI(root, dataset)

print(f'Running DistanceRegression example on {dataset}\n', flush=True)

dataset = split_data(dataset)

dataloader_dict = load_to_dataloaders(dataset, batch_size=batch_size)
# # Create random split and load that into the dataset class

print(f"Number of training samples: {len(dataset.train_set)}", flush=True)
print(f"Number of validation samples: {len(dataset.validation_set)}", flush=True)
print(f"Number of test samples: {len(dataset.test_set)}\n", flush=True)

# %% Train, validate and test

# Initialise loss function and metric function
loss_function = nn.SmoothL1Loss()
metric_function = nn.MSELoss()
improved_function = lambda best, new: new < best if best is not None else True

config['loss_function'] = nn.SmoothL1Loss()
config['metric_function'] = nn.MSELoss()
config['improved_function'] = lambda best, new: new < best if best is not None else True
config['task_function'] = edge_attr_regression
config['params'] = config_dict['Train_config']# {'max_patience': max_patience, 'max_epochs': max_epochs}
config['params']['epochs'] = 200
config['params']['max_patience'] = 20


param_grid = {
      'learning_rate': [0.0001, 0.001],# , 0.01],
      'hidden_channels': [32, 64],# , 128],
      'num_layers': [2, 4, 6],
      'batch_size': [16]#, 32, 64]
      # 
    }
# Training & Validation
tuning_res = find_optimal_params(config, model_class, config_dict, dataset, param_grid)
best_params = tuning_res[0]

loader_dict = load_to_dataloaders(dataset, best_params['batch_size'])
config_dict['Model_config'] = merge_dicts(config_dict['Model_config'], best_params)
config_dict['Train_config'] = merge_dicts(config_dict['Train_config'], best_params)
model = model_class(**config_dict['Model_config']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = best_params['learning_rate'])

# Testing loop
model.eval()
test_error = 0
test_loader = loader_dict['test_loader']
for data in test_loader:

    # Send to device
    data = data.to(device)

    # Perform forward pass
    with torch.no_grad():
        pred = model.forward( 
            x = torch.cat((data.x, data.pos_abs), dim=1),
            edge_index = data.edge_index,
            edge_attr = None,
            edge_weight = None,
            batch = data.batch
        )
        pred = torch.sum(pred[data.edge_index[0, :]] * pred[data.edge_index[1, :]], dim = -1)
        truth = data.edge_attr
        metric = metric_function(pred, truth)

    # Aggregate errors
    test_error += metric.item()

# Final test error
test_error = test_error / len(test_loader)
print(f"Test MSE: {test_error:.4f}")
