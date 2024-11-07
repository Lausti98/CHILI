
from torch_geometric.nn import GCN
import itertools


param_grid = {
      'hidden_channels': [32, 64],# , 128],
      'num_layers': [2, 4, 6],
    }


for values in itertools.product(*param_grid.values()):
    
    params = dict(zip(param_grid.keys(), values))
    #print(f'searching parameter-set: {params}')
    
  
    model = GCN(in_channels=7,
                out_channels=1,
                **params
    )
                #hidden_channels=params['hidden_channels'],
                #num_layers=params['num_layers'])

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameter set: {params} - Number of parameters: {num_params}")