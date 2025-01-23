import torch 
import numpy as np
from torch import nn
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# STATE:
#   - VX
#   - VY
#   - YAW_RATE
#   - THROTTLE_FB
#   - STEERING_FB
# ACTIONS:
#   - THROTTLE_CMD
#   - STEERING_CMD
# PARAMETERS:
#   - Bf: 5.579
#     Min: 5.0
#     Max: 30.0
#   - Cf: 1.2
#     Min: 0.5
#     Max: 2.0
#   - Df: 0.192
#     Min: 0.1
#     Max: 0.9
#   - Ef: -0.083
#     Min: -2.0
#     Max: 0.0
#   - Br: 5.3852
#     Min: 5.0
#     Max: 30.0
#   - Cr: 1.2691
#     Min: 0.5
#     Max: 2.0
#   - Dr: 0.1737
#     Min: 0.1
#     Max: 0.9
#   - Er: -0.019
#     Min: -2.0
#     Max: 0.0
#   - Cm1: 0.287
#     Min: 0.1435
#     Max: 0.574
#   - Cm2: 0.0545
#     Min: 0.02725
#     Max: 0.109
#   - Cr0: 0.0518
#     Min: 0.0259
#     Max: 0.1036
#   - Cr2: 0.00035
#     Min: 1.75e-4
#     Max: 7.0e-4
#   - Iz: 27.8e-6
#     Min: 1.39e-5
#     Max: 5.56e-5
#   - Shf: -0.0013
#     Min: -0.02
#     Max: 0.02
#   - Svf: 0.00043
#     Min: -0.003
#     Max: 0.003
#   - Shr: -0.00376
#     Min: -0.02
#     Max: 0.02
#   - Svr: 0.00091
#     Min: -0.003
#     Max: 0.003
# VEHICLE_SPECS:
#   lf: 0.029
#   lr: 0.033
#   mass: 0.041
# MODEL:
#   NAME: DeepDynamics
#   HORIZON: 5
#   LAYERS:
#     - GRU:
#       OUT_FEATURES: 25
#       LAYERS: 7
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#     - DENSE:
#       OUT_FEATURES: 436
#       ACTIVATION: Mish
#   OPTIMIZATION:
#     LOSS: MSE
#     BATCH_SIZE: 32
#     NUM_EPOCHS: 400
#     OPTIMIZER: Adam
#     LR: 0.0006

horizon = 5
hidden_size=25
num_states=5
num_actions=2
param_num=16


def build_network():
    layers=[
    torch.nn.GRU(input_size=num_states+num_actions, hidden_size=horizon, num_layers=3, batch_first=True),
    torch.nn.Linear(input_size=horizon*horizon, output_size=hidden_size),
    torch.nn.Mish(),
    torch.nn.Linear(input_size=hidden_size, output_size=hidden_size),
    torch.nn.Mish()
    ]
    return layers

class dataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.X_data=torch.from_numpy(features).float().to(device)
        self.y_data=torch.from_numpy(features).float().to(device)
        self.X_norm =torch.zeros(features.shape)
        num_instances,num_time_steps,num_features=features.shape
        train_data=features.reshape((-1,num_features))
        self.scaler = StandardScaler()
        norm_train_data=self.scaler.fit_transform(train_data)
        self.X_norm=torch.from_numpy(norm_train_data.reshape((num_instances,num_time_steps,num_features))).float().to(device)
    
    def __len__(self):
        return (self.X_data.shape[0])

    def __getitem__(self, idx):
        x=self.X_data[idx]
        y=self.y_data[idx]
        x_norm=self.X_norm[idx]
        return x,y,x_norm
    
    def split(self,percent):
        split_idx=int(len(self)*percent)
        torch.manual_seed(0)
        return torch.utils.data.random_split(self,[split_idx,len(self)-split_idx])

class model(nn.Module):
    def __init__(self,eval=False):
        class GuardLayer(nn.Module):
            def __init__(self):
                super.__init__()
                self.guard_dense=torch.nn.Linear(input_size=hidden_size,output_size=param_num)
                self.guard_activation=torch.nn.Sigmoid()
                self.coefficient_ranges=torch.zeros