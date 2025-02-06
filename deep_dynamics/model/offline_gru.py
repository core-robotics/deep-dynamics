import yaml
import torch
from torch import nn
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

string_to_torch = {
    # Layers
    "GRU" :  torch.nn.GRU,
    "DENSE" : torch.nn.Linear,
    "LSTM" : torch.nn.LSTM,
    "RNN" : torch.nn.RNN,
    # Activations
    "ReLU": torch.nn.ReLU,
    "Mish": torch.nn.Mish,
    "Softplus": torch.nn.Softplus,
    "Sigmoid": torch.nn.Sigmoid,
    # Loss Functions
    "MSE" : torch.nn.MSELoss,
    "MAE" : torch.nn.SmoothL1Loss,
    # Optimizers
    "Adam" : torch.optim.Adam,
    "NAdam" : torch.optim.NAdam,
    "AdamW" : torch.optim.AdamW
}

def build_network(param_dict):

    horizon = param_dict["MODEL"]["HORIZON"]
    num_states = len(param_dict["STATE"])
    num_actions = len(param_dict["ACTIONS"])
    layers = []
    
    for i in range(len(param_dict["MODEL"]["LAYERS"])):
        if i == 0:
            input_size = (num_states + num_actions) * horizon
        else:
            input_size = param_dict["MODEL"]["LAYERS"][i-1]["OUT_FEATURES"]
        output_size = param_dict["MODEL"]["LAYERS"][i]["OUT_FEATURES"]
        module = create_module(list(
            param_dict["MODEL"]["LAYERS"][i].keys())[0],
            input_size,
            horizon,
            output_size,
            param_dict["MODEL"]["LAYERS"][i].get("LAYERS"),
            param_dict["MODEL"]["LAYERS"][i].get("ACTIVATION")
            )
        layers += module
    return layers

def create_module(name, input_size, horizon, output_size, layers=None, activation=None, is_tcn=None):
    if layers:
        module = [string_to_torch[name](input_size // horizon, horizon, layers, batch_first=True)]
    elif activation:
        module = [string_to_torch[name](input_size, output_size), string_to_torch[activation]()] 
    else:
        module = [string_to_torch[name](input_size, output_size)]
    return module
class DeepDynamicsDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, scaler=None):
        self.X_data = torch.from_numpy(features).float().to(device)
        self.y_data = torch.from_numpy(labels).float().to(device)
        self.X_norm = torch.zeros(features.shape)
        num_instances, num_time_steps, num_features = features.shape
        train_data = features.reshape((-1, num_features))
        if scaler is None:
            self.scaler = StandardScaler()
            norm_train_data = self.scaler.fit_transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
        else:
            self.scaler = scaler
            norm_train_data = self.scaler.transform(train_data)
            self.X_norm = torch.from_numpy(norm_train_data.reshape((num_instances, num_time_steps, num_features))).float().to(device)
        
    def __len__(self):
        return(self.X_data.shape[0])
    
    def __getitem__(self, idx):
        x = self.X_data[idx]
        y = self.y_data[idx]
        x_norm = self.X_norm[idx]
        return x, y, x_norm
    
    def split(self, percent):
        split_id = int(len(self)* percent)
        torch.manual_seed(0)
        return torch.utils.data.random_split(self, [split_id, (len(self) - split_id)])
                        
class GuardLayer(nn.Module):
    def __init__(self, param_dict):
        super().__init__()
        guard_output = create_module("DENSE", 
                                        param_dict["MODEL"]["LAYERS"][-1]["OUT_FEATURES"],
                                        param_dict["MODEL"]["HORIZON"], len(param_dict["PARAMETERS"]),
                                        activation="Sigmoid")
        
        self.guard_dense = guard_output[0]
        self.guard_activation = guard_output[1]
        self.coefficient_ranges = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
        self.coefficient_mins = torch.zeros(len(param_dict["PARAMETERS"])).to(device)
        for i in range(len(param_dict["PARAMETERS"])):
            self.coefficient_ranges[i] = param_dict["PARAMETERS"][i]["Max"]- param_dict["PARAMETERS"][i]["Min"]
            self.coefficient_mins[i] = param_dict["PARAMETERS"][i]["Min"]

    def forward(self, x):
        guard_output = self.guard_dense(x)
        guard_output = self.guard_activation(guard_output) * self.coefficient_ranges + self.coefficient_mins
        return guard_output

class DeepDynamicsModel(nn.Module):
    def __init__(self, param_dict, eval=False):
        super().__init__()
        self.param_dict = param_dict
        layers = build_network(self.param_dict)
        self.batch_size = self.param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"]
        self.rnn_n_layers = self.param_dict["MODEL"]["LAYERS"][0].get("LAYERS")
        self.rnn_hiden_dim = self.param_dict["MODEL"]["HORIZON"]
        layers.insert(1, nn.Flatten())
        self.horizon = self.param_dict["MODEL"]["HORIZON"]
        layers.extend([GuardLayer(param_dict)])
        self.feed_forward = nn.ModuleList(layers)
        if eval:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]](reduction='none')
        else:
            self.loss_function = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["LOSS"]]()
        self.optimizer = string_to_torch[self.param_dict["MODEL"]["OPTIMIZATION"]["OPTIMIZER"]](self.parameters(),
                                                                                                lr=self.param_dict["MODEL"]["OPTIMIZATION"]["LR"])
        self.epochs = self.param_dict["MODEL"]["OPTIMIZATION"]["NUM_EPOCHS"]
        self.state = list(self.param_dict["STATE"])
        self.actions = list(self.param_dict["ACTIONS"])
        self.sys_params = list([*(list(p.keys())[0] for p in self.param_dict["PARAMETERS"])])
        self.vehicle_specs = self.param_dict["VEHICLE_SPECS"]
        
        
    def differential_equation(self, x, output, Ts=0.02):
        sys_param_dict, _ = self.unpack_sys_params(output)
        state_action_dict = self.unpack_state_actions(x)
        steering = state_action_dict["STEERING_FB"] + state_action_dict["STEERING_CMD"]
        throttle = state_action_dict["THROTTLE_FB"] + state_action_dict["THROTTLE_CMD"]
        alphaf = steering - torch.atan2(self.vehicle_specs["lf"]*state_action_dict["YAW_RATE"] 
                                        + state_action_dict["VY"], torch.abs(state_action_dict["VX"])) + sys_param_dict["Shf"]
        alphar = torch.atan2((self.vehicle_specs["lr"]*state_action_dict["YAW_RATE"] 
                              - state_action_dict["VY"]), torch.abs(state_action_dict["VX"])) + sys_param_dict["Shr"]
        Frx = (sys_param_dict["Cm1"]-sys_param_dict["Cm2"]*state_action_dict["VX"])*throttle - sys_param_dict["Cr0"] 
        - sys_param_dict["Cr2"]*(state_action_dict["VX"]**2)
        Ffy = sys_param_dict["Svf"] + sys_param_dict["Df"] * torch.sin(
            sys_param_dict["Cf"] * torch.atan(sys_param_dict["Bf"] * alphaf 
                                              - sys_param_dict["Ef"] * (sys_param_dict["Bf"] * alphaf - torch.atan(sys_param_dict["Bf"] * alphaf))))
        Fry = sys_param_dict["Svr"] + sys_param_dict["Dr"] * torch.sin(
            sys_param_dict["Cr"] * torch.atan(sys_param_dict["Br"] * alphar 
                                              - sys_param_dict["Er"] * (sys_param_dict["Br"] * alphar - torch.atan(sys_param_dict["Br"] * alphar))))
        dxdt = torch.zeros(len(x), 3).to(device)
        dxdt[:,0] = 1/self.vehicle_specs["mass"] * (Frx - Ffy*torch.sin(steering)) + state_action_dict["VY"]*state_action_dict["YAW_RATE"]
        dxdt[:,1] = 1/self.vehicle_specs["mass"] * (Fry + Ffy*torch.cos(steering)) - state_action_dict["VX"]*state_action_dict["YAW_RATE"]
        dxdt[:,2] = 1/sys_param_dict["Iz"] * (Ffy*self.vehicle_specs["lf"]*torch.cos(steering) - Fry*self.vehicle_specs["lr"])
        dxdt *= Ts
        return x[:,-1,:3] + dxdt

    def forward(self, x, x_norm, h0=None):
        for i in range(len(self.feed_forward)):
            if i == 0:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](x_norm, h0)
                else:
                    ff = self.feed_forward[i](torch.reshape(x_norm, (len(x), -1)))
            else:
                if isinstance(self.feed_forward[i], torch.nn.RNNBase):
                    ff, h0 = self.feed_forward[0](ff, h0)
                else:
                    ff = self.feed_forward[i](ff)

        o = self.differential_equation(x, ff)
        return o, h0, ff
    
    def unpack_sys_params(self, o):
        sys_params_dict = dict()
        for i in range(len(self.sys_params)):
            sys_params_dict[self.sys_params[i]] = o[:,i]
        ground_truth_dict =  dict()
        for p in self.param_dict["PARAMETERS"]:
            ground_truth_dict.update(p)
        return sys_params_dict, ground_truth_dict

    def unpack_state_actions(self, x):
        state_action_dict = dict()
        global_index = 0 
        for i in range(len(self.state)):
            state_action_dict[self.state[i]] = x[:,-1, global_index]
            global_index += 1
        for i in range(len(self.actions)):
            state_action_dict[self.actions[i]] = x[:,-1, global_index]
            global_index += 1
        return state_action_dict
    
    #for GRU
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)
        return hidden
    
    # #for LSTM
    # def init_hidden(self, batch_size):
    #     # Get the data type and device from the model's parameters
    #     weight = next(self.parameters()).data
        
    #     # Initialize hidden state (h0) and cell state (c0) for LSTM
    #     h0 = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)  # Hidden state
    #     c0 = weight.new(self.rnn_n_layers, batch_size, self.rnn_hiden_dim).zero_().to(device)  # Cell state
        
    #     return h0, c0

    
    def weighted_mse_loss(self, input, target, weight):
        return (weight * (input - target) ** 2)

def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def train_epoch(model, data_loader,weights):
    train_steps = 0
    train_loss_accum = 0.0
    h= model.init_hidden(model.batch_size)
    for inputs, labels, norm_inputs in data_loader:
        inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
        h = h.data
        model.zero_grad()
        output, h, _ = model(inputs, norm_inputs, h)
        loss= model.weighted_mse_loss(output, labels, weights).mean()
        train_loss_accum += loss.item()
        train_steps += 1
        loss.backward()
        model.optimizer.step()
    return train_loss_accum/train_steps
    
def val_epoch(model, data_loader, weights):
    val_steps = 0
    val_loss_accum = 0.0
    for inputs, labels, norm_inputs in data_loader:
        val_h = model.init_hidden(inputs.shape[0])
        inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
        val_h = val_h.data
        output, val_h, _ = model(inputs, norm_inputs, val_h)
        val_loss = model.weighted_mse_loss(output, labels, weights).mean()
        val_loss_accum += val_loss.item()
        val_steps += 1
    return val_loss_accum/val_steps
    

def test_epoch(model, data_loader):
    test_losses = []
    predictions = []
    ground_truth = []
    inference_times = []
    errors = []
    max_errors = [0.0, 0.0, 0.0]
    model.to(device)
    sys_params = []
    for inputs, labels, norm_inputs in data_loader:
        h = model.init_hidden(inputs.shape[0])
        h = h.data
        inputs, labels, norm_inputs = inputs.to(device), labels.to(device), norm_inputs.to(device)
        start= time.time()
        output, h, sysid = model(inputs, norm_inputs, h)
        end = time.time()
        inference_times.append(end-start)
        test_loss = model.loss_function(output.squeeze(), labels.squeeze().float())
        error = output.squeeze() - labels.squeeze().float()
        error = np.abs(error.cpu().detach().numpy())
        errors.append(error)
        for i in range(3):
            if error[i] > max_errors[i]:
                max_errors[i] = error[i]
        test_losses.append(test_loss.cpu().detach().numpy())
        predictions.append(output.squeeze())
        ground_truth.append(labels.cpu())
        sys_params.append(sysid.cpu().detach().numpy())
    
    means, _ = model.unpack_sys_params(np.mean(sys_params, axis=0))
    std_dev, _ = model.unpack_sys_params(np.std(sys_params, axis=0))
    min, _ = model.unpack_sys_params(np.min(sys_params, axis=0))
    max, _ = model.unpack_sys_params(np.max(sys_params, axis=0))
    
    coeff_names = list(means.keys())
    coeff_means = np.array(list(means.values())).flatten()  
    coeff_std = np.array(list(std_dev.values())).flatten() 
    
    coeff_dict = dict(zip(coeff_names, coeff_means))
    _, ground_truth_dict = model.unpack_sys_params(np.std(sys_params, axis=0))
    
    del ground_truth_dict["Min"]
    del ground_truth_dict["Max"]    
    

    # Prepare data for the table
    param_table_data = []
    param_table_headers = ["Parameter", "Ground Truth", "Predicted", "Percent Error"]

    for key in coeff_dict.keys():
        coeff_value = coeff_dict[key]
        gt_value = ground_truth_dict[key]
        percent_error = abs((coeff_value - gt_value) / gt_value) * 100
        param_table_data.append([key, gt_value, coeff_value, f"{percent_error:.2f}%"])

    # Print the table
    print(tabulate(param_table_data, headers=param_table_headers, tablefmt="grid"))
    
    state_table_data = []
    state_table_headers = ["State", "Mean Error", "Max Error"]
    
    for i in range(3):
        state_table_data.append([model.state[i], np.mean(np.array(errors)[:,i]), max_errors[i]])
        
    print(tabulate(state_table_data, headers=state_table_headers, tablefmt="grid"))
    
    print("RMSE: ", np.sqrt(np.mean(test_losses, axis=0)))
    print("\nMean Inference Time: ", np.mean(inference_times))
    print("\n")
    

def train(model, train_data_loader, val_data_loader, test_data_loader):
    valid_loss_min = torch.inf
    model.train()
    model.cuda()
    weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
    
    for i in range(model.epochs):
        model.train()
        train_loss=train_epoch(model, train_data_loader, weights)
        
        model.eval()
        val_loss = val_epoch(model, val_data_loader, weights)
        
        if val_loss < valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).'.format(valid_loss_min,val_loss))
            valid_loss_min = val_loss
        
        print("Epoch: {}/{}...".format(i+1, model.epochs),
            "Train Loss: {:.6f}...".format(train_loss),
            "Val Loss: {:.6f}".format(val_loss))
        
        if np.isnan(val_loss):
            break
        
        if (i+1) % 10 == 0:
            model.eval()
            test_epoch(model, test_data_loader)



if __name__ == "__main__":
    # Load dataset and configuration
    data_npz = np.load('/home/a/deep-dynamics/deep_dynamics/data/DYN-PP-ETHZMobil_5.npz')
    param_dict = yaml.load(open('/home/a/deep-dynamics/deep_dynamics/cfgs/model/deep_dynamics.yaml'), Loader=yaml.SafeLoader)
    
    features = data_npz['features'][:, :, :7]
    labels = data_npz['labels']
    
    # Initialize the model
    model = DeepDynamicsModel(param_dict, eval=False)
    print("model:"  , model)
    dataset= DeepDynamicsDataset(features, labels)
    
    train_dataset, val_dataset = dataset.split(0.8)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model.batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Train the model
    train(model, train_data_loader, val_data_loader, test_data_loader)
    

    