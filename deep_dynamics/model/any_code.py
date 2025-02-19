# import torch
# print("PyTorch CUDA Available:", torch.cuda.is_available())
# print("CUDA Version:", torch.version.cuda)
# print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
# print("GPU Count:", torch.cuda.device_count())
# print("Current Device:", torch.cuda.current_device())
# print("CUDA Capability:", torch.cuda.get_device_capability(0))
# print("CUDA Memory Allocated:", torch.cuda.memory_allocated())
# print("CUDA Memory Cached:", torch.cuda.memory_reserved())

import torch
import yaml
import offline_tcn
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    gru_model_name="GRU"
    tcn_model_name="TCN"
    horizon = ["5","50","500"]
    
    data_npz = np.load('/home/a/deep-dynamics/deep_dynamics/data/DYN-PP-ETHZMobil_' + horizon[0] + '.npz')
        
    # data_npz = np.load('/home/a/deep-dynamics/deep_dynamics/data/DYN-PP-ETHZMobil_' + horizon + '.npz')
   
    
    features = data_npz['features'][:, :, :7]
    labels = data_npz['labels']
    
    param_dict_gru = yaml.load(open('/home/a/deep-dynamics/deep_dynamics/cfgs/model/deep_dynamics_gru.yaml'), Loader=yaml.SafeLoader)
    param_dict_tcn = yaml.load(open('/home/a/deep-dynamics/deep_dynamics/cfgs/model/deep_dynamics_tcn.yaml'), Loader=yaml.SafeLoader)
    
    model_gru = offline_tcn.DeepDynamicsModel(param_dict_gru, eval=False)
    model_tcn = offline_tcn.DeepDynamicsModel(param_dict_tcn, eval=False)
    
    dataset = offline_tcn.DeepDynamicsDataset(features, labels)
    train_dataset, val_dataset = dataset.split(0.8)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model_gru.batch_size, shuffle=True, drop_last=True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=model_gru.batch_size, shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    
    model_gru.load_state_dict(torch.load('/home/a/deep-dynamics/deep_dynamics/output/GRU_5h/996_epoch.pth'))
    model_gru.to(device)
    model_gru.eval()
    offline_tcn.test_epoch(model_gru, test_data_loader)
    summary(model_gru, input_size=(model_gru.batch_size, model_gru.horizon, 7))
    
    model_tcn.load_state_dict(torch.load('/home/a/deep-dynamics/deep_dynamics/output/TCN_5h/999_epoch.pth'))
    model_tcn.to(device)
    model_tcn.eval()
    offline_tcn.test_epoch(model_tcn, test_data_loader)
    summary(model_tcn, input_size=(model_tcn.batch_size, model_tcn.horizon, 7))
    
    #loss plot
    gru_loss = np.load('/home/a/deep-dynamics/deep_dynamics/output/GRU_5h/GRU_5_losses.npz')
    tcn_loss = np.load('/home/a/deep-dynamics/deep_dynamics/output/TCN_5h/TCN_5_losses.npz')
    
    # plt.plot(gru_loss['train_loss'], label='GRU Train Loss')
    plt.plot(gru_loss['val_loss'], label='GRU Validation Loss')
    # plt.plot(tcn_loss['train_loss'], label='TCN Train Loss')
    plt.plot(tcn_loss['val_loss'], label='TCN Validation Loss')
    plt.ylim(0, 0.1)
    plt.xlim(1, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    