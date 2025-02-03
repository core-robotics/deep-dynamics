import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        activation=nn.Mish,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(
                nn.Conv1d(
                    n_inputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
        self.chomp1 = Chomp1d(padding)
        self.activation1 = activation()
        
        self.conv2 = nn.utils.weight_norm(
                nn.Conv1d(
                    n_outputs,
                    n_outputs,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
        self.chomp2 = Chomp1d(padding)
        self.activation2 = activation()
        
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation1,
            self.conv2,
            self.chomp2,
            self.activation2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.final_activation = activation()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_activation(out + res)

class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_layers,
        kernel_size=2,
        activation=nn.Mish,
    ):
        super(TemporalConvNet, self).__init__()
        layers=[]
        for i in range(num_layers):
            dilation_size = 2 ** i
            num_input = num_inputs if i == 0 else num_outputs
            num_output = num_outputs
            layers += [
                TemporalBlock(
                    num_input,
                    num_output,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    activation=activation,
                )
            ]
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)