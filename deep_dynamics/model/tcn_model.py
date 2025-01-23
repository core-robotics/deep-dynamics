import torch
from torch.nn.utils import weight_norm




class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
        activation=torch.nn.ReLU,
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            torch.nn.Conv1d(
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
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(
            torch.nn.Conv1d(
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
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(
            self.conv1,
            self.chomp1,
            self.activation1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.activation2,
            self.dropout2,
        )
        self.downsample = (
            torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(torch.nn.Module):
    def __init__(
        self,
        num_inputs,
        num_hidden_channels,
        kernel_size=2,
        dropout=0.2,
        activation=torch.nn.ReLU,
    ):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_hidden_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_hidden_channels[i - 1]
            out_channels = num_hidden_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    activation=activation,
                )
            ]

        # print("receptive field = ", 1 + 2 * (kernel_size - 1) * (2 ** num_levels - 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # only return last component
        return self.network(x)


class TCN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_channels,
        kernel_size,
        dropout,
        activation=torch.nn.ReLU,
    ):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(
            input_size,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=torch.nn.ReLU,
        )
        self.linear = torch.nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.tcn(x)
        x = self.linear(x[:, :, -1])
        return x

