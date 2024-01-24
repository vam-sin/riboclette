from torch.nn import Module, Conv1d, ModuleList
import torch.nn.functional as F


class WaveNet(Module):
    def __init__(
        self,
        dilation_layers: int,
        dilation_blocks: int,
        kernel_size: int = 2,
        sample_channels: int = 1,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 512,
        output_channels=1,
        use_biases: bool = True,
        prefix_pad: bool = False,
        pad_value: int = -1,
    ):
        super().__init__()

        self.dilation_layers = dilation_layers
        self.dilation_blocks = dilation_blocks
        self.prefix_pad = prefix_pad
        self.pad_value = pad_value

        # Causal Convolution
        self.causal_conv = Conv1d(
            in_channels=sample_channels,
            out_channels=residual_channels,
            kernel_size=1,
            bias=use_biases,
        )

        self.dilations = []

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        self.residual_convs = ModuleList()
        self.skip_convs = ModuleList()

        self.receptive_field = 0

        # Dilations stack
        for _ in range(dilation_blocks):
            dilation = 1
            for _ in range(dilation_layers):
                self.dilations.append(dilation)
                self.filter_convs.append(
                    Conv1d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=kernel_size,
                        bias=use_biases,
                        dilation=dilation,
                    )
                )
                self.gate_convs.append(
                    Conv1d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=kernel_size,
                        bias=use_biases,
                        dilation=dilation,
                    )
                )
                self.residual_convs.append(
                    Conv1d(
                        in_channels=dilation_channels,
                        out_channels=residual_channels,
                        kernel_size=1,
                        bias=use_biases,
                    )
                )
                self.skip_convs.append(
                    Conv1d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=1,
                        bias=use_biases,
                    )
                )
                self.receptive_field += dilation
                dilation *= 2

        self.receptive_field *= kernel_size

        self.end_conv_1 = Conv1d(
            in_channels=skip_channels,
            out_channels=skip_channels,
            kernel_size=1,
            bias=use_biases,
        )

        self.end_conv_2 = Conv1d(
            in_channels=skip_channels,
            out_channels=output_channels,
            kernel_size=1,
            bias=use_biases,
        )

    def forward(self, x):
        x = self.causal_conv(x)

        skip = 0

        for i in range(self.dilation_blocks * self.dilation_layers):
            # Left pad input
            if self.prefix_pad:
                pad_size = (self.dilations[i], 0)
            else:
                pad_size = (0, self.dilations[i])

            x_pad = F.pad(x, pad_size, value=self.pad_value)

            # Residual block
            filter = self.filter_convs[i](x_pad)
            filter = F.tanh(filter)
            gate = self.gate_convs[i](x_pad)
            gate = F.sigmoid(gate)
            out = filter * gate

            # Residual
            x += self.residual_convs[i](out)

            # Skip connection
            skip += self.skip_convs[i](out)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x
