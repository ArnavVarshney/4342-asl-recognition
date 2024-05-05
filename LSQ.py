import torch
import torch.nn as nn
import math

class Quantizer(nn.Module):
    def __init__(self, bits, is_weight):
        super(Quantizer, self).__init__()
        self.bits = bits

        if is_weight:
            self.q_min = -(2 ** (bits - 1))
            self.q_max = 2 ** (bits - 1) - 1
        else:
            self.q_min = 0
            self.q_max = 2**bits - 1

        self.step_size = torch.tensor(0.0)

    def init_step_size(self, x: torch.Tensor):
        self.step_size = 2 * x.detach().abs().mean() / math.sqrt(self.q_max)

    def gradscale(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        y_out = x
        y_grad = x * scale
        y = torch.detach(y_out - y_grad) + y_grad
        return y

    def roundpass(self, x: torch.Tensor) -> torch.Tensor:
        y_out = torch.round(x)
        y_grad = x
        y = torch.detach(y_out - y_grad) + y_grad
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_size == 0:
            self.init_step_size(x)

        scale_factor = 1 / math.sqrt(len(torch.flatten(x)) * self.q_max)
        grad_scale = self.gradscale(self.step_size, scale_factor)

        x = x / grad_scale
        x = torch.clamp(x, self.q_min, self.q_max)

        return self.roundpass(x) * grad_scale


class Conv2d(nn.Conv2d):
    def __init__(self, model, weight_quantizer, activation_quantizer, bits=8):
        super(Conv2d, self).__init__(
            model.in_channels,
            model.out_channels,
            model.kernel_size,
            model.stride,
            model.padding,
            model.dilation,
            model.groups,
            model.bias is not None,
            model.padding_mode,
        )

        self.weight = nn.Parameter(model.weight.detach())
        self.bits = bits

        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

        self.weight_quantizer.init_step_size(model.weight)

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_activation = self.activation_quantizer(x)

        return nn.functional.conv2d(
            quantized_activation,
            quantized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Linear(nn.Linear):
    def __init__(self, model, weight_quantizer, activation_quantizer, bits=8):
        super(Linear, self).__init__(
            model.in_features, model.out_features, model.bias is not None
        )

        self.weight = nn.Parameter(model.weight.detach())
        self.bits = bits

        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

        self.weight_quantizer.init_step_size(model.weight)

    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_activation = self.activation_quantizer(x)

        return nn.functional.linear(quantized_activation, quantized_weight, self.bias)
    

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, model, weight_quantizer, activation_quantizer, bits=8):
        super(BatchNorm2d, self).__init__(
            model.num_features,
            model.eps,
            model.momentum,
            model.affine,
            model.track_running_stats
        )
        
        self.weight = nn.Parameter(model.weight.detach())
        self.bits = bits

        self.weight_quantizer = weight_quantizer
        self.activation_quantizer = activation_quantizer

        self.weight_quantizer.init_step_size(model.weight)
        
    def forward(self, x):
        quantized_weight = self.weight_quantizer(self.weight)
        quantized_activation = self.activation_quantizer(x)
        
        return nn.functional.batch_norm(
            quantized_activation,
            self.running_mean,
            self.running_var,
            quantized_weight,
            self.bias,
            self.training,
            self.momentum,
            self.eps,
        )