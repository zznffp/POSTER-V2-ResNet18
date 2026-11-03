from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, \
    MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import torch.nn.functional as F  # PyTorch functional interface (e.g. interpolation, activation functions)
import torch  # PyTorch core library
from collections import namedtuple  # named tuple (for defining structured data)
import math  # math library (not actually used, reserved)
import pdb  # debugging tool (not actually used, reserved)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()  # call parent Module's __init__
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels,  # number of input channels
            channels // reduction,  # output channels (reduced; reduction is the reduction ratio)
            kernel_size=1,  # 1x1 conv kernel (keeps feature map size)
            padding=0,  # no padding
            bias=False)  # no bias (redundant after batch norm)
        self.relu = ReLU(inplace=True)  # ReLU activation (inplace=True saves memory)
        self.fc2 = Conv2d(
            channels // reduction,  # input channels (after reduction)
            channels,  # output channels (restore original)
            kernel_size=1,
            padding=0,
            bias=False)
        self.sigmoid = Sigmoid()  # Sigmoid activation: outputs channel weights (0~1)

    def forward(self, x):
        module_input = x  # save the original input (for later weighting)
        x = self.avg_pool(x)  # Squeeze: compress to a (1,1) feature map
        x = self.fc1(x)  # reduce dimension
        x = self.relu(x)  # activation
        x = self.fc2(x)  # restore dimension
        x = self.sigmoid(x)  # generate channel weights
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()  # call parent __init__
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),  # 1x1 conv to adjust channels and stride
                BatchNorm2d(depth)  # batch normalization
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),  # normalize first (IR structure: BN before conv)
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),  # 3x3 conv (stride 1, padding 1, keeps size)
            PReLU(depth),  # parametric ReLU (independent learnable param per channel)
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),  # 3x3 conv (stride for downsampling)
            BatchNorm2d(depth)  # batch normalization
        )
        i = 0  # temp variable (unused, reserved or debug leftover)

    def forward(self, x):
        shortcut = self.shortcut_layer(x)  # compute shortcut path output
        res = self.res_layer(x)  # compute residual path output
        return res + shortcut  # residual add: main path + shortcut path


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()  # call parent __init__
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16)  # add SE module, reduction ratio 16 (empirical)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)  # compute shortcut
        res = self.res_layer(x)  # compute residual with SE attention
        return res + shortcut  # residual add


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks1 = [
            get_block(in_channel=64, depth=64, num_units=3),
        ]
        blocks2 = [
            get_block(in_channel=64, depth=128, num_units=4),
        ]
        blocks3 = [
            get_block(in_channel=128, depth=256, num_units=14),
        ]

    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks1, blocks2, blocks3


class Backbone(Module):
    def __init__(self, num_layers, drop_ratio, mode='ir'):
        super(Backbone, self).__init__()  # call parent __init__
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'  # assert: mode must be ir or ir_se
        blocks1, blocks2, blocks3 = get_blocks(num_layers)

        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),  # 3x3 conv: in 3, out 64, stride 1, padding 1 (size unchanged)
            BatchNorm2d(64),  # batch normalization
            PReLU(64)  # parametric ReLU activation
        )

        self.output_layer = Sequential(
            BatchNorm2d(512),  # batch normalization
            Dropout(drop_ratio),  # dropout layer (prevents overfitting, ratio given by drop_ratio)
            Flatten(),  # flatten: (B, 512, 7, 7) -> (B, 512*7*7)
            Linear(512 * 7 * 7, 512),  # fully connected: 512*7*7 -> 512 dims
            BatchNorm1d(512)  # 1D batch normalization (normalize output features)
        )

        modules1 = []
        for block in blocks1:
            for bottleneck in block:
                modules1.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        modules2 = []
        for block in blocks2:
            for bottleneck in block:
                modules2.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
        modules3 = []
        for block in blocks3:
            for bottleneck in block:
                modules3.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        self.body1 = Sequential(*modules1)
        self.body2 = Sequential(*modules2)
        self.body3 = Sequential(*modules3)

    def forward(self, x):
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)

        return x1, x2, x3


def load_pretrained_weights(model, checkpoint):
    import collections  # collections library (for ordered dict)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []  # record matched / discarded weight layer names

    for i, (k, v) in enumerate(state_dict.items()):

        if k.startswith('module.'):
            k = k[7:]

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v  # save matched weight
            matched_layers.append(k)  # record matched layer name
        else:
            discarded_layers.append(k)  # record discarded layer name

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    print('load_weight', len(matched_layers))
    return model
