'''
update weight as masked_weight
reset threshold
threshold ---> size 1
'''
#version12
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math

class BinaryStep(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input>0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2-4*torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input*additional

class MaskedConv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_c 
        self.out_channels = out_c
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.dilation = dilation
        self.groups = groups

        ## define weight 
        self.weight = nn.Parameter(torch.Tensor(
            out_c, in_c // groups, *kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.threshold = nn.Parameter(torch.Tensor(1))
        self.step = BinaryStep.apply
        '''define a mask'''
        self.mask = torch.zeros(out_c, in_c // groups, *kernel_size)
        self.task_mask = torch.zeros(out_c, in_c // groups, *kernel_size)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            self.threshold.data.fill_(0.)

    def reset_threshold(self): #Initialization of Network Weights + Bias + Threshold
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0.) #initialize threshold as 0
            
    def forward(self, x):
        weight_shape = self.weight.shape 
#         threshold = self.threshold.view(weight_shape[0], -1)
        threshold = self.threshold
        weight = torch.abs(self.weight)
        weight = weight.view(weight_shape[0], -1)
        weight = weight - threshold
        mask = self.step(weight)
        mask = mask.view(weight_shape)
        ratio = torch.sum(mask) / mask.numel()
        # print("threshold {:3f}".format(self.threshold[0]))
        # print("keep ratio {:.2f}".format(ratio))
        if ratio <= 0.01:
            with torch.no_grad():
                self.threshold.data.fill_(0.)
#             threshold = self.threshold.view(weight_shape[0], -1)
            threshold = self.threshold
            weight = torch.abs(self.weight)
            weight = weight.view(weight_shape[0], -1)
            weight = weight - threshold
            mask = self.step(weight)
            mask = mask.view(weight_shape)
        self.weight.data = self.weight * mask
        '''update the mask'''
        self.mask = mask
        conv_out = torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return conv_out

    
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.threshold = nn.Parameter(torch.Tensor(1))
        self.step = BinaryStep.apply
        '''define a mask'''
        self.mask = torch.zeros(out_size, in_size)
        self.task_mask = torch.zeros(out_size, in_size)
        self.reset_parameters() #Initialization of Network Weights + Bias + Threshold


    def reset_parameters(self): #Initialization of Network Weights + Bias + Threshold
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #Uniformly distributed initialization functions
        if self.bias is not None:
            #Calculate fan_IN (number of input neurons) and FAN_OUT (number of output neurons) of the current network layer
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight) 
            bound = 1 / math.sqrt(fan_in) #square root
            nn.init.uniform_(self.bias, -bound, bound) 
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0.) #initialize threshold as 0
            
    def reset_threshold(self): #Initialization of Network Weights + Bias + Threshold
        with torch.no_grad():
            #std = self.weight.std()
            self.threshold.data.fill_(0.) #initialize threshold as 0
    
    def forward(self, input): #threshold not 0
        abs_weight = torch.abs(self.weight) #out absolute value of tensor
#         threshold = self.threshold.view(abs_weight.shape[0], -1)
        threshold = self.threshold
        abs_weight = abs_weight-threshold
        mask = self.step(abs_weight) #binary mask?
        ratio = torch.sum(mask) / mask.numel() #sum/number of elements
        #print("keep ratio {:.2f}".format(ratio))
        #to calculate ratio? or mask?
        if ratio <= 0.01:
            with torch.no_grad():
                #std = self.weight.std()
                self.threshold.data.fill_(0.)
            abs_weight = torch.abs(self.weight)
#             threshold = self.threshold.view(abs_weight.shape[0], -1)
            threshold = self.threshold
            abs_weight = abs_weight-threshold
            mask = self.step(abs_weight)
        self.weight.data = self.weight * mask 
        '''update the mask'''
        self.mask = mask
        output = torch.nn.functional.linear(input, self.weight, self.bias)
        return output

    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class MaskedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MaskedBasicBlock, self).__init__()
        self.conv1 = MaskedConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = MaskedConv2d(planes, planes, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class ResNet(nn.Module):
    def __init__(self, channels, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4) 
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
# class ResNet(nn.Module):
#     def __init__(self, channels, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear = nn.Linear(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
    
    
# class MaskedResNet(nn.Module):
#     def __init__(self, channels, block, num_blocks, num_classes=10):
#         super(MaskedResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = MaskedConv2d(channels, 64, kernel_size=(3, 3),
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = MaskedMLP(512*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out
    
class MaskedResNet(nn.Module):
    def __init__(self, channels, block, num_blocks, num_classes=10):
        super(MaskedResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = MaskedConv2d(channels, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = MaskedMLP(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    
class MaskedResNet_grown(nn.Module): 
    #version1 and version2
    #duplicate outstanding 1 conv+ 1 mlp layers from the start
    def __init__(self, channels, block, num_blocks, num_classes=10):
        super(MaskedResNet_grown, self).__init__()
        self.in_planes = 64

        self.conv1a = MaskedConv2d(channels, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        #and here!!!
        self.conv1b = MaskedConv2d(channels, 64, kernel_size=(3, 3),
                               stride=1, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        #and here!!!
        self.linear1a = MaskedMLP(512*block.expansion, num_classes)
        self.linear1b = MaskedMLP(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #work here!!!
        out = self.conv1a(x)+self.conv1b(x)
        out = F.relu(self.bn1(out))
#         out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear1a(out)+self.linear1b(out)
        return out
    
class MaskedBasicBlock_grown(nn.Module): #for version2 duplicate all conv_layers and fully connected layers from the start
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(MaskedBasicBlock_grown, self).__init__()
        #here!!!
        self.conv1a = MaskedConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.conv1b = MaskedConv2d(in_planes, planes, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        
        self.bn1 = nn.BatchNorm2d(planes)
        #and here!!!
        self.conv2a = MaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.conv2b = MaskedConv2d(planes, planes, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MaskedConv2d(in_planes, self.expansion*planes,
                          kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1a(x) + self.conv1b(x)
        out = F.relu(self.bn1(out))
        out = self.conv2a(out) + self.conv2b(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
    
class LeNet5(nn.Module):
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.num_classes)  
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 16*5*5)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
    
class MaskedLeNet5(nn.Module):
    
    def __init__(self, num_classes=10):
        super(MaskedLeNet5, self).__init__()

        self.num_classes = num_classes

        self.conv1 = MaskedConv2d(1, 6, kernel_size=(5, 5))
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = MaskedConv2d(6, 16, kernel_size=(5, 5))
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = MaskedMLP(16*5*5, 120)
        self.fc2 = MaskedMLP(120, 84)
        self.fc3 = MaskedMLP(84, self.num_classes)  
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 16*5*5)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
    


defaultcfg = {
    11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}

class masked_vgg(nn.Module):
    def __init__(self, num_classes=10, depth=19, init_weights=True, cfg=None):
        super(masked_vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)
        self.classifier = nn.Sequential(
              MaskedMLP(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              MaskedMLP(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=(3, 3), padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                m.reset_parameters()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, MaskedMLP):
                m.reset_parameters()
                

class vgg(nn.Module):
    def __init__(self, num_classes=10, depth=19, init_weights=True, cfg=None):
        super(vgg, self).__init__()
        if cfg is None:
            cfg = defaultcfg[depth]

        self.cfg = cfg

        self.feature = self.make_layers(cfg, True)

        self.classifier = nn.Sequential(
              nn.Linear(cfg[-1], 512),
              nn.BatchNorm1d(512),
              nn.ReLU(inplace=True),
              nn.Linear(512, num_classes)
            )
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()