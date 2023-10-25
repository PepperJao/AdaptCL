'''
update weight as masked_weight
add update task mask + update DSD mask
Dynamic sparse phase and other phases
zeros_like masks
updated re-initialization
reset threshold
'''
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.step = BinaryStep.apply
        '''define masks'''
        self.mask = torch.zeros_like(self.weight)
        self.freeze_mask = torch.zeros_like(self.weight)
        self.prune_mask = torch.zeros_like(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        
    def update_mask(self, sparsity):  #to get mask, not to update weight 
        # calculate threshold
        free_weight = self.weight[(1-self.mask).type(torch.bool)] #weight without freezed tensors
        threshold = torch.quantile(torch.abs(free_weight), q = sparsity)
        # get prune mask for free parameters
        weight = torch.abs(self.weight) - threshold
        mask = self.step(weight.to(device))
        self.mask = self.step(self.mask.to(device) + mask.to(device)) #freezed part should not be pruned
        
        
    def forward(self, x):
        conv_out = torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, groups=self.groups)
        return conv_out

    
class MaskedMLP(nn.Module):
    def __init__(self, in_size, out_size):
        super(MaskedMLP, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(out_size, in_size))
        self.bias = nn.Parameter(torch.Tensor(out_size))
        self.step = BinaryStep.apply
        '''define masks'''
        self.mask = torch.zeros_like(self.weight)
        self.freeze_mask = torch.zeros_like(self.weight)
        self.prune_mask = torch.zeros_like(self.weight)
        self.reset_parameters() #Initialization of Network Weights + Bias + Threshold


    def reset_parameters(self): #Initialization of Network Weights + Bias + Threshold
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) #Uniformly distributed initialization functions
        if self.bias is not None:
            #Calculate fan_IN (number of input neurons) and FAN_OUT (number of output neurons) of the current network layer
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight) 
            bound = 1 / math.sqrt(fan_in) #square root
            nn.init.uniform_(self.bias, -bound, bound) 

    def update_mask(self, sparsity):  #to get mask, not to update weight 
        # calculate threshold
        free_weight = self.weight[(1-self.mask).type(torch.bool)] #weight without freezed tensors
        threshold = torch.quantile(torch.abs(free_weight), q = sparsity)
        # get prune mask for free parameters
        weight = torch.abs(self.weight) - threshold
        mask = self.step(weight.to(device))
        self.mask = self.step(self.mask.to(device) + mask.to(device)) #freezed part should not be pruned
        
    def forward(self, input): #threshold not 0
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
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
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
    
