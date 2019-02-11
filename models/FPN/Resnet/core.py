# Defines our FPN backbone architecture
# The FPN is based on the Resnet 34 architecture

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        '''
          block: Signifies what type of block is used to construct the 4 layers of the Resnet
          layers: Signifies the size of each layer of Resent. There are 4 layers, hence 4 sizes.
        '''
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def init_fpn(self):
        '''
         This function modifies resnet architecture to add the feature pyramid layers.
         There are two types of varibales declared here, 
         1. 1 by 1 convs for each laayer for the lateral connections as mentioned in the FPN paper.
         2. 3 by 3 convs to be applied at each scale to reduce the antialaising from the upsampling
         
         FPN paper link: https://arxiv.org/pdf/1612.03144.pdf
        '''
        
        # For lateral connections of 4 layers (1 by 1 conv to get 256 channel dimensions)
        self.conv1x1_layer1 = conv1x1(64,256)
        self.conv1x1_layer2 = conv1x1(128,256)
        self.conv1x1_layer3 = conv1x1(256,256)
        self.conv1x1_layer4 = conv1x1(512,256)
        
        # For final 3 by 3 conv after upsampling to reduce anti aliasing
        self.conv3x3_layer1 = conv3x3(256,256)
        self.conv3x3_layer2 = conv3x3(256,256)
        self.conv3x3_layer3 = conv3x3(256,256)
        self.conv3x3_layer4 = conv3x3(256,256)
        
        

    def forward(self, x):
        '''
          x: Input image(s)          
            Output is 4 feature maps at various scales  of 1/4, 1/8, 1/16 and 1/32th the size of the original input           
        '''
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
  
        scale1 = self.layer1(x)          
        scale2 = self.layer2(scale1)      
        scale3 = self.layer3(scale2) 
        scale4 = self.layer4(scale3) 

        # Lateral connections
        final4 = self.conv1x1_layer4(scale4)

        # Merging top down and bottom up pathways through lateral connections and upsampling
        final3 = self.conv1x1_layer3(scale3)
        final3 = final3 + F.interpolate(final4, size=[ final3.size()[2], final3.size()[3] ])

        final2 = self.conv1x1_layer2(scale2)
        final2 = final2 + F.interpolate(final3, size=[ final2.size()[2], final2.size()[3] ])

        final1 = self.conv1x1_layer1(scale1)
        final1 = final1 + F.interpolate(final2, size=[ final1.size()[2], final1.size()[3] ])

        # Final 3 by 3 conv to reduce anti aliasing to get the final multi scale feature maps
        return ( self.conv3x3_layer1(final1),
                 self.conv3x3_layer2(final2),
                 self.conv3x3_layer3(final3), 
                 self.conv3x3_layer4(final4) )
          
          
          
          

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model
