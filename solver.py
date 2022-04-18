import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from schedulers import ExponentialScheduler
from modules import MIEstimator, Encoder,Decoder
import torch
import torchvision.models as models
from torch.distributions import Normal, Independent
from torch.nn.functional import softplus

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
 

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # shotcut

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

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []  #
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class resnet18(nn.Module):
    def __init__(self, num_cuda, batch_size, multi_gpu=True):  # num_cuda = 3,batch_size = 9
        super(resnet18, self).__init__()
        

        # sesnet18
        self.view_00 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_01 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_02 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_03 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)

        if multi_gpu:
            self.batch_size = batch_size // num_cuda
        else:
            self.batch_size = batch_size
        self.classifier = nn.Sequential(
            nn.Linear(512, 2) 
        )
        self.logsoftmax = nn.Softmax()
        self.reduce_dim_1 = torch.nn.Conv1d(4, 1, kernel_size=1)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self, x):
        # x: (nL, 4L, 3L, 227L, 227L)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # x: (4L, nL, 3L, 227L, 227L)
        split_x = torch.split(x, 1, dim=0)

        x00 = self.view_00.forward(split_x[0].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x01 = self.view_01.forward(split_x[1].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x02 = self.view_02.forward(split_x[2].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x03 = self.view_03.forward(split_x[3].view(self.batch_size, 3, 227, 227)).unsqueeze(1)

        merge_result = torch.cat([x00, x01, x02, x03], 1)
        merge_result = self.reduce_dim_1(merge_result)
        merge_result = merge_result.view(merge_result.size(0), -1)
        merge_result = self.classifier(merge_result)

        return merge_result

class NetwithIB(nn.Module):

    def __init__(self, num_cuda, batch_size, K=256, multi_gpu=True):
        
        super(NetwithIB, self).__init__()
        self.K = K
        self.mi_estimator = MIEstimator(self.K, self.K)
        self.encoder = Encoder(self.K)
        self.decoder = Decoder(self.K)
        self.mu = nn.Parameter(torch.zeros(self.K), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones(self.K), requires_grad=False)
        self.prior = Normal(loc=self.mu, scale=self.sigma)
        self.prior = Independent(self.prior, 1)
        
        self.view_00 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_01 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_02 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        self.view_03 = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=True)
        
        if multi_gpu:
            self.batch_size = batch_size // num_cuda
        else:
            self.batch_size = batch_size
            
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 2) 
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512, 2) 
        )
        self.logsoftmax = nn.Softmax()
        self.reduce_dim_1 = torch.nn.Conv1d(4, 1, kernel_size=1)
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if 'bn' in key:
                    self.state_dict()[key][...] = 1
            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0
                
    def forward(self, x, num_sample=1):
        #if x.dim() > 2 : x = x.view(x.size(0),-1)
        
        # x: (nL, 4L, 3L, 227L, 227L)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        # x: (4L, nL, 3L, 227L, 227L)
        split_x = torch.split(x, 1, dim=0)

        x00 = self.view_00.forward(split_x[0].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x01 = self.view_01.forward(split_x[1].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x02 = self.view_02.forward(split_x[2].view(self.batch_size, 3, 227, 227)).unsqueeze(1)
        x03 = self.view_03.forward(split_x[3].view(self.batch_size, 3, 227, 227)).unsqueeze(1)   
        
        p_z1_given_m1 = self.encoder(x00)
        p_z2_given_m2 = self.encoder(x01)
        p_z3_given_m3 = self.encoder(x02)
        p_z4_given_m4 = self.encoder(x03)
        
        z1 = p_z1_given_m1.rsample() # z1: torch.Size([8, 0, 512])
        z2 = p_z2_given_m2.rsample()
        z3 = p_z3_given_m3.rsample()
        z4 = p_z4_given_m4.rsample()
                
        mi_gradient_z1_z2, mi_estimation_z1_z2 = self.mi_estimator(z1, z2)
        mi_gradient_z1_z2 = mi_gradient_z1_z2.mean() # I(z1,z2)
        mi_estimation_z1_z2 = mi_estimation_z1_z2.mean()
        
        mi_gradient_z1_z3, mi_estimation_z1_z3 = self.mi_estimator(z1, z3)
        mi_gradient_z1_z3 = mi_gradient_z1_z3.mean() # I(z1,z3)
        mi_estimation_z1_z3 = mi_estimation_z1_z3.mean()
        
        mi_gradient_z1_z4, mi_estimation_z1_z4 = self.mi_estimator(z1, z4)
        mi_gradient_z1_z4 = mi_gradient_z1_z4.mean() # I(z1,z4)
        mi_estimation_z1_z4 = mi_estimation_z1_z4.mean()
        
        mi_gradient_z2_z3, mi_estimation_z2_z3 = self.mi_estimator(z2, z3)
        mi_gradient_z2_z3 = mi_gradient_z2_z3.mean() # I(z2,z3)
        mi_estimation_z2_z3 = mi_estimation_z2_z3.mean()
        
        mi_gradient_z2_z4, mi_estimation_z2_z4 = self.mi_estimator(z2, z4)
        mi_gradient_z2_z4 = mi_gradient_z2_z4.mean() # I(z2,z4)
        mi_estimation_z2_z4 = mi_estimation_z2_z4.mean()
        
        mi_gradient_z3_z4, mi_estimation_z3_z4 = self.mi_estimator(z3, z4)
        mi_gradient_z3_z4 = mi_gradient_z3_z4.mean() # I(z3,z4)
        mi_estimation_z3_z4 = mi_estimation_z3_z4.mean()
        
        kl_1_2 = p_z1_given_m1.log_prob(z1) - p_z2_given_m2.log_prob(z1)
        kl_2_1 = p_z2_given_m2.log_prob(z2) - p_z1_given_m1.log_prob(z2)
        skl_1_2_2_1 = (kl_1_2 + kl_2_1).mean() / 2.
        
        kl_1_3 = p_z1_given_m1.log_prob(z1) - p_z3_given_m3.log_prob(z1)
        kl_3_1 = p_z3_given_m3.log_prob(z3) - p_z1_given_m1.log_prob(z3)
        skl_1_3_3_1 = (kl_1_3 + kl_3_1).mean() / 2.

        kl_1_4 = p_z1_given_m1.log_prob(z1) - p_z4_given_m4.log_prob(z1)
        kl_4_1 = p_z4_given_m4.log_prob(z4) - p_z1_given_m1.log_prob(z4)
        skl_1_4_4_1 = (kl_1_4 + kl_4_1).mean() / 2.
        
        kl_2_3 = p_z2_given_m2.log_prob(z2) - p_z3_given_m3.log_prob(z2)
        kl_3_2 = p_z3_given_m3.log_prob(z3) - p_z2_given_m2.log_prob(z3)
        skl_2_3_3_2 = (kl_2_3 + kl_3_2).mean() / 2.
        
        kl_2_4 = p_z2_given_m2.log_prob(z2) - p_z4_given_m4.log_prob(z2)
        kl_4_2 = p_z4_given_m4.log_prob(z4) - p_z2_given_m2.log_prob(z4)
        skl_2_4_4_2 = (kl_2_4 + kl_4_2).mean() / 2.
        
        kl_3_4 = p_z3_given_m3.log_prob(z3) - p_z4_given_m4.log_prob(z3)
        kl_4_3 = p_z4_given_m4.log_prob(z4) - p_z3_given_m3.log_prob(z4)
        skl_3_4_4_3 = (kl_3_4 + kl_4_3).mean() / 2.
        
        skl = ( skl_1_2_2_1 + skl_1_3_3_1 + skl_1_4_4_1 + skl_2_3_3_2 + skl_2_4_4_2 + skl_3_4_4_3 ) / 6.  
        mi_gradient = ( mi_gradient_z1_z2 + mi_gradient_z1_z3 + mi_gradient_z1_z4 + mi_gradient_z2_z3 + mi_gradient_z2_z4+ mi_gradient_z3_z4) / 6.  

        beta = 0.01
        beta1 = 0.1
        beta2 = 0.1
        beta3 = 0.1
        beta4 = 0.1
        loss_mvib = - mi_gradient + beta * skl  
        
        rate1 = p_z1_given_m1.log_prob(z1) - self.prior.log_prob(z1)
        rate2 = p_z2_given_m2.log_prob(z2) - self.prior.log_prob(z2)
        rate3 = p_z3_given_m3.log_prob(z3) - self.prior.log_prob(z3)
        rate4 = p_z4_given_m4.log_prob(z4) - self.prior.log_prob(z4)
        
        prob_x1_given_z1 = self.decoder(z1)
        prob_x2_given_z2 = self.decoder(z2)
        prob_x3_given_z3 = self.decoder(z3)
        prob_x4_given_z4 = self.decoder(z4)
        
                        
        distortion1 = -prob_x1_given_z1.log_prob(x00.view(x00.shape[0], -1))
        distortion2 = -prob_x2_given_z2.log_prob(x01.view(x01.shape[0], -1))
        distortion3 = -prob_x3_given_z3.log_prob(x02.view(x02.shape[0], -1))
        distortion4 = -prob_x4_given_z4.log_prob(x03.view(x03.shape[0], -1))
        
        rate1 = rate1.mean()
        rate2 = rate2.mean()
        rate3 = rate3.mean()
        rate4 = rate4.mean()
                          
        distortion1 = distortion1.mean()
        distortion2 = distortion2.mean()
        distortion3 = distortion3.mean()
        distortion4 = distortion4.mean()
        
        loss1 = distortion1 + beta1 * rate1
        loss2 = distortion2 + beta2 * rate2
        loss3 = distortion3 + beta3 * rate3
        loss4 = distortion4 + beta4 * rate4
                
        lossib = (loss1 + loss2 + loss3 + loss4)/ 4.
                
        merge_result_ib = torch.cat([z1, z2, z3, z4], 1)
        merge_result_ib = merge_result_ib.view(merge_result_ib.size(0), -1)
        merge_result_ib = self.classifier(merge_result_ib)
        merge_result_ib = self.logsoftmax(merge_result_ib)
        
        merge_result = torch.cat([x00, x01, x02, x03], 1)
        merge_result = self.reduce_dim_1(merge_result)
        merge_result = merge_result.view(merge_result.size(0), -1)
        merge_result = self.classifier2(merge_result)
        merge_result = self.logsoftmax(merge_result)
                
        return lossib, merge_result_ib, loss_mvib, merge_result
