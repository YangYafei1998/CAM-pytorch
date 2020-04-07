from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models

from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class TCLoss(nn.Module):
    
    def __init__(self, num_classes):
        super(TCLoss, self).__init__()
        self.num_classes = num_classes

        
        self.a = torch.zeros((1, num_classes), dtype=torch.float32)
        self.a[0, 0] = 1.0
        self.b = torch.ones((1, num_classes, ))
        self.b = self.b/torch.sum(self.b).view(1)
        
        self.max_ent = self.b*torch. log(self.b)
        self.max_ent = -1.0*torch.sum(self.max_ent.cuda(), dim=1)
        print('max entropy: {}'.format(self.max_ent))
    
    """
    inputs: N-C-H-W
    targets: N-1-H-W
    """
    # def __call__(self, inputs, targets, inputs_prev, inputs_next):
    #     temp_loss = TemporalConsistencyLoss(inputs, inputs_prev, inputs_next)
    #     cls_loss = ClassficationLoss(inputs, targets)
    #     return cls_loss + 0.1*temp_loss
        
    def TemporalConsistencyLoss(self, inputs, inputs_prev, inputs_next):

        
        t_loss_prev = nn.functional.mse_loss(inputs, inputs_prev)
        t_loss_next = nn.functional.mse_loss(inputs, inputs_next)
        #print('t_loss_prev: {}, t_loss_next: {}'.format(t_loss_prev, t_loss_next))
        return t_loss_prev + t_loss_next
    
    def ImgLvlClassLoss(self, inputs, targets):
        # 
        cls_loss = nn.functional.cross_entropy(inputs, targets)
        _, preds = torch.max(inputs, 1)
        return cls_loss, preds
    
    def ComputeEntropyAsWeight(self, inputs):
        entropies = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
        entropies = -1.0*torch.sum(entropies, dim=1)
        # entropy high -> confidence low -> weight low
        weights = 1.0 - entropies/self.max_ent
        return weights 
    
    def PerLocClassLoss(self, inputs, targets):
        # size of input
        n, c, h, w = inputs.size()
        targets=targets.view((n, 1, 1, 1)).expand(n, 1, h, w)
        # compute perLocCEloss
        per_loc_celoss = self._per_pixel_cross_entropy(inputs, targets)
        #print('per_loc_celoss: {}'.format(per_loc_celoss))
        # compute confidence [Transferable Attention for Domain Adaptation]
        confidence = 1 - self._per_pixel_entropy(inputs) # high entropy means low confidence 
        # normalize confidence
        max_vals, _ = torch.max(confidence.view(n, -1), dim=1)
        min_vals, _ = torch.min(confidence.view(n, -1), dim=1)
        #print('max {}; min {}'.format(max_vals, min_vals))
        min_vals = min_vals.view(n, 1).expand(n, h*w)
        max_vals = max_vals.view(n, 1).expand(n, h*w)
        conf_map = (confidence.view(n, -1) - min_vals)/(max_vals-min_vals)
        #conf_map = confidence.view(n, -1) == max_vals # alternative
        conf_map = conf_map.view(n, 1, h, w)
        #print("conf_map: ", conf_map)
        conf_map_sq = conf_map #**2

        if torch.sum(torch.isnan(confidence)) > 0:
            raise('confidence contains nan')
        
        # loss = confidence weighted per location CE loss
        weighted_loss = torch.sum(conf_map_sq*per_loc_celoss)
        total_loss = (weighted_loss)/(n*h*w)
        
        """
        visualization module
        """        
        inputs_mask = (conf_map == 1.0)
        inputs_mask = inputs_mask.expand(n, 3, h, w).type(torch.FloatTensor).cuda()
        inputs_map = inputs*inputs_mask
        # final preds for each image
        _, final_preds = torch.max(torch.sum(torch.sum(inputs_map, dim=3), dim=2), dim=1)

        return total_loss, final_preds, conf_map_sq
        
    def _per_pixel_cross_entropy(self, inputs, targets):
        """
        Compute cross entropy loss with respect to each location
        the input has a size of (n, c, h, wh) and the target of (n, 1, h, w)
        """
        n, c, h, w = inputs.size()
        if torch.sum(torch.isnan(inputs)) > 0:
            raise('inputs contain nan')
        
        # from (n, c, h, w) to (n*h*w, c)
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        inputs = inputs.view(-1, c)
        # from (n, 1, h, w) to (n*h*w)
        #print('targets size: {}'.format(targets.size()))
        targets = targets.contiguous().view(n*h*w)
        # compute per-location ce_loss with reduction='none'
        per_loc_celoss = F.cross_entropy(inputs, targets, reduction='none')
        # reshape conf to target size
        per_loc_celoss = per_loc_celoss.view(n, h, w, 1).transpose(3, 2).transpose(2, 1).contiguous()
        return per_loc_celoss
        
    def _per_pixel_entropy(self, inputs):
        n, c, h, w = inputs.size()
        #print("n, c, h, w = ", inputs.size())
        # from (n, c, h, w) to (n*h*w, c)
        inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
        inputs = inputs.view(-1, c)
        #print('prob: ', F.softmax(inputs, dim=1))
        
        per_loc_entropy = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
        per_loc_entropy = -1.0*torch.sum(per_loc_entropy, dim=1)
        #print(per_loc_entropy.size())
        per_loc_entropy = per_loc_entropy.view(n, h, w, 1).transpose(3, 2).transpose(2, 1).contiguous()
        #print(per_loc_entropy)
        return per_loc_entropy 

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,
                 num_classes=1000,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        # self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2(pretrained=True, progress=True, **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model