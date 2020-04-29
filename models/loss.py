from torch import nn
import torch
import torch.nn.functional as F

class TCLoss(nn.Module):
    
    def __init__(self, num_classes):
        super(TCLoss, self).__init__()
        self.num_classes = num_classes

        
        self.a = torch.zeros((1, num_classes), dtype=torch.float32)
        self.a[0, 0] = 1.0
        self.b = torch.ones((1, num_classes, ))
        self.b = self.b/torch.sum(self.b).view(1)
        
        self.max_ent = self.b*torch. log(self.b)
        self.max_ent = -1.0*torch.sum(self.max_ent, dim=1)
        print('max entropy: {}'.format(self.max_ent))
    
    """
    inputs: N-C-H-W
    targets: N-1-H-W
    """
        
    def TemporalConsistencyLoss(self, inputs, inputs_prev, inputs_next, reduction='mean'):
        t_loss_prev = nn.functional.mse_loss(inputs, inputs_prev, reduction=reduction)
        t_loss_next = nn.functional.mse_loss(inputs, inputs_next, reduction=reduction)
        #print('t_loss_prev: {}, t_loss_next: {}'.format(t_loss_prev, t_loss_next))
        return t_loss_prev + t_loss_next
    
    def ImgLvlClassLoss(self, inputs, targets, reduction='mean'):
        # 
        cls_loss = nn.functional.cross_entropy(inputs, targets, reduction=reduction)
        _, preds = torch.max(inputs, 1)
        return cls_loss, preds
    
    def ComputeEntropyAsWeight(self, inputs):
        entropies = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
        entropies = -1.0*torch.sum(entropies, dim=1)
        # entropy high -> confidence low -> weight low
        weights = 1.0 - entropies/self.max_ent
        return weights 

    def PairwiseRankingLoss(self, prob_0, prob_1, margin):
        loss = prob_0 - prob_1 + margin
        return torch.clamp(loss, min=0.0)

    def MarginLoss(self, prob, target, positive_margin, negative_margin):
        target_ = torch.FloatTensor(target.shape[0], self.num_classes)
        target_.zero_()
        target_.scatter_(1, target, 1)

        positive_loss = target_ * torch.clamp(positive_margin-probs, max=0.0)**2
        negative_loss = (1-target_) * torch.clamp(probs-negative_margin, max=0.0)**2

        return positive_loss.mean() + negative_loss.mean()
        