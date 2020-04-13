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
    
    # def ComputeEntropyAsWeight(self, inputs):
    #     entropies = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
    #     entropies = -1.0*torch.sum(entropies, dim=1)
    #     # entropy high -> confidence low -> weight low
    #     weights = 1.0 - entropies/self.max_ent
    #     return weights 
    
    # def PerLocClassLoss(self, inputs, targets):
    #     # size of input
    #     n, c, h, w = inputs.size()
    #     targets=targets.view((n, 1, 1, 1)).expand(n, 1, h, w)
    #     # compute perLocCEloss
    #     per_loc_celoss = self._per_pixel_cross_entropy(inputs, targets)
    #     #print('per_loc_celoss: {}'.format(per_loc_celoss))
    #     # compute confidence [Transferable Attention for Domain Adaptation]
    #     confidence = 1 - self._per_pixel_entropy(inputs) # high entropy means low confidence 
    #     # normalize confidence
    #     max_vals, _ = torch.max(confidence.view(n, -1), dim=1)
    #     min_vals, _ = torch.min(confidence.view(n, -1), dim=1)
    #     #print('max {}; min {}'.format(max_vals, min_vals))
    #     min_vals = min_vals.view(n, 1).expand(n, h*w)
    #     max_vals = max_vals.view(n, 1).expand(n, h*w)
    #     conf_map = (confidence.view(n, -1) - min_vals)/(max_vals-min_vals)
    #     #conf_map = confidence.view(n, -1) == max_vals # alternative
    #     conf_map = conf_map.view(n, 1, h, w)
    #     #print("conf_map: ", conf_map)
    #     conf_map_sq = conf_map #**2

    #     if torch.sum(torch.isnan(confidence)) > 0:
    #         raise('confidence contains nan')
        
    #     # loss = confidence weighted per location CE loss
    #     weighted_loss = torch.sum(conf_map_sq*per_loc_celoss)
    #     total_loss = (weighted_loss)/(n*h*w)
        
    #     """
    #     visualization module
    #     """        
    #     inputs_mask = (conf_map == 1.0)
    #     inputs_mask = inputs_mask.expand(n, 3, h, w).type(torch.FloatTensor).cuda()
    #     inputs_map = inputs*inputs_mask
    #     # final preds for each image
    #     _, final_preds = torch.max(torch.sum(torch.sum(inputs_map, dim=3), dim=2), dim=1)

    #     return total_loss, final_preds, conf_map_sq
        
    # def _per_pixel_cross_entropy(self, inputs, targets):
    #     """
    #     Compute cross entropy loss with respect to each location
    #     the input has a size of (n, c, h, wh) and the target of (n, 1, h, w)
    #     """
    #     n, c, h, w = inputs.size()
    #     if torch.sum(torch.isnan(inputs)) > 0:
    #         raise('inputs contain nan')
        
    #     # from (n, c, h, w) to (n*h*w, c)
    #     inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
    #     inputs = inputs.view(-1, c)
    #     # from (n, 1, h, w) to (n*h*w)
    #     #print('targets size: {}'.format(targets.size()))
    #     targets = targets.contiguous().view(n*h*w)
    #     # compute per-location ce_loss with reduction='none'
    #     per_loc_celoss = F.cross_entropy(inputs, targets, reduction='none')
    #     # reshape conf to target size
    #     per_loc_celoss = per_loc_celoss.view(n, h, w, 1).transpose(3, 2).transpose(2, 1).contiguous()
    #     return per_loc_celoss
        
    # def _per_pixel_entropy(self, inputs):
    #     n, c, h, w = inputs.size()
    #     #print("n, c, h, w = ", inputs.size())
    #     # from (n, c, h, w) to (n*h*w, c)
    #     inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous()
    #     inputs = inputs.view(-1, c)
    #     #print('prob: ', F.softmax(inputs, dim=1))
        
    #     per_loc_entropy = F.softmax(inputs, dim=1) * F.log_softmax(inputs, dim=1)
    #     per_loc_entropy = -1.0*torch.sum(per_loc_entropy, dim=1)
    #     #print(per_loc_entropy.size())
    #     per_loc_entropy = per_loc_entropy.view(n, h, w, 1).transpose(3, 2).transpose(2, 1).contiguous()
    #     #print(per_loc_entropy)
    #     return per_loc_entropy 