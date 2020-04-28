import time
import numpy as np
import os
import tqdm
from datetime import datetime
import collections
from utils.cam_drawer import CAMDrawer,camforCount, largest_component, generate_bbox, returnCAM, returnTensorCAMs, write_text_to_img

import torch
import torch.nn.functional as F

from utils.augmentation import DataAugmentation
from utils.logger import AverageMeter
from make_video import *

import shutil


### helper functions
def theta2coordinate(theta, x, y, x_len, y_len):
    new_center_x = (1 + theta[0][0] * theta[0][2]) * 128 
    new_center_y = (1 + theta[0][1] * theta[0][2]) * 128
    new_len = 256 * theta[0][2]
    if (new_center_x - new_len/2).item() < 0:
        new_upper_left_x = 0
        border_x = (new_center_x - new_len/2).item()
        x = x - border_x
    else:
        new_upper_left_x = (new_center_x - new_len/2).item()

    if (new_center_y - new_len/2).item() < 0:
        new_upper_left_y = 0
        border_y = (new_center_y - new_len/2).item()
        y = y - border_y
    else:
        new_upper_left_y = (new_center_y - new_len/2).item()
    
    if x - new_upper_left_x < 0:
        x_len = int((x_len + (x - new_upper_left_x))/theta[0][2])
        x = 0
    else:
        x = int((x - new_upper_left_x)/theta[0][2])
        x_len = int(x_len/theta[0][2])

    if y - new_upper_left_y < 0:
        y_len = int((y_len + (y - new_upper_left_y))/theta[0][2])
        y = 0
    else:
        y = int((y - new_upper_left_y)/theta[0][2])
        y_len = int(y_len/theta[0][2])

    return x, y, x_len, y_len

def ListDatatoCPU(list_data):
    assert isinstance(list_data, list)
    cpu_list_data = []
    for d in list_data:
        cpu_list_data.append(d.cpu())
    return cpu_list_data


def ListSpecificBatchDatatoCPU(list_data, batch):
    assert isinstance(list_data, list)
    cpu_list_data = []
    for i in range(0,len(list_data)):
        # print("unsqueeze ",list_data[i][batch].unsqueeze(0))
        cpu_list_data.append(list_data[i][batch].unsqueeze(0).cpu())
        # print("cpu list data, ",cpu_list_data )
    return cpu_list_data

def compute_acc(out, target, batch_size):
    correct = (torch.max(out, dim=1)[1].view(target.size()).data == target.data).sum()
    return 100. * correct / batch_size


class RACNN_Trainer():

    loss_config_list = {'classification':0, 'apn':1, 'whole':-1}

    def __init__(self, model, optimizer, lr_scheduler, criterion, train_dataset, test_dataset, logger, config):
        
        self.loss_config = 'whole'
        
        ## 
        self.config = config
        self.logger = logger
        self.logger.info(config)

        ## get a savefolder here
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ckpt_folder = config['ckpt_folder']
        self.log_folder = config['log_folder']
        self.result_folder = config['result_folder']
      
        ## device
        if config['device'] != -1:
            cuda_id = f"cuda:{config['device']}"
            self.device = torch.device(cuda_id)
        else:
            self.device = torch.device("cpu")

        self.drawer = CAMDrawer(self.result_folder, bar=0.8)
        self.draw_cams = True

        self.model = model.to(self.device)
        if config['resume'] is not None:
            pass
            ## resume the network
            self._resume_checkpoint(config['resume'])

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        ## Hyper-parameters
        self.interleaving_step = config['interleave']
        self.margin = config['margin']
        self.max_epoch = config['max_epoch']
        self.time_consistency = config.get('temporal', False) ## default is false
        if self.time_consistency:
            self.tc_weight = config['temp_consistency_weight']

        self.save_period = config['ckpt_save_period']
        batch_size = config['batch_size']
        
        self.mini_train = config.get('mini_train', False)

        if config['disable_workers']:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            self.pretrainloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        else:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
            self.pretrainloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 

        self.augmenter = DataAugmentation()



    ## save checkpoint including model and other relevant information
    def _save_checkpoint(self, epoch, save_best=False, pretrain_ckp=False):
        arch = type(self.model).__name__
        state = {
            'model': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        if pretrain_ckp:
            filename = os.path.join(
                self.ckpt_folder,
                'pretrain-checkpoint-e{}.pth'.format(epoch)
            )
        else:
            filename = os.path.join(
                self.ckpt_folder,
                'checkpoint-e{}.pth'.format(epoch)
            )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ..\n".format(filename))
        if save_best:
            best_path = os.path.join(self.ckpt_folder, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...\n".format('model_best.pth'))


    ## resume from a checkpoint
    def _resume_checkpoint(self, resume_path, load_pretrain=False):
        self.logger.info("Loading checkpoint: {} ..\n".format(resume_path))
        checkpoint = torch.load(resume_path)
        print(checkpoint['config'])
        if not load_pretrain:
            self.start_epoch = checkpoint['epoch'] + 1
        
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if not load_pretrain:
            print("load to resume")
        else:
            print("load pretrained model")

    ## pre-train session
    def pretrain(self):
        # for _ in range(5):
        for _ in range(5):
            self.pretrain_classification()
        self._save_checkpoint(0, pretrain_ckp=True)
        for _ in range(3):
            self.pretrain_apn()
        self._save_checkpoint(1, pretrain_ckp=True)
        print("Pre-train Finished")


    ## training session
    def train(self, max_epoch, do_validation=True):
        if max_epoch is not None:
            self.max_epoch = max_epoch
        

        # self.logger.info(f"APN\n")
        # log = self.train_one_epoch(0, 1)
        # self.logger.info(log)
        # self._save_checkpoint(epoch)
        # self.test_one_epoch(0)

        for epoch in range(max_epoch):
            ## training
            self.logger.info(f"epoch {epoch}:\n")
            self.logger.info(f"Training:\n")

            self.logger.info(f"Classification\n")
            log = self.train_one_epoch(epoch, 0)
            self.logger.info(log)

            if (epoch != 0 and epoch % self.interleaving_step == 0) or epoch==0:
                self.logger.info(f"APN\n")
                log = self.train_one_epoch(epoch, 1)
                self.logger.info(log)
                self._save_checkpoint(epoch)

            ## call after the optimizer
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            ## testing
            self.logger.info(f"Validation:")
            log = self.test_one_epoch(epoch)
            self.logger.info(log)
            self.logger.info("  \n")

        print("Finished")


    ##
    def train_one_epoch(self, epoch, loss_config=-1):
        print("loss_config: ", loss_config)
        self.model.train()

        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        info_loss_meter = AverageMeter()
        temp_loss_meter = AverageMeter()
        accuracy_0 = AverageMeter()
        accuracy_1 = AverageMeter()
        accuracy_2 = AverageMeter()


        params_classifier_0 = list(self.model.classifier_0.parameters())
        params_classifier_1 = list(self.model.classifier_1.parameters())
        params_classifier_2 = list(self.model.classifier_2.parameters())
        weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
        weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())   
        weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())
        
        f_conv_0, f_conv_1, f_conv_2 = [], [], []
        def hook_feature_conv_scale_0(module, input, output):
            # print("#################in hook feature!")
            f_conv_0.clear()
            f_conv_0.append(output.data.cpu().numpy())
        def hook_feature_conv_scale_1(module, input, output):
            f_conv_1.clear()
            f_conv_1.append(output.data.cpu().numpy())
        def hook_feature_conv_scale_2(module, input, output):
            f_conv_2.clear()
            f_conv_2.append(output.data.cpu().numpy())


        h0=self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
        h1=self.model.conv_scale_1[-2].register_forward_hook(hook_feature_conv_scale_1)
        h2=self.model.conv_scale_2[-2].register_forward_hook(hook_feature_conv_scale_2)
        # print("h0, ", h0)

        
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):

            # batch_idx = 30
            data, target, idx = batch
            target = self.generate_confusion_target(target)

            if self.mini_train and batch_idx == 5: break

            ## data augmentation via imgaug
            if self.trainloader.dataset.augmentation:
                data = self.augmenter(data)
                # print(data)
            
            # data, target = data.to(self.device), target.to(self.device)
            data, target = data.to(self.device), target.to(self.device)
            B = data.shape[0]
            H, W = data.shape[-2:] # [-2:]==[-2,-1]; [-2:-1] == [-2]
            

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # data [B, C, H, W]

                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)
                print("current batch id : ",batch_idx, "idx is ", idx)
                print("target.shape", target.shape)
                print("data.shape", data.shape)
                print("current img path: ",self.trainloader.dataset.get_fname(idx))

                out_list, t_list = self.model(data, target.unsqueeze(1), loss_config) ## [B, NumClasses]
                out_0, out_1, out_2 = out_list[0], out_list[1], out_list[2]
                # t_01, t_12 = t_list[0], t_list[1]

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='none')
                cls_loss_2, preds_2 = self.criterion.ImgLvlClassLoss(out_2, target, reduction='none')
                cls_loss = cls_loss_0.sum() + cls_loss_1.sum() + cls_loss_2.sum()


                params_classifier_0 = list(self.model.classifier_0.parameters())
                params_classifier_1 = list(self.model.classifier_1.parameters())
                params_classifier_2 = list(self.model.classifier_2.parameters())
                weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
                weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
                weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())   
                weight_softmax_0_gt = weight_softmax_0[target.cpu(), :]
                weight_softmax_1_gt = weight_softmax_1[target.cpu(), :] 
                weight_softmax_2_gt = weight_softmax_2[target.cpu(), :]       

                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                probs_2 = F.softmax(out_2, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                gt_probs_2 = probs_2[list(range(B)), target]
                
                # rank_loss_1 = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                # rank_loss_2 = self.criterion.PairwiseRankingLoss(gt_probs_1, gt_probs_2, margin=self.margin)
                # rank_loss = rank_loss_1.sum() + rank_loss_2.sum()
                
                img_path = self.trainloader.dataset.get_fname(idx)
                rank_loss_1=0.0
                rank_loss_2=0.0
                for batch_inner_id, d in enumerate(weight_softmax_0_gt):
                    cpudata = ListSpecificBatchDatatoCPU(t_list[0:2],batch_inner_id)
                    count0_fullsize,count0_masked,count1_fullsize,count1_masked = self.drawer.camforCountWithZoomIn(epoch, 
                                    weight_softmax_0_gt[batch_inner_id], f_conv_0[-1][batch_inner_id], 
                                    weight_softmax_1_gt[batch_inner_id], f_conv_1[-1][batch_inner_id], 
                                    weight_softmax_2_gt[batch_inner_id], f_conv_2[-1][batch_inner_id], 
                                    img_path[batch_inner_id],2,cpudata)
                    rank_loss_1 += self.criterion.RankingLossDivideByCount(gt_probs_0, count0_fullsize, gt_probs_1, count0_masked, margin=self.margin)
                    rank_loss_2 += self.criterion.RankingLossDivideByCount(gt_probs_1, count1_fullsize, gt_probs_2, count1_masked, margin=self.margin)                
                    
                    # count0 = camforCount(weight_softmax_0_gt[batch_inner_id], f_conv_0[-1][batch_inner_id], img_path[batch_inner_id])                   
                    # count1 = camforCount(weight_softmax_1_gt[batch_inner_id], f_conv_1[-1][batch_inner_id], img_path[batch_inner_id])
                    # count2 = camforCount(weight_softmax_2_gt[batch_inner_id], f_conv_2[-1][batch_inner_id], img_path[batch_inner_id])
                    # rank_loss_1 += self.criterion.RankingLossDivideByCount(gt_probs_0, count0, gt_probs_1, count1, margin=self.margin)
                    # rank_loss_2 += self.criterion.RankingLossDivideByCount(gt_probs_1, count1, gt_probs_2, count2, margin=self.margin)                
                rank_loss = rank_loss_1.sum() + rank_loss_2.sum()
                



                B_out = out_0.shape[0] ## if temporal_coherence, B_out = B*3

                ### Temporal coherence
                if self.time_consistency:
                    temp_loss = 0.0
                    temp_loss_0 = self.criterion.TemporalConsistencyLoss(out_0[0::3], out_0[1::3], out_0[2::3], reduction='none')
                    temp_loss_1 = self.criterion.TemporalConsistencyLoss(out_1[0::3], out_1[1::3], out_1[2::3], reduction='none')
                    temp_loss_2 = self.criterion.TemporalConsistencyLoss(out_2[0::3], out_2[1::3], out_2[2::3], reduction='none')
                    temp_loss = temp_loss_0.sum() + temp_loss_1.sum() + temp_loss_2.sum()

                loss = 0.0

                if loss_config == 0: ##'classification'
                    loss += cls_loss
                elif loss_config == 1: ##'apn'
                    loss += rank_loss
                else: ##'whole'
                    loss += (rank_loss + cls_loss + info_loss)
                
                if self.time_consistency:
                    loss += self.tc_weight*temp_loss

                loss.backward()
                self.optimizer.step()

                    
                loss_meter.update(loss, 1)
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)
                if self.time_consistency:
                    temp_loss_meter.update(temp_loss, 1)
                # info_loss_meter.update(info_loss, 1)
                # calculate accuracy
                accuracy_0.update(compute_acc(out_0, target, B_out), 1)
                accuracy_1.update(compute_acc(out_1, target, B_out), 1)
                accuracy_2.update(compute_acc(out_2, target, B_out), 1)
            h0.remove()
            h1.remove()
            h2.remove()

        return {
            'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': accuracy_0.avg, 'cls_acc_1': accuracy_1.avg, 'cls_acc_2': accuracy_2.avg, 
            'rank_loss': rank_loss_meter.avg, 
            'temp_loss': temp_loss_meter.avg,
            # 'info_loss': info_loss_meter.avg,
            'total_loss': loss_meter.avg
            }

    ##

    def test_one_epoch(self, epoch):
        self.model.eval()
        
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        accuracy_0 = AverageMeter()
        accuracy_1 = AverageMeter()
        accuracy_2 = AverageMeter()


        with torch.no_grad():
            if (self.draw_cams and epoch % self.save_period == 0) or epoch == 0 or epoch == 1:
                # self._save_checkpoint(epoch)
                ## weight
                params_classifier_0 = list(self.model.classifier_0.parameters())
                params_classifier_1 = list(self.model.classifier_1.parameters())
                params_classifier_2 = list(self.model.classifier_2.parameters())
                ## -2 because we have bias in the classifier
                weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
                weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
                weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())

                # hook the feature extractor instantaneously and remove it once data is hooked
                f_conv_0, f_conv_1, f_conv_2 = [], [], []
                def hook_feature_conv_scale_0(module, input, output):
                    f_conv_0.clear()
                    f_conv_0.append(output.data.cpu().numpy())
                def hook_feature_conv_scale_1(module, input, output):
                    f_conv_1.clear()
                    f_conv_1.append(output.data.cpu().numpy())
                def hook_feature_conv_scale_2(module, input, output):
                    f_conv_2.clear()
                    f_conv_2.append(output.data.cpu().numpy())
                ## place hooker
                h0=self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
                h1=self.model.conv_scale_1[-2].register_forward_hook(hook_feature_conv_scale_1)
                h2=self.model.conv_scale_2[-2].register_forward_hook(hook_feature_conv_scale_2)
                print("saving CAMs")

            # params_classifier_0 = list(self.model.classifier_0.parameters())
            # params_classifier_1 = list(self.model.classifier_1.parameters())
            # params_classifier_2 = list(self.model.classifier_2.parameters())
            # ## -2 because we have bias in the classifier
            # weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
            # weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
            # weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())

            # # hook the feature extractor instantaneously and remove it once data is hooked
            # f_conv_0, f_conv_1, f_conv_2 = [], [], []
            # def hook_feature_conv_scale_0(module, input, output):
            #     # print("called f conv 0")
            #     f_conv_0.clear()
            #     f_conv_0.append(output.data.cpu().numpy())
            # def hook_feature_conv_scale_1(module, input, output):
            #     f_conv_1.clear()
            #     f_conv_1.append(output.data.cpu().numpy())
            # def hook_feature_conv_scale_2(module, input, output):
            #     f_conv_2.clear()
            #     f_conv_2.append(output.data.cpu().numpy())
            # ## place hooker
            # h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
            # h1 = self.model.conv_scale_1[-2].register_forward_hook(hook_feature_conv_scale_1)
            # h2 = self.model.conv_scale_2[-2].register_forward_hook(hook_feature_conv_scale_2)

        #for batch_idx, batch in tqdm.tqdm(
            #    enumerate(self.testloader), 
            #    total=len(self.testloader),
            #    desc='test'+': ', 
            #    ncols=80, 
            #    leave=False):
            for batch_idx, batch in enumerate(self.testloader):
                data, target, idx = batch
                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]
                H, W = data.shape[-2:]
                assert B == 1, "test batch size should be 1"
                # print("currently in for loop")
                # input()
                # data [B, C, H, W]
                out_list, t_list = self.model(data,target=None) ## [B, NumClasses]
                out_0, out_1, out_2 = out_list[0], out_list[1], out_list[2]
                t_01, t_12 = t_list[0], t_list[1]
                B_out = out_0.shape[0] ## if temporal_coherence, B_out = B*3

                print(f"{batch_idx}: GT: {target.item()} // theta: {t_01} // theta: {t_12}")
                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='none')
                cls_loss_2, preds_2 = self.criterion.ImgLvlClassLoss(out_2, target, reduction='none')
                cls_loss = cls_loss_0.sum() + cls_loss_1.sum() + cls_loss_2.sum()


                params_classifier_0 = list(self.model.classifier_0.parameters())
                params_classifier_1 = list(self.model.classifier_1.parameters())
                params_classifier_2 = list(self.model.classifier_2.parameters())
                weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
                weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
                weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())   
                weight_softmax_0_gt = weight_softmax_0[target, :]
                weight_softmax_1_gt = weight_softmax_1[target, :] 
                weight_softmax_2_gt = weight_softmax_2[target, :]       

                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                probs_2 = F.softmax(out_2, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                gt_probs_2 = probs_2[list(range(B)), target]
                
                # rank_loss_1 = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                # rank_loss_2 = self.criterion.PairwiseRankingLoss(gt_probs_1, gt_probs_2, margin=self.margin)
                
                
                # img_path = self.testloader.dataset.get_fname(idx)
                # count0 = camforCount(weight_softmax_0_gt, f_conv_0[-1], img_path[0])
                # count1 = camforCount(weight_softmax_1_gt, f_conv_1[-1], img_path[0])
                # count2 = camforCount(weight_softmax_2_gt, f_conv_2[-1], img_path[0])
                # print("##########count ,", count0, count1, count2)
                # rank_loss_1 = self.criterion.RankingLossDivideByCount(gt_probs_0, count0, gt_probs_1, count1, margin=self.margin)
                # rank_loss_2 = self.criterion.RankingLossDivideByCount(gt_probs_1, count1, gt_probs_2, count2, margin=self.margin)
                
                
                # img_path = self.testloader.dataset.get_fname(idx)
                # count0_fullsize,count0_masked,count1_fullsize,count1_masked = self.drawer.camforCountWithZoomIn(epoch, 
                #                 weight_softmax_0_gt , f_conv_0[-1] , 
                #                 weight_softmax_1_gt , f_conv_1[-1] , 
                #                 weight_softmax_2_gt , f_conv_2[-1] , 
                #                 img_path[0],2,ListDatatoCPU(t_list[0:2]))
                # rank_loss_1 += self.criterion.RankingLossDivideByCount(gt_probs_0, count0_fullsize, gt_probs_1, count0_masked, margin=self.margin)
                # rank_loss_2 += self.criterion.RankingLossDivideByCount(gt_probs_1, count1_fullsize, gt_probs_2, count1_masked, margin=self.margin)                
                


                # rank_loss = rank_loss_1.sum() + rank_loss_2.sum()

                ##
                # cls_loss_meter.update(cls_loss, 1)
                # rank_loss_meter.update(rank_loss, 1)
                # calculate accuracy
                accuracy_0.update(compute_acc(out_0, target, B_out), 1)
                accuracy_1.update(compute_acc(out_1, target, B_out), 1)
                accuracy_2.update(compute_acc(out_2, target, B_out), 1)

                if (self.draw_cams and epoch % self.save_period == 0)  or epoch==0 or epoch == 1:
                    img_path = self.testloader.dataset.get_fname(idx)
                    # print(img_path)
                    # print("##########################3target ", target)
                    weight_softmax_0_gt = weight_softmax_0[target, :]
                    weight_softmax_1_gt = weight_softmax_1[target, :]
                    weight_softmax_2_gt = weight_softmax_2[target, :]

                    self.drawer.draw_single_cam(
                        epoch, target, img_path[0], 
                        gt_probs_0, weight_softmax_0_gt, f_conv_0[-1], 
                        theta=None, sub_folder='scale_0')

                    self.drawer.draw_single_cam(
                        epoch, target, img_path[0], 
                        gt_probs_1, weight_softmax_1_gt, f_conv_1[-1], 
                        lvl = 1, theta=ListDatatoCPU(t_list[0:1]), sub_folder='scale_1')
                    
                    self.drawer.draw_single_cam(
                        epoch, target, img_path[0], 
                        gt_probs_2, weight_softmax_2_gt, f_conv_2[-1], 
                        lvl = 2, theta=ListDatatoCPU(t_list[0:2]), sub_folder='scale_2')

                    # input()
            # if self.draw_cams:
            if self.draw_cams and epoch % self.save_period == 0:
                timestamp = self.result_folder.split(os.sep)[-2]

                cam_path_scale_0 = os.path.join(self.result_folder, 'scale_0', f"epoch{epoch}")
                videoname_0 = f'video_epoch{epoch}_{timestamp}_scale0.avi'
                videoname_0 = os.path.join(self.result_folder, videoname_0)
                write_video_from_images(cam_path_scale_0, videoname_0)
                

                shutil.rmtree(cam_path_scale_0)
                
                cam_path_scale_1 = os.path.join(self.result_folder, 'scale_1', f"epoch{epoch}")
                videoname_1 = f'video_epoch{epoch}_{timestamp}_scale1.avi'
                videoname_1 = os.path.join(self.result_folder, videoname_1)
                write_video_from_images(cam_path_scale_1, videoname_1)
                shutil.rmtree(cam_path_scale_1)

                # draw_fig(self.log_folder)
                cam_path_scale_2 = os.path.join(self.result_folder, 'scale_2', f"epoch{epoch}")
                videoname_2 = f'video_epoch{epoch}_{timestamp}_scale2.avi'
                videoname_2 = os.path.join(self.result_folder, videoname_2)
                write_video_from_images(cam_path_scale_2, videoname_2)
                shutil.rmtree(cam_path_scale_2)

                draw_fig(self.log_folder)
                h0.remove()
                h1.remove()
                h2.remove()


        return {
            # 'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': accuracy_0.avg, 
            'cls_acc_1': accuracy_1.avg, 
            'cls_acc_2': accuracy_2.avg, 
            # 'rank_loss': rank_loss_meter.avg,
        }

    ##
    def pretrain_classification(self, max_epoch=5):
        print("pretrain Classification")

        loss_meter = AverageMeter()
        acc_meter= AverageMeter()

        self.model.train()
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='pretrain_cls'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            target = self.generate_confusion_target(target)
            data, target = data.to(self.device), target.to(self.device)
            B = data.shape[0]
            H, W = data.shape[-2:]

            if self.mini_train and batch_idx == 5: break

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)
                # data [B, C, H, W]
                if self.time_consistency:
                    
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)

                out_list, t_list = self.model(data, target=target.unsqueeze(1)) ## [B, NumClasses]
                out_0, out_1, out_2 = out_list[0], out_list[1], out_list[2]
                t_01, t_12 = t_list[0], t_list[1]
                B_out = out_0.shape[0]
                
                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss = cls_loss_0.sum()

                loss = cls_loss
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item(), 1)
                
                # calculate accuracy
                train_acc_0 = compute_acc(out_0, target, B_out)
                acc_meter.update(train_acc_0)

        print("pretrain classification loss: ", loss_meter.avg)
        print("pretrain classification acc: ", acc_meter.avg)

        return


    def pretrain_apn(self, max_epoch=3):
        print("pretran APN")

        loss_meter = AverageMeter()

        self.model.train()
        ## weight
        params_classifier_0 = list(self.model.classifier_0.parameters())
        ## -2 because we have bias in the classifier
        # weight_softmax_0 = params_classifier_0[-2].data.cpu().numpy() 
        weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())

        # hook the feature extractor instantaneously and remove it once data is hooked
        f_conv_0 = []
        def hook_feature_conv_scale_0(module, input, output):
            f_conv_0.clear()
            f_conv_0.append(output.data.detach().cpu().numpy())
        ## place hooker
        h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)

        apn_grad = []
        def hook(grad):
            print(grad)
        
        # def hook_apn_linear_grad(module, grad_in, grad_out):
        #     print("grad_in: ", grad_in)
        #     print("grad_out: ", grad_out)
        # h_apn = self.model.apn_regress_01[3].register_backward_hook(hook_apn_linear_grad)

        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='pretrain_apn'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            
            if self.mini_train and batch_idx == 5: break

            data, target, idx = batch

            target = self.generate_confusion_target(target)
            data, target = data.to(self.device), target.to(self.device)
            B = data.shape[0]
            H, W = data.shape[-2:]
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)
                # data [B, C, H, W]
                
                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)
                
                out_list, t_list = self.model(data, target.unsqueeze(1), 1) ## [B, NumClasses]
                out_0, out_1, out_2 = out_list[0], out_list[1], out_list[2]
                t_01, t_12 = t_list[0], t_list[1]
                B_out = out_0.shape[0]

                ### ---- original implementation for batchsize 1
                ### get cropped region calculated by the APN
                # out_len = 256 * t_01[0][2]
                # out_center_x = (1 + t_01[0][0]) * 128
                # out_center_y = (1 + t_01[0][1]) * 128
                # ### get CAM peak
                # coordinate, heatmap = self.get_cam_peak(f_conv_0[-1], weight_softmax_0[target,:])
                # cam = heatmap * 0.3 + img * 0.5
                # cv2.rectangle(cam, (coordinate[1], coordinate[0]), (coordinate[3], coordinate[2]), (0, 255, 0), 2)#peak activation region
                # cv2.rectangle(cam, (out_center_x - out_len/2, out_center_x - out_len/2), (out_center_x + out_len/2, out_center_x + out_len/2), (0, 0, 255), 2) # cropped region by APN
                # center_x = (coordinate[1]+coordinate[3])/2
                # cam_x = (center_x - 128)/128
                # center_y = (coordinate[0]+coordinate[2])/2
                # cam_y = (center_y - 128)/128
                # cam_l = (coordinate[3]-coordinate[1]+coordinate[2]-coordinate[0])/(2*256)
                ### ---- original implementation for batchsize 1

                out_len = 256 * t_01[:,2]
                out_center_x = (1 + t_01[:,0]) * 128
                out_center_y = (1 + t_01[:,1]) * 128
                
                ### get CAM peak
                ## coordinate is the pixel of the peak
                coordinates, heatmaps = self.get_batch_cam_peak(f_conv_0[-1], weight_softmax_0[target.cpu(),:])
                center_x = (coordinates[:,1]+coordinates[:,3])/2
                gt_center_x = (center_x - 128)/128
                center_y = (coordinates[:,0]+coordinates[:,2])/2
                gt_center_y = (center_y - 128)/128
                gt_len = (coordinates[:,3]-coordinates[:,1]+coordinates[:,2]-coordinates[:,0])/(2*256)
                target_pos = torch.FloatTensor(np.stack([gt_center_x, gt_center_y, gt_len], axis=1)).to(self.device)

                loss = F.mse_loss(t_01, target_pos, reduction='none').sum()
                # print(loss)
                loss.backward()

                self.optimizer.step()
                loss_meter.update(loss.item())

                ## draw images
                img_path = self.trainloader.dataset.get_fname(idx)
                for i in range(B):
                    # img_path = self.trainloader.dataset.get_fname([idx[j][0].item()])
                    img = cv2.imread(img_path[i], -1) ## [H, W, C]
                    cam = heatmaps[i,:] * 0.3 + img * 0.5
                    coordinate = coordinates[i,:]
                    cv2.rectangle(
                        cam, 
                        (coordinate[1], coordinate[0]), 
                        (coordinate[3], coordinate[2]), 
                        (0, 255, 0), 2)#peak activation region
                    cv2.rectangle(
                        cam, 
                        (out_center_x[i] - out_len[i]/2, out_center_y[i] - out_len[i]/2), 
                        (out_center_x[i] + out_len[i]/2, out_center_y[i] + out_len[i]/2), 
                        (0, 0, 255), 2) # cropped region by APN

                    write_text_to_img(cam, f"gt_lbl: {target[i]}", org=(20,50), fontScale = 0.3)
                    write_text_to_img(cam, f"t_01: {t_01[i,:]}", org=(20,65), fontScale = 0.3)
                    write_text_to_img(cam, f"target: {target_pos[i,:]}", org=(20,80), fontScale = 0.3)

                    path = os.path.join(self.result_folder, 'debug')
                    if not os.path.exists(path):
                        print("makepath: ", path)
                        os.mkdir(path)
                    fname = os.path.join(path, f'{int(time.time())}.jpg')
                    cv2.imwrite(fname, cam)
                
        print(loss_meter.avg)
        h0.remove()
        return 
    
    def get_cam_peak(self, feature, weight_softmax):
        # cam = returnCAM(feature.detach().cpu().numpy(), weight_softmax.detach().cpu().numpy())
        cam = returnCAM(feature, weight_softmax)
        cv2.imwrite('test.jpg', cam)
        image = cv2.resize(cam, (256, 256))
        prop = generate_bbox(image, 0.8)
        heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return prop.bbox, heatmap

    def get_batch_cam_peak(self, features, class_weights):
        CAMs = returnTensorCAMs(features, class_weights)
        # print(CAMs.shape)
        heatmap_list=[]
        prop_bbox_list=[]
        for i in range(CAMs.shape[0]):
            cam = CAMs[i].permute(1,2,0).numpy()
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cam = np.uint8(255 * cam)
            cam = cv2.resize(cam, (256, 256))
            prop_bbox_list.append(generate_bbox(cam, 0.8).bbox)
            heatmap_list.append(cv2.applyColorMap(cam, cv2.COLORMAP_JET))
        prop_bboxs = np.stack(prop_bbox_list, axis=0)
        heatmaps = np.stack(heatmap_list, axis=0)
        return prop_bboxs, heatmaps
        
    def generate_confusion_target(self, target):
        for confusion_idx in range(3, 6):
            target_cofusion = target==confusion_idx
            target_replace = torch.randn(torch.sum(target_cofusion))
            if confusion_idx == 3:
                target_replace = torch.where(target_replace>0, torch.tensor([0]), torch.tensor([2]))
                target[target_cofusion] = target_replace
            elif confusion_idx == 4:
                target_replace = torch.where(target_replace>0, torch.tensor([0]), torch.tensor([1]))
                target[target_cofusion] = target_replace
            elif confusion_idx == 5:
                target_replace = torch.where(target_replace>0, torch.tensor([1]), torch.tensor([2]))
                target[target_cofusion] = target_replace
        assert torch.sum(target==0)+torch.sum(target==1)+torch.sum(target==2) == target.numel()
        return target