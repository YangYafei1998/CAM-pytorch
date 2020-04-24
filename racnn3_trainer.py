import time
import numpy as np
import os
import tqdm
from datetime import datetime
import collections
from utils.cam_drawer import CAMDrawer

import torch
import torch.nn.functional as F

from utils.augmentation import DataAugmentation
from utils.logger import AverageMeter
from make_video import *

import shutil

# from trainer import Trainer
def ListDatatoCPU(list_data):
    assert isinstance(list_data, list)
    cpu_list_data = []
    for d in list_data:
        cpu_list_data.append(d.cpu())
    return cpu_list_data

class RACNN3_Trainer():

    loss_config_list = {'classification':0, 'apn':1, 'whole':-1}

    def __init__(self, model, optimizer, lr_scheduler, criterion, train_dataset, test_dataset, logger, config):
        
        self.loss_config = 'whole'
        self.lvls = model.lvls 

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
        
        if config['disable_workers']:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        else:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 

        self.augmenter = DataAugmentation()



    ## save checkpoint including model and other relevant information
    def _save_checkpoint(self, epoch, save_best=False):
        arch = type(self.model).__name__
        state = {
            'model': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }
        filename = os.path.join(
            self.ckpt_folder,
            'checkpoint-epoch{}.pth'.format(epoch)
        )
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.ckpt_folder, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: {} ...".format('model_best.pth'))


    ## resume from a checkpoint
    def _resume_checkpoint(self, resume_path, load_pretrain=False):
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        if not load_pretrain:
            self.start_epoch = checkpoint['epoch'] + 1
        
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['model'] != self.config['model']:
            msg = ("Warning: Architecture configuration given in config file is"
                   " different from that of checkpoint."
                   " This may yield an exception while state_dict is being loaded.")
            self.logger.warning(msg)
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        if not load_pretrain:
            # uncomment this line if you want to use the resumed optimizer
            # load optimizer state from checkpoint only when optimizer type is not changed.
            ckpt_opt_type = checkpoint['config']['optimizer']['type']
            if ckpt_opt_type != self.config['optimizer']['type']:
                msg = ("Warning: Optimizer type given in config file is different from"
                    "that of checkpoint.  Optimizer parameters not being resumed.")
                self.logger.warning(msg)
            else:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

        # self.logger = checkpoint['logger']
        msg = "Checkpoint '{}' (epoch {}) loaded"
        self.logger.info(msg .format(resume_path, self.start_epoch))
        if not load_pretrain:
            print("load to resume")
        else:
            print("load pretrained model")

    ## pre-train session
    def pretrain(self):
        # self.pretrain_classification()
        self.pretrain_apn()
        print("Pre-train Finished")


    ## training session
    def train(self, max_epoch, do_validation=True):
        if max_epoch is not None:
            self.max_epoch = max_epoch

        # self.test_one_epoch(0)

        for epoch in range(max_epoch):
            ## training
            self.logger.info(f"epoch {epoch}:\n")
            self.logger.info(f"Training:\n")

            self.logger.info(f"Classification\n")
            log = self.train_one_epoch(epoch, 0)
            self.logger.info(log)

            if epoch % self.interleaving_step == 0:
                self.logger.info(f"APN\n")
                log = self.train_one_epoch(epoch, 1)
                self.logger.info(log)
                
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
        acc_list = []
        for _ in range(self.lvls):
            acc_list.append(AverageMeter())
        
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            target = self.generate_confusion_target(target)

            ## data augmentation via imgaug
            if self.trainloader.dataset.augmentation:
                data = self.augmenter(data)
                # print(data)
            
            ## 
            data, target = data.to(self.device), target.to(self.device)

            B = data.shape[0]
            H, W = data.shape[-2:] # [-2:]==[-2,-1]; [-2:-1] == [-2]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # data [B, C, H, W]
                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)

                out_list, t_list = self.model(data, target.unsqueeze(1), loss_config) ## [B, NumClasses]
                
                ### Classification loss
                cls_loss = 0.0
                temp_loss = 0.0
                for lvl in range(len(out_list)):
                    cls_loss += self.criterion.ImgLvlClassLoss(out_list[lvl], target, reduction='none')[0].sum()
                    
                    ### Temporal coherence
                    if self.time_consistency:
                        # conf_0 = self.criterion.ComputeEntropyAsWeight(out_0).view(B, 3)
                        # conf_1 = self.criterion.ComputeEntropyAsWeight(out_1).view(B, 3)
                        ## [0::3] staring from 0, get every another three elements
                        temp_loss += self.criterion.TemporalConsistencyLoss(
                            out_list[lvl][0::3], out_list[lvl][1::3], out_list[lvl][2::3], reduction='none').sum()
                        # temp_loss += (temp_loss_0*(conf_0**2) + (1-conf_0)**2).sum()
                        # temp_loss += (temp_loss_1*(conf_1**2) + (1-conf_1)**2).sum()


                ### Ranking loss
                B_out = data.shape[0] ## if temporal_coherence, B_out = B*3
                rank_loss = 0.0
                for lvl in range(len(out_list)-1):
                    probs_0 = F.softmax(out_list[lvl], dim=-1)
                    probs_1 = F.softmax(out_list[lvl+1], dim=-1)
                    gt_probs_0 = probs_0[list(range(B_out)), target]
                    gt_probs_1 = probs_1[list(range(B_out)), target] 
                    rank_loss += self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin).sum()

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

                # calculate accuracy
                # print(out_0.shape)
                # print(target.shape)
                for lvl in range(len(out_list)):
                    correct = (torch.max(out_list[lvl], 1)[1].view(target.size()).data == target.data).sum()
                    train_acc = 100. * correct / B_out
                    acc_list[lvl].update(train_acc, 1)
                
                loss_meter.update(loss, 1)
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)
                if self.time_consistency:
                    temp_loss_meter.update(temp_loss, 1)
                # info_loss_meter.update(info_loss, 1)
                

        return {
            'cls_loss': cls_loss_meter.avg, 
            'rank_loss': rank_loss_meter.avg, 
            'temp_loss': temp_loss_meter.avg,
            # 'info_loss': info_loss_meter.avg,
            'cls_acc_0': acc_list[0].avg, 
            'cls_acc_1': acc_list[1].avg, 
            'cls_acc_2': acc_list[2].avg, 
            'total_loss': loss_meter.avg
            }

    ##
    def test_one_epoch(self, epoch):
        self.model.eval()
        
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        acc_list = []
        for lvl in range(self.lvls):
            acc_list.append(AverageMeter())

        with torch.no_grad():
            if self.draw_cams and epoch % self.save_period == 0:
                ## weight
                weight_softmax_list=[]
                for lvl in range(self.lvls):
                    params_classifier = list(self.model.clsfierList[lvl].parameters())
                    weight_softmax_list.append(np.squeeze(params_classifier[-2].data.cpu().numpy()))

                # hook the feature extractor instantaneously and remove it once data is hooked
                # f_conv_0, f_conv_1, f_conv_2 = [], [], []
                f_conv_list = []
                def hook_feature_conv_scale_0(module, input, output):
                    # f_conv_0.clear()
                    # f_conv_0.append(output.data.cpu().numpy())
                    f_conv_list.clear()
                    f_conv_list.append(output.data.cpu().numpy())

                def hook_feature_conv_scale_1(module, input, output):
                    # f_conv_1.clear()
                    # f_conv_1.append(output.data.cpu().numpy())
                    assert 1==len(f_conv_list)
                    f_conv_list.append(output.data.cpu().numpy())

                def hook_feature_conv_scale_2(module, input, output):
                    # f_conv_2.clear()
                    # f_conv_2.append(output.data.cpu().numpy())
                    assert 2==len(f_conv_list)
                    f_conv_list.append(output.data.cpu().numpy())

                ## place hooker
                h0 = self.model.convList[0][-2].register_forward_hook(hook_feature_conv_scale_0)
                h1 = self.model.convList[1][-2].register_forward_hook(hook_feature_conv_scale_1)
                h2 = self.model.convList[2][-2].register_forward_hook(hook_feature_conv_scale_2)
                print("saving CAMs")

            
            #for batch_idx, batch in tqdm.tqdm(
            #    enumerate(self.testloader), 
            #    total=len(self.testloader),
            #    desc='test'+': ', 
            #    ncols=80, 
            #    leave=False):
            for batch_idx, batch in enumerate(self.testloader):
                data, target, idx = batch
                # target = self.generate_confusion_target(target)

                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]
                assert B == 1, "test batch size should be 1"

                # data [B, C, H, W]
                out_list, t_list = self.model(data,target=None) ## [B, NumClasses]
                print(f"{batch_idx}: GT: {target.item()} // theta_01: {t_list[0].data} // theta_12: {t_list[1].data}")

                ### Classification loss
                cls_loss = 0.0
                gt_probs_list = []
                for lvl in range(self.lvls):
                    cls_loss += self.criterion.ImgLvlClassLoss(out_list[lvl], target, reduction='none')[0].sum()
                    probs = F.softmax(out_list[lvl], dim=-1)
                    gt_probs_list.append(probs[list(range(B)), target])


                ### Ranking loss
                rank_loss = 0.0
                for lvl in range(self.lvls-1):
                    rank_loss += self.criterion.PairwiseRankingLoss(gt_probs_list[lvl], gt_probs_list[lvl+1], margin=self.margin).sum()

                # calculate accuracy
                for lvl in range(self.lvls):
                    correct = (torch.max(out_list[lvl], dim=1)[1].view(target.size()).data == target.data).sum()
                    train_acc = 100. * correct / self.testloader.batch_size
                    acc_list[lvl].update(train_acc, 1)
                
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)

                if self.draw_cams and epoch % self.save_period == 0:
                    img_path = self.testloader.dataset.get_fname(idx)
                    self.drawer.draw_cam_at_different_scales(
                        epoch, target, img_path[0], 
                        gt_probs_list, weight_softmax_list, f_conv_list, ListDatatoCPU(t_list))

                    # # print(img_path)
                    # lvl = 0
                    # weight_softmax_gt = weight_softmax_list[lvl][target, :]
                    # self.drawer.draw_cam(
                    #     epoch, gt_probs_list[lvl], target, 
                    #     weight_softmax_gt, f_conv_0[-1], 
                    #     img_path[0], theta=None, sub_folder=f'scale_{lvl}')
                
                    # for lvl in range(1,self.lvls):
                    #     weight_softmax_gt = weight_softmax_list[lvl][target, :]
                    #     if lvl == 1: 
                    #         f_conv = f_conv_1
                    #     elif lvl == 2:
                    #         f_conv = f_conv_2
                    #     else:
                    #         raise NotImplementedError

                    #     self.drawer.draw_cam(
                    #         epoch, gt_probs_list[lvl], target, 
                    #         weight_softmax_gt, f_conv[-1], 
                    #         img_path[0], lvl=lvl, theta=ListDatatoCPU(t_list), sub_folder=f'scale_{lvl}')
                    

            if self.draw_cams and epoch % self.save_period == 0:
                timestamp = self.result_folder.split(os.sep)[-2]

                for lvl in range(self.lvls):
                    cam_path_scale = os.path.join(self.result_folder, f'scale_{lvl}', f"epoch{epoch}")
                    videoname = f'video_epoch{epoch}_{timestamp}_scale{lvl}.avi'
                    videoname = os.path.join(self.result_folder, videoname)
                    write_video_from_images(cam_path_scale, videoname)
                    shutil.rmtree(cam_path_scale)
                    
                h0.remove()
                h1.remove()
                h2.remove()


        return {
            'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': acc_list[0].avg, 
            'cls_acc_1': acc_list[1].avg,
            'cls_acc_2': acc_list[2].avg, 
            'rank_loss': rank_loss_meter.avg,
        }

    ##
    def pretrain_classification(self, max_epoch=1):

        print("pretran Classification")

        self.model.train()
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
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

                # data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data, target=target.unsqueeze(1)) ## [B, NumClasses]

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss = cls_loss_0.sum()

                loss = cls_loss
                loss.backward()
                self.optimizer.step()

        return


    ##
    def pretrain_apn(self, max_epoch=1):
        
        print("pretran APN")

        self.model.train()
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            ## weight
            params_classifier_0 = list(self.model.classifier_0.parameters())
            ## -2 because we have bias in the classifier
            weight_softmax_0 = params_classifier_0[-2].data
            # hook the feature extractor instantaneously and remove it once data is hooked
            f_conv_0 = []
            def hook_feature_conv_scale_0(module, input, output):
                f_conv_0.clear()
                f_conv_0.append(output.data)
            ## place hooker
            h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)

            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.trainloader), 
                total=len(self.trainloader),
                desc='pretrain_apn'+': ', 
                ncols=80, 
                leave=False):
            # for batch_idx, batch in enumerate(self.trainloader):
                data, target, idx = batch
                target = self.generate_confusion_target(target)

                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]

                # data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data, target=target.unsqueeze(1)) ## [B, NumClasses]

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss = cls_loss_0.sum()

                ### get CAM peak
                theta = self.get_cam_peak(f_conv_0[-1], weight_softmax_0[target,:])

                loss = F.mse_loss(t_01, theta)
                loss.backward()
                self.optimizer.step()

        return 
    
    # generate class activation mapping for the top1 prediction
    # def returnCAM(feature_conv, weight_softmax):
    #     # generate the class activation maps upsample to 256x256
    #     size_upsample = (256, 256)
    #     bz, nc, h, w = feature_conv.shape
    #     # print(weight_softmax.shape)
    #     cam = weight_softmax.dot(feature_conv.reshape((nc, h*w)))
    #     cam = cam.reshape(h, w)
    #     cam = cam - np.min(cam)
    #     cam_img = cam / np.max(cam)
    #     cam_img = np.uint8(255 * cam_img)
    #     return cam_img
    def get_cam_peak(self, feature, weight_softmax):
        B, C = weight_softmax.shape
        weight_softmax = weight_softmax.reshape(B, 1, C)
        _, nc, h, w = feature.shape
        cam = weight_softmax.bmm(feature.flatten(start_dim=2))
        cam = cam.reshape(B, -1)
        print('cam.shape: ', cam.shape)
        max_loc = torch.argmax(cam, dim=-1)
        # print('max_loc.shape: ', max_loc.shape)
        # print('max_loc: ', max_loc)
        # input()
        return

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