import time
import numpy as np
import os
import tqdm
from datetime import datetime
import collections
from utils.cam_drawer import CAMDrawer

import torch
import torch.nn.functional as F


from utils.logger import AverageMeter
from make_video import *

import shutil

# from trainer import Trainer


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
        self.margin = config['margin']
        self.max_epoch = config['max_epoch']
        self.time_consistency = True if config['temp_consistency_weight']>0.0 else False
        self.consistency_weight = config['temp_consistency_weight']
        self.save_period = config['ckpt_save_period']
        batch_size = config['batch_size']
        
        if config['disable_workers']:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        else:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 



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


    ## training session
    def train(self, max_epoch, do_validation=True):
        if max_epoch is not None:
            self.max_epoch = max_epoch

        # log = self.test_one_epoch(0)

        for epoch in range(max_epoch):
            ## training
            self.logger.info(f"epoch {epoch}:")
            self.logger.info(f"Training:")

           ##
            self.logger.info(f"Both")
            log = self.train_one_epoch(epoch)
            self.logger.info(log)

            # ##
            # self.logger.info(f"Classification")
            # log = self.train_one_epoch(epoch, self.loss_config_list['classification'])
            # self.logger.info(log)
            
            # ##
            # self.logger.info(f"Apn")
            # log = self.train_one_epoch(epoch, self.loss_config_list['apn'])
            # self.logger.info(log)
            
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
        
        self.model.train()

        loss_meter = AverageMeter()
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        info_loss_meter = AverageMeter()
        accuracy_0 = AverageMeter()
        accuracy_1 = AverageMeter()

        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch

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

            data, target = data.to(self.device), target.to(self.device)
            B = data.shape[0]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # data [B, C, H, W]
                # if loss_config == 'classification':
                #    out_0, out_1, t_01 = self.model(data, 0) ## [B, NumClasses]
                # elif loss_config == 'apn':
                #    out_0, out_1, t_01 = self.model(data, 1) ## [B, NumClasses]
                # else:
                #    ## whole
                #    out_0, out_1, t_01 = self.model(data, -1) ## [B, NumClasses]
                out_0, out_1, t_01, f_gap_1 = self.model(data, target.unsqueeze(1), -1) ## [B, NumClasses]
                
                # print("theta: ", t_01)
                
                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='mean')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='mean')
                cls_loss = cls_loss_0 + cls_loss_1
                
                # weights_0 = self.criterion.ComputeEntropyAsWeight(out_0)
                # weights_1 = self.criterion.ComputeEntropyAsWeight(out_1)
                # cls_loss = (cls_loss_0*(weights_0**2)+(1-weights_0)**2) + 0.3*(cls_loss_1*(weights_1**2)+(1-weights_1)**2)
                # cls_loss = cls_loss.sum()

                info_loss = self.criterion.ContrastiveLoss(f_gap_1)
                
                ### Ranking loss
                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                rank_loss = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                rank_loss = rank_loss.sum()
                # print("gt_probs_0: ", gt_probs_0)
                # print("gt_probs_1: ", gt_probs_1)
                # print("rank_loss: ", rank_loss)
                # print("info_loss: ", info_loss)
                # print("cls_loss: ", cls_loss)

                # if loss_config == 0: ##'classification'
                #     loss = cls_loss + 0.1*rank_loss
                # elif loss_config == 1: ##'apn'
                #     loss = 0.1*cls_loss + rank_loss
                # else: ##'whole'
                loss = 5.0*rank_loss + cls_loss + info_loss

                loss.backward()
                self.optimizer.step()

                # calculate accuracy
                # print(out_0.shape)
                # print(target.shape)
                correct_0 = (torch.max(out_0, dim=1)[1].view(target.size()).data == target.data).sum()
                train_acc_0 = 100. * correct_0 / self.trainloader.batch_size
                correct_1 = (torch.max(out_1, 1)[1].view(target.size()).data == target.data).sum()
                train_acc_1 = 100. * correct_1 / self.trainloader.batch_size
                
                loss_meter.update(loss, 1)
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)
                info_loss_meter.update(info_loss, 1)
                accuracy_0.update(train_acc_0, 1)
                accuracy_1.update(train_acc_1, 1)


        return {
            'cls_loss': cls_loss_meter.avg, 'cls_acc_0': accuracy_0.avg, 'cls_acc_1': accuracy_1.avg, 
            'rank_loss': rank_loss_meter.avg, 'info_loss': info_loss_meter.avg,
            'total_loss': loss_meter.avg
            }

    
    def train_rank_loss_one_epoch(self, epoch):
        print("train APN")
        self.model.train()
        loss_avg = AverageMeter()
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            B = data.shape[0]
            with torch.set_grad_enabled(True):
                ## data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data) ## [B, D]
                # print("theta: ", t_01)

                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                rank_loss = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)

                # print("gt_probs_0: ", gt_probs_0)
                # print("gt_probs_1: ", gt_probs_1)
                # print("rank_loss: ", rank_loss)
                rank_loss = rank_loss.mean()
                rank_loss.backward()
                self.optimizer.step()
                loss_avg.update(rank_loss, 1)
        return {'rank_loss': loss_avg.avg}

    ##
    def test_one_epoch(self, epoch):
        self.model.eval()
        
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        accuracy_0 = AverageMeter()
        accuracy_1 = AverageMeter()

        with torch.no_grad():
            if self.draw_cams and epoch % self.save_period == 0:
                ## weight
                params_classifier_0 = list(self.model.classifier_0.parameters())
                params_classifier_1 = list(self.model.classifier_1.parameters())
                ## -2 because we have bias in the classifier
                weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
                # weight_softmax_1 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
                weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
                
                # hook the feature extractor instantaneously and remove it once data is hooked
                f_conv_0, f_conv_1 = [], []
                def hook_feature_conv_scale_0(module, input, output):
                    f_conv_0.clear()
                    f_conv_0.append(output.data.cpu().numpy())
                def hook_feature_conv_scale_1(module, input, output):
                    f_conv_1.clear()
                    f_conv_1.append(output.data.cpu().numpy())
                ## place hooker
                h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
                # h1 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_1)
                h1 = self.model.conv_scale_1[-2].register_forward_hook(hook_feature_conv_scale_1)
                print("saving CAMs")

            
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
                assert B == 1, "test batch size should be 1"

                # for confusion_idx in range(3, 6):
                #     target_cofusion = target==confusion_idx
                #     target_replace = torch.randn(torch.sum(target_cofusion))
                #     if confusion_idx == 3:
                #         target_replace = torch.where(target_replace>0, torch.tensor([0]), torch.tensor([2]))
                #         target[target_cofusion] = target_replace
                #     elif confusion_idx == 4:
                #         target_replace = torch.where(target_replace>0, torch.tensor([0]), torch.tensor([1]))
                #         target[target_cofusion] = target_replace
                #     elif confusion_idx == 5:
                #         target_replace = torch.where(target_replace>0, torch.tensor([1]), torch.tensor([2]))
                #         target[target_cofusion] = target_replace
                # assert torch.sum(target==0)+torch.sum(target==1)+torch.sum(target==2) == target.numel()

                # data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data,target=None) ## [B, NumClasses]
                print(f"{batch_idx}: GT: {target} // theta: {t_01}")

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='none')
                weights_0 = self.criterion.ComputeEntropyAsWeight(out_0)
                weights_1 = self.criterion.ComputeEntropyAsWeight(out_1)
                cls_loss = cls_loss_0 * (weights_0**2) + (1-weights_0)**2 + cls_loss_1 * (weights_1**2) + (1-weights_1)**2
                cls_loss = cls_loss.sum()

                ### Ranking loss
                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                rank_loss = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                rank_loss = rank_loss.sum()

                # calculate accuracy
                correct_0 = (torch.max(out_0, dim=1)[1].view(target.size()).data == target.data).sum()
                train_acc_0 = 100. * correct_0 / self.testloader.batch_size
                correct_1 = (torch.max(out_1, 1)[1].view(target.size()).data == target.data).sum()
                train_acc_1 = 100. * correct_1 / self.testloader.batch_size
                
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)
                accuracy_0.update(train_acc_0, 1)
                accuracy_1.update(train_acc_1, 1)

                if self.draw_cams and epoch % self.save_period == 0:
                    img_path = self.testloader.dataset.get_fname(idx)
                    # print(img_path)
                    weight_softmax_0_gt = weight_softmax_0[target, :]
                    weight_softmax_1_gt = weight_softmax_1[target, :]
                    self.drawer.draw_cam(
                        epoch, gt_probs_0, target, 
                        weight_softmax_0_gt, f_conv_0[-1], 
                        img_path[0], sub_folder='scale_0')
                    self.drawer.draw_cam(
                        epoch, gt_probs_1, target, 
                        weight_softmax_1_gt, f_conv_1[-1], 
                        img_path[0], theta=t_01.cpu(), sub_folder='scale_1')
                    # input()

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

                h0.remove()
                h1.remove()


        return {
            'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': accuracy_0.avg, 'cls_acc_1': accuracy_1.avg, 
            'rank_loss': rank_loss_meter.avg,
        }

    ##
    def pretrain_classification(self, max_epoch=1):
        for i in range(max_epoch):
            log = self.train_classification_one_epoch(epoch)
        return log

    ##
    def pretrain_apn(self, max_epoch=1):
        return self.train_apn_one_epoch(epoch, detach='classification')
        
    ##
    def train_apn_one_epoch(self, epoch):
        raise NotImplementedError