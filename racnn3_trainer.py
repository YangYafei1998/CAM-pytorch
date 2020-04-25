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

## pt_float [-1.0, 1.0]
## scaling [0,1.0]
def attend_to_pt(x, pt_float, scaling):
    assert x.shape[0] == pt_float.shape[0]
    assert x.shape[0] == scaling.shape[0]

    B, C, H, W = x.shape
    grid_X, grid_Y = np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
    grid_X = torch.Tensor(grid_X).unsqueeze(0).unsqueeze(3)
    grid_Y = torch.Tensor(grid_Y).unsqueeze(0).unsqueeze(3)
    grid = torch.cat([grid_X, grid_Y], dim=-1)
    grid = grid.repeat([B, 1, 1, 1]).to(pt_float.device)

    ## when pt is pixel point (x, y)
    # size = torch.tensor([H//2, W//2])
    # ctr = torch.tensor([H//2, W//2]).type(torch.FloatTensor)
    # ctr = ctr.repeat([B, 1])
    # offset = (ctr-pt)/size
    offset = -1.0*pt_float
    offset = offset.unsqueeze(1).unsqueeze(1)
    ## grid_1 = grid_0 + ((x0,y0) - (x1,y1)) / (H/2, W/2)
    grid_deform = (grid + offset)
    scaling = scaling.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # print("scaling shape", scaling.shape)
    # print("grid_deform shape", grid_deform.shape)
    grid_deform = grid_deform*scaling
    out = F.grid_sample(x.to(pt_float.device), grid_deform, align_corners=True, padding_mode='zeros')
    return out

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
        for _ in range(4):
            self.pretrain_classification()
        for i in range(2):
            self.pretrain_apn(epoch=i)
        self.test_pretrained_apn(epoch=0)
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

            if epoch != 0 and epoch % self.interleaving_step == 0:
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
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        self.model.train()
        with torch.set_grad_enabled(True):
            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.trainloader), 
                total=len(self.trainloader),
                desc='pretrain_cls'+': ', 
                ncols=80, 
                leave=False):
            # for batch_idx, batch in enumerate(self.trainloader):

                # if batch_idx == 5: break

                self.optimizer.zero_grad()
                data, target, idx = batch
                target = self.generate_confusion_target(target)

                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]

                # data [B, C, H, W]
                out_list, t_list = self.model(data, target=target.unsqueeze(1), train_config=0) ## [B, NumClasses]

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_list[0], target, reduction='none')
                loss = cls_loss_0.sum()
                loss.backward()
                self.optimizer.step()
                
                correct = (torch.max(out_list[0], dim=1)[1].view(target.size()).data == target.data).sum()
                train_acc = 100. * correct / self.trainloader.batch_size
                acc_meter.update(train_acc, 1)
                loss_meter.update(loss)
            
            print("pretrain_classification loss: ", loss_meter.avg)
            print("pretrain_classification acc: ", acc_meter.avg)


        return


    ##
    def pretrain_apn(self, epoch, max_epoch=1, b_draw_cams=False):
        print("pretran APN")

        loss_meter = AverageMeter()

        self.model.train()
        with torch.set_grad_enabled(True):
            ## weight
            weight_softmax_list=[]
            for lvl in range(self.lvls):
                params_classifier = list(self.model.clsfierList[lvl].parameters())
                weight_softmax_list.append(np.squeeze(params_classifier[-2].data.detach().cpu().numpy()))
            # hook the feature extractor instantaneously and remove it once data is hooked
            # f_conv_0, f_conv_1, f_conv_2 = [], [], []
            f_conv_list = []
            def hook_feature_conv_scale_0(module, input, output):
                f_conv_list.clear()
                assert 0==len(f_conv_list)
                f_conv_list.append(output.data.detach().cpu().numpy())
            def hook_feature_conv_scale_1(module, input, output):
                assert 1==len(f_conv_list)
                f_conv_list.append(output.detach().cpu().numpy())
            def hook_feature_conv_scale_2(module, input, output):
                assert 2==len(f_conv_list)
                f_conv_list.append(output.data.detach().cpu().numpy())
            ## place hooker
            h0 = self.model.convList[0][-2].register_forward_hook(hook_feature_conv_scale_0)
            h1 = self.model.convList[1][-2].register_forward_hook(hook_feature_conv_scale_1)
            h2 = self.model.convList[2][-2].register_forward_hook(hook_feature_conv_scale_2)

            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.trainloader), 
                total=len(self.trainloader),
                desc='pretrain_apn'+': ', 
                ncols=80, 
                leave=False):
            # for batch_idx, batch in enumerate(self.trainloader):

                # if batch_idx == 5: break

                self.optimizer.zero_grad()
                data, target, idx = batch
                target = self.generate_confusion_target(target)

                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]

                # data [B, C, H, W]
                out_list, t_list = self.model(data, target=target.unsqueeze(1), train_config=1) ## [B, NumClasses]
                probs = F.softmax(out_list[lvl], dim=-1)
                gt_probs = probs[list(range(B)), target]

                ### get CAM peak
                lvl = 0
                feature = torch.from_numpy(f_conv_list[lvl])
                class_weight = torch.from_numpy(weight_softmax_list[lvl])[target,:]
                loss = self.compute_cam_area_in_box(feature, class_weight, t_list[lvl])
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.detach().item(), 1)

            #     if b_draw_cams:
            #         lvl = 0
            #         img_path = self.trainloader.dataset.get_fname(idx)
            #         weight_softmax_gt = weight_softmax_list[lvl][target.cpu(), :]
            #         self.drawer.draw_multiple_cams_with_zoom_in_box(
            #             epoch, target, img_path, 
            #             gt_probs, weight_softmax_gt, f_conv_list[lvl], 
            #             thetas=t_list[lvl].detach().cpu(), sub_folder=f'pretrain_apn_scale_{lvl}')

            # if b_draw_cams and epoch % self.save_period == 0:
            #     timestamp = self.result_folder.split(os.sep)[-2]
            #     for lvl in range(self.lvls):
            #         cam_path_scale = os.path.join(self.result_folder, f'pretrain_apn_scale_{lvl}', f"epoch{epoch}")
            #         videoname = f'video_apn_epoch{epoch}_{timestamp}_scale{lvl}.avi'
            #         videoname = os.path.join(self.result_folder, videoname)
            #         write_video_from_images(cam_path_scale, videoname)
            #         shutil.rmtree(cam_path_scale)
            
            ## end-for
        print("pretrain_apn loss: ", loss_meter.avg)                    
        h0.remove()
        h1.remove()
        h2.remove()
        return 
    
    def test_pretrained_apn(self, epoch=None):
        loss_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            ## weight
            weight_softmax_list=[]
            for lvl in range(self.lvls):
                params_classifier = list(self.model.clsfierList[lvl].parameters())
                weight_softmax_list.append(np.squeeze(params_classifier[-2].data.cpu().numpy()))
            # hook the feature extractor instantaneously and remove it once data is hooked
            # f_conv_0, f_conv_1, f_conv_2 = [], [], []
            f_conv_list = []
            def hook_feature_conv_scale_0(module, input, output):
                f_conv_list.clear()
                assert 0==len(f_conv_list)
                f_conv_list.append(output.data.cpu().numpy())
            def hook_feature_conv_scale_1(module, input, output):
                assert 1==len(f_conv_list)
                f_conv_list.append(output.data.cpu().numpy())
            def hook_feature_conv_scale_2(module, input, output):
                assert 2==len(f_conv_list)
                f_conv_list.append(output.data.cpu().numpy())
            ## place hooker
            h0 = self.model.convList[0][-2].register_forward_hook(hook_feature_conv_scale_0)
            h1 = self.model.convList[1][-2].register_forward_hook(hook_feature_conv_scale_1)
            h2 = self.model.convList[2][-2].register_forward_hook(hook_feature_conv_scale_2)

            for batch_idx, batch in tqdm.tqdm(
                enumerate(self.testloader), 
                total=len(self.testloader),
                desc='test pretrain_apn'+': ', 
                ncols=80, 
                leave=False):
            # for batch_idx, batch in enumerate(self.trainloader):

                
                data, target, idx = batch
                target = self.generate_confusion_target(target)

                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]

                # data [B, C, H, W]
                out_list, t_list = self.model(data, target=target.unsqueeze(1), train_config=1) ## [B, NumClasses]
                probs = F.softmax(out_list[lvl], dim=-1)
                gt_probs = probs[list(range(B)), target]

                ### get CAM peak
                lvl = 0
                feature = torch.from_numpy(f_conv_list[lvl])
                class_weight = torch.from_numpy(weight_softmax_list[lvl])[target,:]
                loss = self.compute_cam_area_in_box(feature, class_weight, t_list[lvl])
                
                loss_meter.update(loss.detach().item(), 1)

                ## write cams
                lvl = 0
                img_path = self.testloader.dataset.get_fname(idx)
                weight_softmax_gt = weight_softmax_list[lvl][target.cpu(), :]
                # epoch, gt_lbl, img_path, 
                # prob, weight_softmax, feature, theta, 
                # sub_folder=None, GT=None
                self.drawer.draw_single_cam_and_zoom_in_box(
                    epoch, target, img_path[0], 
                    gt_probs, weight_softmax_gt, f_conv_list[lvl], theta=t_list[lvl][0].cpu(), 
                    sub_folder=f'pretrain_apn_scale_{lvl}')

            
            timestamp = self.result_folder.split(os.sep)[-2]
            lvl = 0
            cam_path_scale = os.path.join(self.result_folder, f'pretrain_apn_scale_{lvl}', f"epoch{epoch}")
            videoname = f'video_apn_epoch{epoch}_{timestamp}_scale{lvl}.avi'
            videoname = os.path.join(self.result_folder, videoname)
            write_video_from_images(cam_path_scale, videoname)
            shutil.rmtree(cam_path_scale)
    
            ## end-for
        print("pretrain_apn loss: ", loss_meter.avg)                    
        h0.remove()
        h1.remove()
        h2.remove()
        return


    def compute_cam_area_in_box(self, feature, class_weight, box_theta):
        B, C, H, W = feature.shape
        class_weight = class_weight.reshape(B, 1, C)
        act_map = class_weight.bmm(feature.view(B, C, -1)).reshape(B, 1, H, W)
        x,y,s = box_theta[:,0], box_theta[:,1], box_theta[:,2]
        xy = torch.stack([x,y], dim=1)
        shift_map = attend_to_pt(act_map, xy, s)
        scores = torch.sum(shift_map.flatten(start_dim=1), dim=1)
        return -1.0*scores.sum()


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