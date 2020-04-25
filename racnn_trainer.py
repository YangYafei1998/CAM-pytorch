import time
import numpy as np
import os
import tqdm
from datetime import datetime
import collections
from utils.cam_drawer import CAMDrawer, largest_component, generate_bbox, returnCAM, write_text_to_img

import torch
import torch.nn.functional as F


from utils.logger import AverageMeter
from make_video import *

import shutil

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
        self.time_consistency = config.get('temporal', False) ## default is fault
        if self.time_consistency:
            self.consistency_weight = config['temp_consistency_weight']

        self.save_period = config['ckpt_save_period']
        batch_size = config['batch_size']
        
        if config['disable_workers']:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            self.pretrainloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        else:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
            self.pretrainloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 



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
                'pretrain-checkpoint-epoch{}.pth'.format(epoch)
            )
        else:
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
        print(checkpoint['config'])
        if not load_pretrain:
            self.start_epoch = checkpoint['epoch'] + 1
        
        # self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        # if checkpoint['config']['model'] != self.config['model']:
        #     msg = ("Warning: Architecture configuration given in config file is"
        #            " different from that of checkpoint."
        #            " This may yield an exception while state_dict is being loaded.")
        #     self.logger.warning(msg)
        # else:
        #     self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # if not load_pretrain:
        #     # uncomment this line if you want to use the resumed optimizer
        #     # load optimizer state from checkpoint only when optimizer type is not changed.
        #     ckpt_opt_type = checkpoint['config']['optimizer']['type']
        #     if ckpt_opt_type != self.config['optimizer']['type']:
        #         msg = ("Warning: Optimizer type given in config file is different from"
        #             "that of checkpoint.  Optimizer parameters not being resumed.")
        #         self.logger.warning(msg)
        #     else:
        #         self.optimizer.load_state_dict(checkpoint['optimizer'])

        # # self.logger = checkpoint['logger']
        # msg = "Checkpoint '{}' (epoch {}) loaded"
        # self.logger.info(msg .format(resume_path, self.start_epoch))
        if not load_pretrain:
            print("load to resume")
        else:
            print("load pretrained model")

    ## pre-train session
    def pretrain(self):
        for i in range(0, 5):
            self.pretrain_classification()
            self._save_checkpoint(i, pretrain_ckp=True)
        for _ in range(6):
            self.pretrain_apn()
        print("Pre-train Finished")


    ## training session
    def train(self, max_epoch, do_validation=True):
        if max_epoch is not None:
            self.max_epoch = max_epoch
        
        self.pretrain()

        for epoch in range(max_epoch):
            ## training
            self.logger.info(f"epoch {epoch}:\n")
            self.logger.info(f"Training:\n")

            self.logger.info(f"Classification\n")
            log = self.train_one_epoch(epoch, 0)
            self.logger.info(log)
            # self.pretrain()


            if epoch % self.interleaving_step == 0:
                self.logger.info(f"APN\n")
                log = self.train_one_epoch(epoch, 1)
                self.logger.info(log)

            # if epoch % 7 == 5 or epoch % 7 == 6:
            #     self.logger.info(f"APN")
            #     log = self.train_one_epoch(epoch, 1)
            #     self.logger.info(log)
            # else:
            #     self.logger.info(f"Classification")
            #     log = self.train_one_epoch(epoch, 0)
            #     self.logger.info(log)
                
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
            target = self.generate_confusion_target(target)
            
            data, target = data.to(self.device), target.to(self.device)
            
            B = data.shape[0]
            H, W = data.shape[-2:] # [-2:]==[-2,-1]; [-2:-1] == [-2]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # data [B, C, H, W]
                if self.time_consistency:
                    data = data.view(B*3, 3, H, W)
                    target = target.view(B*3)

                out_0, out_1, t_01, f_gap_1 = self.model(data, target.unsqueeze(1)) ## [B, NumClasses]
                
                # print("theta: ", t_01)
                
                # ### infoNCE loss
                # info_loss = self.criterion.ContrastiveLoss(f_gap_1)

                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='none')
                cls_loss = cls_loss_0.sum() + cls_loss_1.sum()
                
                ### Ranking loss
                # B3 = out_0.shape[0]
                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                rank_loss = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                rank_loss = rank_loss.sum()

                # ### Temporal coherence
                # if self.time_consistency:
                #     temp_loss = 0.0
                #     out_0 = out_0.view(B, 3, 3)
                #     out_1 = out_1.view(B, 3, 3)
                #     target = target.view(B, 3)
                #     # t_01 = t_01.view(B, 3)
                #     # f_gap_1 = f_gap_1.view(B, 3)
                #     temp_loss += self.criterion.TemporalConsistencyLoss(out_0[0], out_0[1], out_0[2])
                #     temp_loss += self.criterion.TemporalConsistencyLoss(out_1[0], out_1[1], out_1[2])

                # print("gt_probs_0: ", gt_probs_0)
                # print("gt_probs_1: ", gt_probs_1)
                # print("rank_loss: ", rank_loss)
                # print("info_loss: ", info_loss)
                # print("cls_loss: ", cls_loss)

                if loss_config == 0: ##'classification'
                    loss = cls_loss
                elif loss_config == 1: ##'apn'
                    loss = rank_loss
                else: ##'whole'
                    loss = rank_loss + cls_loss + info_loss
                    if self.time_consistency:
                        loss += 0.1*temp_loss

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
                # info_loss_meter.update(info_loss, 1)
                accuracy_0.update(train_acc_0, 1)
                accuracy_1.update(train_acc_1, 1)


        return {
            'cls_loss': cls_loss_meter.avg, 'cls_acc_0': accuracy_0.avg, 'cls_acc_1': accuracy_1.avg, 
            'rank_loss': rank_loss_meter.avg, 
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

        with torch.no_grad():
            # if self.draw_cams:
            if self.draw_cams and epoch % self.save_period == 0:
                self._save_checkpoint(epoch)
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
                data, target, idx, locationGT = batch
                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]
                assert B == 1, "test batch size should be 1"


                # data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data,target=None) ## [B, NumClasses]
                print(f"{batch_idx}: GT: {target} // theta: {t_01}")
                ### Classification loss
                cls_loss_0, preds_0 = self.criterion.ImgLvlClassLoss(out_0, target, reduction='none')
                cls_loss_1, preds_1 = self.criterion.ImgLvlClassLoss(out_1, target, reduction='none')
                cls_loss = cls_loss_0.sum() + cls_loss_1.sum()

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
                # if self.draw_cams:
                if self.draw_cams and epoch % self.save_period == 0:
                    img_path = self.testloader.dataset.get_fname(idx)
                    # print(img_path)
                    weight_softmax_0_gt = weight_softmax_0[target, :]
                    weight_softmax_1_gt = weight_softmax_1[target, :]
                    self.drawer.draw_cam(
                        epoch, gt_probs_0, target, 
                        weight_softmax_0_gt, f_conv_0[-1], 
                        img_path[0], GT = locationGT[0], sub_folder='scale_0')
                    self.drawer.draw_cam(
                        epoch, gt_probs_1, target, 
                        weight_softmax_1_gt, f_conv_1[-1], 
                        img_path[0], GT = locationGT[0], theta=t_01.cpu(), sub_folder='scale_1')
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

                h0.remove()
                h1.remove()


        return {
            'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': accuracy_0.avg, 'cls_acc_1': accuracy_1.avg, 
            'rank_loss': rank_loss_meter.avg,
        }

    ##
    def pretrain_classification(self, max_epoch=5):
        print("pretran Classification")

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

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
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
            f_conv_0.append(output.data.data.cpu().numpy())
        ## place hooker
        h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
        # print(len(f_conv_0))
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.pretrainloader), 
            total=len(self.pretrainloader),
            desc='pretrain_apn'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx, _ = batch
            img_path = self.testloader.dataset.get_fname(idx)
            img = cv2.imread(img_path[0], -1) ## [H, W, C]
            target = self.generate_confusion_target(target)

            data, target = data.to(self.device), target.to(self.device)
            B = data.shape[0]

            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # data [B, C, H, W]
                out_0, out_1, t_01, _ = self.model(data, target=target.unsqueeze(1), train_config=1) ## [B, NumClasses]
                ### get cropped region calculated by the APN
                out_len = 256 * t_01[0][2]
                out_center_x = (1 + t_01[0][0]) * 128
                out_center_y = (1 + t_01[0][1]) * 128
                ### get CAM peak
                coordinate, heatmap = self.get_cam_peak(f_conv_0[-1], weight_softmax_0[target,:])
                cam = heatmap * 0.3 + img * 0.5
                cv2.rectangle(cam, (coordinate[1], coordinate[0]), (coordinate[3], coordinate[2]), (0, 255, 0), 2)#peak activation region
                cv2.rectangle(cam, (out_center_x - out_len/2, out_center_x - out_len/2), (out_center_x + out_len/2, out_center_x + out_len/2), (0, 0, 255), 2) # cropped region by APN

                center_x = (coordinate[1]+coordinate[3])/2
                cam_x = (center_x - 128)/128
                center_y = (coordinate[0]+coordinate[2])/2
                cam_y = (center_y - 128)/128
                cam_l = (coordinate[3]-coordinate[1]+coordinate[2]-coordinate[0])/(2*256)

                target_pos = torch.FloatTensor(np.array([cam_x, cam_y, cam_l]))

                write_text_to_img(cam, f"target: {target}", org=(20,50), fontScale = 0.3)
                write_text_to_img(cam, f"t_01: {t_01}", org=(20,65), fontScale = 0.3)
                write_text_to_img(cam, f"target: {target_pos}", org=(20,80), fontScale = 0.3)
                cv2.imwrite('/userhome/30/yfyang/CAM-pytorch/saved/debug/'+f'{int(time.time())}.jpg', cam)
                
                loss = F.mse_loss(t_01[0], target_pos.cuda())
                ##print(loss)
                loss.backward()
                self.optimizer.step()
                loss_meter.update(loss.item())
        print(loss_meter.avg)
        h0.remove()
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
        # cam = returnCAM(feature.detach().cpu().numpy(), weight_softmax.detach().cpu().numpy())
        cam = returnCAM(feature, weight_softmax)
        cv2.imwrite('test.jpg', cam)
        image = cv2.resize(cam, (256, 256))
        prop = generate_bbox(image, 0.8)
        heatmap = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        return prop.bbox, heatmap
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