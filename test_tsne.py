import os
import cv2
import numpy as np
import time
import tqdm
from datetime import datetime
import collections
import argparse
from PIL import Image
import glob
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.new_dataloader import ImageDataset
from models import TCLoss, RACNN, RACNN3Scale
from racnn_trainer import RACNN_Trainer
from racnn3_trainer import RACNN3_Trainer
from utils.config import ConfigParser
from utils.logger import SimpleLogger
from utils.cam_drawer import CAMDrawer, largest_component, generate_bbox, returnCAM, returnTensorCAMs, write_text_to_img
from utils.logger import AverageMeter
from make_video import *

from sklearn.manifold import TSNE ## tsne

import matplotlib as mpl
import matplotlib.pyplot as plt ## plot

def discrete_cmap(N, base_cmap='viridis'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def colorFader(mix,c1='red',c2='blue'): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1-mix)*c1 + mix*c2)

class Tester():
    def __init__(self, model, criterion, test_dataset, resume_path, logger, device, config, bar=0.8):

        self.model = model.eval()
        self.criterion = criterion
        self.device = device
        self.test_dataset = test_dataset

        self.config = config
        self.logger = logger
        self.logger.info(config)
        
        ## Hyper-parameters
        self.margin = config['margin']

        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=device)
        # load architecture params from checkpoint.
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model = self.model.to(device)

        num_workers = 0 if config['disable_workers'] else 4
        self.testloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=num_workers) 
        
        ## get a savefolder here
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ckpt_folder = config['ckpt_folder']
        self.log_folder = config['log_folder']
        self.result_folder = config['result_folder']
        self.drawer = CAMDrawer(self.result_folder, bar=bar)
        
        
    def test_draw_results(self):
        
        ##
        cls_loss_meter = AverageMeter()
        rank_loss_meter = AverageMeter()
        accuracy_0 = AverageMeter()
        accuracy_1 = AverageMeter()
        accuracy_2 = AverageMeter()

        ##
        with torch.no_grad():
            ## weight
            params_classifier_0 = list(self.model.classifier_0.parameters())
            params_classifier_1 = list(self.model.classifier_1.parameters())
            params_classifier_2 = list(self.model.classifier_2.parameters())
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
            h0 = self.model.conv_scale_0[-2].register_forward_hook(hook_feature_conv_scale_0)
            h1 = self.model.conv_scale_1[-2].register_forward_hook(hook_feature_conv_scale_1)
            h2 = self.model.conv_scale_2[-2].register_forward_hook(hook_feature_conv_scale_2)
            
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

                ### Ranking loss
                probs_0 = F.softmax(out_0, dim=-1)
                probs_1 = F.softmax(out_1, dim=-1)
                probs_2 = F.softmax(out_2, dim=-1)
                gt_probs_0 = probs_0[list(range(B)), target]
                gt_probs_1 = probs_1[list(range(B)), target]
                gt_probs_2 = probs_2[list(range(B)), target]
                rank_loss_1 = self.criterion.PairwiseRankingLoss(gt_probs_0, gt_probs_1, margin=self.margin)
                rank_loss_2 = self.criterion.PairwiseRankingLoss(gt_probs_1, gt_probs_2, margin=self.margin)
                rank_loss = rank_loss_1.sum() + rank_loss_2.sum()

                ##
                cls_loss_meter.update(cls_loss, 1)
                rank_loss_meter.update(rank_loss, 1)
                # calculate accuracy
                accuracy_0.update(compute_acc(out_0, target, B_out), 1)
                accuracy_1.update(compute_acc(out_1, target, B_out), 1)
                accuracy_2.update(compute_acc(out_2, target, B_out), 1)

                img_path = self.testloader.dataset.get_fname(idx)
                # print(img_path)
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
            ## end for

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

            cam_path_scale_2 = os.path.join(self.result_folder, 'scale_2', f"epoch{epoch}")
            videoname_2 = f'video_epoch{epoch}_{timestamp}_scale2.avi'
            videoname_2 = os.path.join(self.result_folder, videoname_2)
            write_video_from_images(cam_path_scale_2, videoname_2)
            shutil.rmtree(cam_path_scale_2)

            h0.remove()
            h1.remove()
            h2.remove()

        return {
            'cls_loss': cls_loss_meter.avg, 
            'cls_acc_0': accuracy_0.avg, 
            'cls_acc_1': accuracy_1.avg, 
            'cls_acc_2': accuracy_2.avg, 
            'rank_loss': rank_loss_meter.avg,
        }



    def test_tsne(self, batch_size):
        
        ## init a testloader that can load a batch of data
        testloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4) 

        ## init the tsne method
        tsne_rnd_seed = 0
        tsne_method = TSNE(n_components=2, random_state=tsne_rnd_seed, perplexity=50, n_iter=2000)

        ## get latent representation
        with torch.no_grad():
            # self._save_checkpoint(epoch)
            ## weight
            params_classifier_0 = list(self.model.classifier_0.parameters())
            params_classifier_1 = list(self.model.classifier_1.parameters())
            params_classifier_2 = list(self.model.classifier_2.parameters())
            ## -2 because we have bias in the classifier
            weight_softmax_0 = np.squeeze(params_classifier_0[-2].data.cpu().numpy())
            weight_softmax_1 = np.squeeze(params_classifier_1[-2].data.cpu().numpy())
            weight_softmax_2 = np.squeeze(params_classifier_2[-2].data.cpu().numpy())
            print(params_classifier_2[-2])
            C = weight_softmax_0.shape[1]
            print("channel size: ", C)

            # hook the feature extractor instantaneously and remove it once data is hooked
            f_gap = []
            def hook_feature_gap(module, input, output):
                f_gap.append(output.squeeze().data.cpu())
                print("inside the hook: ", len(f_gap))
            
            ## place hooker
            h_gap = self.model.gap.register_forward_hook(hook_feature_gap)

            xy_embed_list = []
            target_list = []
            for batch_idx, batch in tqdm.tqdm(
                                            enumerate(testloader), 
                                            total=len(testloader),
                                            desc='test_tsne'+': ', 
                                            ncols=60, 
                                            leave=False):

                # if batch_idx == 2: break                            
                
                data, target, idx = batch
                target_list.append(target)
                data, target = data.to(self.device), target.to(self.device)
                B = data.shape[0]
                H, W = data.shape[-2:]
                # run forward to hook the features
                self.model(data, target=None) ## [B, NumClasses]
                
            ## convert latent reprs to low-dim vectors
            targets = torch.cat(target_list)
            print("lbls: ", targets.shape[0])

            colors = []
            M = targets.shape[0]
            for x in range(M):
                colors.append(colorFader(x/M))
            colors= np.array(colors)

            feats = torch.cat(f_gap)
            print("feats: ", feats.shape)
            feats = feats.reshape(len(target_list), 3, -1, C)
            feats = feats.permute(1,0,2,3).reshape(3, -1, C)
            print("feats: ", feats.shape)

            xy_embed = tsne_method.fit_transform(feats.reshape(-1, C))
            xy_embed = xy_embed.reshape(3, -1, 2)
            print(xy_embed.shape)
            print(type(xy_embed))
            xy_embed_save = xy_embed.reshape(-1,2)
            np.savetxt(os.path.join(self.result_folder,'embedding.txt'), xy_embed_save, fmt='%10.4f', delimiter=',')

            ## draw t-sne results
            fig=plt.Figure()
            ax=fig.add_axes([0,0,1,1])

            lbl0 = targets == 0 ## C
            lbl1 = targets == 1 ## H
            lbl2 = targets == 2 ## P

            # colors = discrete_cmap(targets.shape[0],base_cmap='cubehelix')

            ax.scatter(xy_embed[0,lbl0,0], xy_embed[0,lbl0,1], c=colors[lbl0,:], marker='o', edgecolors='y')
            ax.scatter(xy_embed[0,lbl1,0], xy_embed[0,lbl1,1], c=colors[lbl1,:], marker='v', edgecolors='m')
            ax.scatter(xy_embed[0,lbl2,0], xy_embed[0,lbl2,1], c=colors[lbl2,:], marker='s', edgecolors='c')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('embedding')

            # plt.savefig(os.path.join(self.result_folder,'embedding_0.png'))
            # plt.savefig(os.path.join(self.result_folder,'embedding_1.png'))
            # plt.savefig(os.path.join(self.result_folder,'embedding_2.png'))
            fig.savefig(os.path.join(self.result_folder,'embedding.png'))
            plt.close()



            ##
            h_gap.remove()
        return


def test_main(config):
    
    ## set seed
    torch.manual_seed(config['seed'])
    test_dataset = ImageDataset(
        'image_path_folder/test_image_list_sorted.txt', 
        'image_path_folder/test_image_label_sorted.txt', 
        is_training=False)

    ## device
    if config['device'] != -1:
        cuda_id = f"cuda:{config['device']}"
        device = torch.device(cuda_id)
    else:
        device = torch.device("cpu")

    ## model
    net = RACNN(num_classes=3, device=device)
    ## loss
    criterion = TCLoss(num_classes=3)

    ## logger
    logfname = os.path.join(config["log_folder"], "info.log")
    logger = SimpleLogger(logfname, 'debug')

    resume_path = config['resume']

    ## 
    # def __init__(self, model, criterion, test_dataset, resume_path, logger, device, config, bar=0.8):
    tester = Tester(net, criterion, test_dataset, resume_path, logger, device, config)
    tester.test_tsne(batch_size=32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=int,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-s', '--seed', default=None, help='random seeds')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='the size of each minibatch')
    parser.add_argument('--disable_workers', action="store_true")
    parser.add_argument('--comment', help="comments to the session", type=str)
    parser.add_argument('--config', default=None, type=str,
                        help='JSON config path')

    args = parser.parse_args()
    assert args.config is not None, "Please provide the JSON file path containing hyper-parameters for config the network"

    config = ConfigParser(args.config)

    ## allow cmd-line overide
    assert args.resume is not None, "In order to run test, resume path is required"
    config.set_content('resume', args.resume)
    
    if args.device is not None:
        config.set_content('device', args.device)
    if args.seed is not None:
        config.set_content('seed', args.seed)
    
    ##
    if args.comment is not None:
        config.set_content('comment', args.comment)

    test_main(config=config.get_config_parameters())