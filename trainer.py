import torch
import time
import numpy as np
import os
from torch.autograd import Variable
import tqdm
from datetime import datetime
import collections
from utils.cam_drawer import CAMDrawer

from utils.logger import AverageMeter

# hook the feature extractor
final_conv=''
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.clear()
    # print("hook feature")
    # print(len(features_blobs))
    features_blobs.append(output.data.cpu().numpy())

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, criterion, train_dataset, test_dataset, logger, config):
        
        ## 
        self.config = config
        self.logger = logger

        self.logger.info(config)

        ## set seed
        torch.manual_seed(config['seed'])

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
        self.model._modules.get('features')[-2].register_forward_hook(hook_feature)
        if config['resume'] is not None:
            pass
            ## resume the network
            self._resume_checkpoint(config['resume'])

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        ## Hyper-parameters
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

        # self.test_one_epoch(0)

        for epoch in range(max_epoch):
            ## training
            if self.time_consistency:
                log = self.train_one_epoch_temporal_coherence(epoch)
            else:
                log = self.train_one_epoch(epoch)
            self.logger.info('Training Epoch {}/{}\n'.format(epoch, self.max_epoch - 1))
            self.logger.info(log)
            self.logger.info(' ') ## spacing
            
            ## testing
            test_log = self.test_one_epoch(epoch)
            self.logger.info('Testing Epoch {}/{}\n'.format(epoch, self.max_epoch - 1))
            self.logger.info(test_log)
            self.logger.info(' ') ## spacing
            self.logger.info(' ') ## spacing

            ## save
            if epoch % self.save_period == 0:
                # ckpt_name = os.path.join(self.ckpt_folder, f'checkpoint_epoch{epoch}.pt')
                # torch.save(self.model.state_dict(), ckpt_name)
                self._save_checkpoint(epoch)
                print("save checkpoint...")

        print("training finished")


    ##
    def train_one_epoch(self, epoch):
        if self.time_consistency:
            print("Build with time coherence loss: ", self.consistency_weight)
        else:
            assert self.consistency_weight==0
            print("Build with only classification loss")

        self.model.train()
        correct, total = 0.0, 0.0
        acc_sum, loss_sum = 0.0, 0.0
        cls_sum, temp_sum = 0.0, 0.0
        
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            data, target = data.to(self.device), target.to(self.device)
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = self.model(data)
                
                # if batch_idx == 3:
                #     break

                # calculate accuracy
                correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
                total += self.trainloader.batch_size
                # total += 32
                train_acc = 100. * correct / total
                # print(train_acc)
                acc_sum += train_acc
                
                loss, preds = self.criterion.ImgLvlClassLoss(output, target)
                cls_loss = loss

                temp_loss = 0.0            
                if self.time_consistency:
                    idx_prev = torch.max(idx-1, torch.tensor(0))
                    idx_next = torch.min(idx+1, torch.tensor(len(self.trainloader.dataset)-1))

                    inputs_prev, target_prev = self.trainloader.dataset.get_data_with_idx(idx_prev)
                    inputs_next, target_next = self.trainloader.dataset.get_data_with_idx(idx_next)

                    outputs_prev = self.model(inputs_prev.to(self.device, dtype=torch.float))
                    outputs_next = self.model(inputs_next.to(self.device, dtype=torch.float))
                                                    
                    temp_loss = self.criterion.TemporalConsistencyLoss(output, outputs_prev, outputs_next)
                    loss = loss + self.consistency_weight * temp_loss

                ## backward
                loss.backward()
                self.optimizer.step()
                # loss = torch.sum(loss)

                ## recording
                loss_sum += loss.item()
                if self.time_consistency:
                    cls_sum += cls_loss.item()
                    temp_sum +=  temp_loss.item()
            # print('Train Epoch: {}-{}\tLoss: {:.3f}\tAccuracy: {:.3f}%'.format(epoch, batch_idx, loss_sum/total, train_acc))

        acc_avg = acc_sum.item() / len(self.trainloader)
        loss_avg = loss_sum / len(self.trainloader)
        cls_loss_avg = cls_sum / len(self.trainloader)
        temp_loss_avg = temp_sum / len(self.trainloader)
            
        # print()
        # print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
        # if self.time_consistency:
        #     print('Train Epoch: {}\tClassification Loss: {:.3f}\tCoherence Loss: {:.3f}'.format(epoch, cls_loss_avg, temp_loss_avg))

        # ## original code section 
        # with open(f'result/train_acc.txt', 'a') as f:
        #     f.write(str(acc_avg) + ",")
        # f.close()
        # with open(f'result/train_loss.txt', 'a') as f:
        #     f.write(str(loss_avg) + ",") 
        # f.close()
        # if time_consistency:
        #     with open(f'result/train_cls_loss.txt', 'a') as f:
        #         f.write(str(cls_loss_avg) + ",") 
        #     f.close()
        #     with open(f'result/train_temp_loss.txt', 'a') as f:
        #         f.write(str(temp_loss_avg) + ",") 
        #     f.close()

        log = {
            'train_acc': acc_avg,
            'train_loss': loss_avg,
            'train_cls_loss': cls_loss_avg,
            'train_temp_loss': temp_loss_avg,
        }
        return log
        

    def train_one_epoch_temporal_coherence(self, epoch):
        print("Build with time coherence loss: ", self.consistency_weight)

        self.model.train()
        correct, total = 0.0, 0.0
        acc_sum, loss_sum = 0.0, 0.0
        cls_sum, temp_sum = 0.0, 0.0
        
        for batch_idx, batch in tqdm.tqdm(
            enumerate(self.trainloader), 
            total=len(self.trainloader),
            desc='train'+': ', 
            ncols=80, 
            leave=False):
        # for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            data, target = data.to(self.device), target.to(self.device)
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):

                B, _, C, H, W = data.shape
                data = data.view(B*3, C, H, W) ## [B, 3, 3, H, W] --> [B*3, 3, H, W]
                target = target.view(-1) ## [B, 3] --> [B*3]

                output = self.model(data) ## [B*3, D, 1]
                
                # calculate accuracy
                correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
                total += self.trainloader.batch_size
                # total += 32
                train_acc = 100. * correct / total
                # print(train_acc)
                acc_sum += train_acc
                
                loss, preds = self.criterion.ImgLvlClassLoss(output, target)
                cls_loss = loss

                ## temporal coherence loss
                D = output.shape[1]
                output = output.view(B, 3, D) ## [B*3, D] --> [B, 3, D]
                temp_loss = self.criterion.TemporalConsistencyLoss(output[:, 1, :], output[:, 0, :], output[:, 2, :])
                loss = loss + self.consistency_weight * temp_loss

                ## backward
                loss.backward()
                self.optimizer.step()
                # loss = torch.sum(loss)

                ## recording
                loss_sum += loss.item()
                if self.time_consistency:
                    cls_sum += cls_loss.item()
                    temp_sum +=  temp_loss.item()
            # print('Train Epoch: {}-{}\tLoss: {:.3f}\tAccuracy: {:.3f}%'.format(epoch, batch_idx, loss_sum/total, train_acc))

        acc_avg = acc_sum.item() / len(self.trainloader)
        loss_avg = loss_sum / len(self.trainloader)
        cls_loss_avg = cls_sum / len(self.trainloader)
        temp_loss_avg = temp_sum / len(self.trainloader)
            
        # print()
        # print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
        # if self.time_consistency:
        #     print('Train Epoch: {}\tClassification Loss: {:.3f}\tCoherence Loss: {:.3f}'.format(epoch, cls_loss_avg, temp_loss_avg))

        # ## original code section 
        # with open(f'result/train_acc.txt', 'a') as f:
        #     f.write(str(acc_avg) + ",")
        # f.close()
        # with open(f'result/train_loss.txt', 'a') as f:
        #     f.write(str(loss_avg) + ",") 
        # f.close()
        # if time_consistency:
        #     with open(f'result/train_cls_loss.txt', 'a') as f:
        #         f.write(str(cls_loss_avg) + ",") 
        #     f.close()
        #     with open(f'result/train_temp_loss.txt', 'a') as f:
        #         f.write(str(temp_loss_avg) + ",") 
        #     f.close()

        log = {
            'train_acc': acc_avg,
            'train_loss': loss_avg,
            'train_cls_loss': cls_loss_avg,
            'train_temp_loss': temp_loss_avg,
        }
        return log
        

    ##
    def test_one_epoch(self, epoch):
        self.model.eval()
        
        test_loss = 0
        correct = 0
        with torch.no_grad():
            if self.draw_cams:
                params = list(self.model.parameters())
                weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

            for batch_idx, batch in enumerate(self.testloader):
                data, target, idx = batch
                data, target = data.to(self.device), target.to(self.device)
                data, target = Variable(data), Variable(target)
                output = self.model(data)

                B = output.shape[0]
                assert B == 1, "test batch size should be 1"

                # sum up batch loss
                loss, preds = self.criterion.ImgLvlClassLoss(output, target)

                # test_loss += torch.sum(loss)
                test_loss += loss
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
    
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

                if self.draw_cams and epoch % 10 == 0:
                    img_path = self.testloader.dataset.get_fname(idx)
                    self.drawer.draw_cam(epoch, output, weight_softmax, features_blobs, img_path[0])

            test_loss /= len(self.testloader)
            test_acc = 100. * correct / len(self.testloader.dataset)
            test_acc = test_acc.item()
            test_loss = test_loss.item()

         
            
            # result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            #     test_loss, correct, len(self.testloader.dataset), test_acc)
            # print(result)
            # if test_acc > 95:
            #     print(f"save best model with accuracy {test_acc}")
            #     torch.save(self.model.state_dict(), 'checkpoint/' +'epoch' +str(int(epoch)) + f'-acc{test_acc}' + f'-{int(time.time())}.pt')
            # # if epoch % 10 == 0:
            #     with open('result/result.txt', 'a') as f:
            #         f.write(result)

            # with open('result/test_acc.txt', 'a') as f:
            #     f.write(str(test_acc) + ',')
            
            # with open('result/test_loss.txt', 'a') as f:
            #     f.write(str(test_loss) + ',')
            
        log = {
                'test_acc': test_acc,
                'test_loss': test_loss,
            }
        return log



    ## this should be moved to loss
    def localization_loss(inPath, net, features_blobs, classes, outPath, GT):
        image_list = []
        f = open(GT, "r")
        for filename in glob.glob(inPath): 
            image_list.append(filename)
        image_list.sort()
        total_kl_loss = 0
        total_box_iou = 0
        total_pixel_iou = 0
        for i in range(len(image_list)):
            im = Image.open(image_list[i])
            line = f.readline()
            kl_loss, box_iou, pixel_iou = get_localization_loss(net, features_blobs, im, classes, image_list[i], outPath, line)
            total_kl_loss = total_kl_loss + kl_loss
            total_box_iou = total_box_iou + box_iou
            total_pixel_iou = total_pixel_iou + pixel_iou
        return total_kl_loss/len(image_list), total_kl_loss, total_box_iou/len(image_list), total_box_iou, total_pixel_iou/len(image_list), total_pixel_iou


    ## evaluate localization
    def test_localization(self, features_blobs, classes):
        ## harddrive
        harddrive_output = localization_loss(
            '/userhome/30/yfyang/fyp_data/test/images/CHardDrive/*.png', 
            self.model, 
            features_blobs, 
            classes, 
            '/userhome/30/yfyang/pytorch-CAM/result/CAM/', 
            '/userhome/30/yfyang/fyp_data/test/images/CHardDrive_GT.txt')
        harddrive_loss, harddrive_loss_total, \
        harddrive_box_iou, harddrive_total_box_iou, \
        harddrive_pixel_iou, harddrive_total_pixel_iou = harddrive_output


        powersupply_output = localization_loss(
            '/userhome/30/yfyang/fyp_data/test/images/CPowerSupply/*.png', 
            self.model, features_blobs, classes, 
            '/userhome/30/yfyang/pytorch-CAM/result/CAM/', 
            '/userhome/30/yfyang/fyp_data/test/images/CPowerSupply_GT.txt')
        powersupply_loss, powersupply_loss_total, \
        powersupply_box_iou, powersupply_total_box_iou, \
        powersupply_pixel_iou, powersupply_total_pixel_iou = powersupply_output

        cdrom_output = localization_loss(
            '/userhome/30/yfyang/fyp_data/test/images/CCDRom/*.png', 
            self.model, features_blobs, classes, 
            '/userhome/30/yfyang/pytorch-CAM/result/CAM/', 
            '/userhome/30/yfyang/fyp_data/test/images/CCDRom_GT.txt')
        cdrom_loss, cdrom_loss_total, \
        cdrom_box_iou, cdrom_total_box_iou, \
        cdrom_pixel_iou, cdrom_total_pixel_iou = cdrom_output 
        
        total_avg = (harddrive_loss_total + powersupply_loss_total + cdrom_loss_total)/len(testloader.dataset)
        box_iou_avg = (harddrive_total_box_iou + powersupply_total_box_iou + cdrom_total_box_iou)/len(testloader.dataset)
        pixel_iou_avg = (harddrive_total_pixel_iou + powersupply_total_pixel_iou + cdrom_total_pixel_iou)/len(testloader.dataset)
        log = {
            'test_kl_loss': total_avg,
            'box_iou_avg': box_iou_avg,
            'pixel_iou_avg': pixel_iou_avg,

            'harddrive_kl_loss': harddrive_loss, 
            'powersupply_kl_loss': powersupply_loss, 
            'cdrom_kl_loss': cdrom_loss, 
            
            'harddrive_box_iou': harddrive_box_iou, 
            'powersupply_box_iou': powersupply_box_iou, 
            'cdrom_box_iou': cdrom_box_iou, 

            'harddrive_pixel_iou': harddrive_pixel_iou,
            'powersupply_pixel_iou': powersupply_pixel_iou,
            'cdrom_pixel_iou': cdrom_pixel_iou,
        }
        return log