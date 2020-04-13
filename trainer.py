import torch
import time
import numpy as np
import os
from torch.autograd import Variable
import tqdm
from datetime import datetime


# hook the feature extractor
final_conv=''
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.clear()
    # print("hook feature")
    # print(len(features_blobs))
    features_blobs.append(output.data.cpu().numpy())

class Trainer():
    def __init__(self, model, optimizer, lr_scheduler, criterion, train_dataset, test_dataset, config):
        
        self.config = config
        
        ## device
        if config['device'] != -1:
            cuda_id = "cuda:"+config['device']
            self.device = torch.device(cuda_id)
        else:
            self.device = torch.device("cpu")
        
        self.model = model.to(self.device)
        self.model._modules.get('features')[-2].register_forward_hook(hook_feature)
        if config['resume'] is not None:
            pass
            ## resume the network

        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        ## Hyper-parameters
        self.max_epoch = config['max_epoch']
        self.time_consistency = True if config['temp_consistency_weight']>0.0 else False
        self.consistency_weight = config['temp_consistency_weight']
        
        ## set seed
        torch.manual_seed(config['seed'])

        ## get a savefolder here
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.ckpt_folder = config['ckpt_folder']
        self.log_folder = config['log_folder']
        self.result_folder = config['result_folder']

        batch_size = config['batch_size']
        if config['disable_workers']:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0) 
        else:
            self.trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4) 
            self.testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4) 


    def train(self, max_epoch, do_validation=True):
        if max_epoch is not None:
            self.max_epoch = max_epoch

        for epoch in range(max_epoch):
            self.train_one_epoch(epoch)
            self.test_one_epoch(epoch)
            if epoch % 10 == 0:
                ckpt_name = self.save_foldername + 'epoch' +str(int(epoch)) + f'-acc{acc_avg}' + f'-{int(time.time())}.pt'
                torch.save(model.state_dict(), ckpt_name)
        print("training finished")


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
        print('Epoch {}/{}'.format(epoch, self.max_epoch - 1))
        
        # for batch_idx, batch in tqdm.tqdm(
        #     enumerate(self.trainloader), 
        #     total=len(self.trainloader),
        #     desc='train'+': ', 
        #     ncols=80, 
        #     leave=False):
        for batch_idx, batch in enumerate(self.trainloader):
            data, target, idx = batch
            data, target = data.to(self.device), target.to(self.device)
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                output = self.model(data)
                
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

        acc_avg = acc_sum.item() / len(self.trainloader)
        loss_avg = loss_sum / len(self.trainloader.dataset)
        cls_loss_avg = cls_sum / len(self.trainloader.dataset)
        temp_loss_avg = temp_sum / len(self.trainloader.dataset)
            
        print()
        print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
        if self.time_consistency:
            print('Train Epoch: {}\tClassification Loss: {:.3f}\tCoherence Loss: {:.3f}'.format(epoch, cls_loss_avg, temp_loss_avg))

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
        
        return   
        

    def test_one_epoch(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.set_grad_enabled(False):
            for data, target in testloader:
                if use_cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)

                # sum up batch loss
                loss, preds = criterion.ImgLvlClassLoss(output, target)


                test_loss += torch.sum(loss)
                # get the index of the max log-probability
                pred = output.data.max(1, keepdim=True)[1]
    
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()


            test_loss /= len(testloader.dataset)
            test_acc = 100. * correct / len(testloader.dataset)
            test_acc = test_acc.item()
            test_loss = test_loss.item()

            
            result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
                test_loss, correct, len(testloader.dataset), test_acc)
            print(result)
            if test_acc > 95:
                print(f"save best model with accuracy {test_acc}")
                torch.save(model.state_dict(), 'checkpoint/' +'epoch' +str(int(epoch)) + f'-acc{test_acc}' + f'-{int(time.time())}.pt')
            if epoch % 10 == 0:
                with open('result/result.txt', 'a') as f:
                    f.write(result)
                f.close()

            with open('result/test_acc.txt', 'a') as f:
                f.write(str(test_acc) + ',')
            f.close()
            with open('result/test_loss.txt', 'a') as f:
                f.write(str(test_loss) + ',')
            f.close()
