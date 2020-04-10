import torch
import time
import numpy as np

from torch.autograd import Variable
import tqdm
torch.manual_seed(0)

def train(trainloader, model, use_cuda, epoch, num_epochs, criterion, optimizer, time_consistency = True):
    if time_consistency:
        print("Build with time coherence loss")
    else:
        print("Build with only classification loss")
    
    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    cls_sum, temp_sum = 0, 0
    i = 0
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    for data, target, idx in tqdm.tqdm(trainloader, total=len(trainloader),desc='train'+': ', ncols=80, leave=False):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            output = model(data)
            # calculate accuracy
            correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
            total += trainloader.batch_size
            # total += 32
            train_acc = 100. * correct / total
            acc_sum += train_acc
            i += 1
            loss, preds = criterion.ImgLvlClassLoss(output, target)

            temp_loss = 0.0

            
            if time_consistency:
                idx_prev = torch.max(idx-1, torch.tensor(0))
                idx_next = torch.min(idx+1, torch.tensor(len(trainloader.dataset)-1))

                inputs_prev, target_prev = trainloader.dataset.get_data_with_idx(idx_prev)
                inputs_next, target_next = trainloader.dataset.get_data_with_idx(idx_next)

                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                outputs_prev = model(inputs_prev.to(device, dtype=torch.float))
                outputs_next = model(inputs_next.to(device, dtype=torch.float))
                                                
                temp_loss = criterion.TemporalConsistencyLoss(output, outputs_prev, outputs_next)

            cls_loss = loss
            loss = loss + 0.5 * temp_loss
            loss.backward()
            optimizer.step()
            loss = torch.sum(loss)
            loss_sum += loss.item()
            if time_consistency:
                cls_sum += cls_loss.item()
                temp_sum +=  temp_loss.item()
        acc_avg = acc_sum.item() / i
        loss_avg = loss_sum / len(trainloader.dataset)
        cls_loss_avg = cls_sum / len(trainloader.dataset)
        temp_loss_avg = temp_sum / len(trainloader.dataset)
        
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
    if time_consistency:
        print('Train Epoch: {}\tClassification Loss: {:.3f}\tCoherence Loss: {:.3f}'.format(epoch, cls_loss_avg, temp_loss_avg))

    with open(f'result/train_acc.txt', 'a') as f:
        f.write(str(acc_avg) + ",")
    f.close()
    with open(f'result/train_loss.txt', 'a') as f:
        f.write(str(loss_avg) + ",") 
    f.close()
    if time_consistency:
        with open(f'result/train_cls_loss.txt', 'a') as f:
            f.write(str(cls_loss_avg) + ",") 
        f.close()
        with open(f'result/train_temp_loss.txt', 'a') as f:
            f.write(str(temp_loss_avg) + ",") 
        f.close()
    
    return model   
    
    



def test(testloader, model, use_cuda, criterion, epoch):
    model.eval()
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
