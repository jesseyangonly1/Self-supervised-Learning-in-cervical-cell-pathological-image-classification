from __future__ import print_function
import torch
import torch.optim as optim
import torchvision as torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import time


num_epochs = 50
momentum = 0.8
random_seed = 1
log_interval = 10
torch.manual_seed(random_seed)
batch_size = 64
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
normalize = transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
brightness_change = transforms.ColorJitter(brightness=0.1)
transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    normalize,
    brightness_change
])

train_data = torchvision.datasets.ImageFolder('/home/std3/yyt/experiment2/dataset/3%', transform=transform)

train_loader = DataLoader(dataset=train_data, num_workers=0, batch_size=batch_size, shuffle=True)

valid_data = torchvision.datasets.ImageFolder('/home/std3/yyt/experiment2/dataset/100%/valid', transform=transform)

valid_loader = DataLoader(dataset=valid_data, num_workers=0, batch_size=batch_size, shuffle=True)

test_data = torchvision.datasets.ImageFolder('/home/std3/yyt/experiment2/dataset/100%/test', transform=transform)
print(len(test_data))

test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=batch_size, shuffle=False)

# build NET
net = models.vgg11_bn(pretrained=False, num_classes=2)

#load weight
pretrain_dict = torch.load("/home/std3/yyt/experiment2/model/vggmodel100.pth")
for k in pretrain_dict.keys():
        print(k)
model_dict = {}
state_dict = net.state_dict()
for k in state_dict.keys():
        print(k)
print("分界线")
for k, v in pretrain_dict.items():
        for i, j in state_dict.items():    #加上前缀后寻找对应的keys
            m = "backbone."+i
            if k == m and 'fc' not in k:
                model_dict[i] = v

state_dict.update(model_dict)
net.load_state_dict(state_dict)
for k in model_dict.keys():
        print(k)
# for k, v in net.named_parameters():
#     if k not in ['fc.weight', 'fc.bias']:
#         v.requires_grad = False
#         print(k)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
net.to(device)


def train_and_valid(model, loss_function, optimizer, epochs):
    history = []
    best_acc = 0.0
    best_epoch = 0
    with open("C3%.txt", "w")as f:
        for epoch in range(epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, epochs))

            model.train()

            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0
            test_loss = 0.0
            test_acc = 0.0
            

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 因为这里梯度是累加的，所以每次记得清零
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                train_acc += acc.item() * inputs.size(0)

            with torch.no_grad():
                model.eval()
                vTP, vTN, vFP, vFN = 0,0,0,0
                for j, (inputs, labels) in enumerate(valid_loader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    re = outputs.argmax(dim=1)

                    for xx in range(len(re)):
                        if re[xx] == 1 and labels[xx] == 1:
                              vTP += 1
                        if re[xx] == 1 and labels[xx] == 0:
                              vFP += 1
                        if re[xx] == 0 and labels[xx] == 0:
                              vTN += 1
                        if re[xx] == 0 and labels[xx] == 1:
                              vFN += 1
                              
                    loss = loss_function(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))


                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    valid_acc += acc.item() * inputs.size(0)

            with torch.no_grad():
                model.eval()
                tTP, tTN, tFP, tFN = 0,0,0,0
                for j, (inputs, labels) in enumerate(test_loader):
                    inputs = inputs.to(device)

                    labels = labels.to(device)

                    outputs = model(inputs)

                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    test_acc += acc.item() * inputs.size(0)

                    re = outputs.argmax(dim=1)

                    for xx in range(len(re)):
                        if re[xx] == 1 and labels[xx] == 1:
                              tTP += 1
                        if re[xx] == 1 and labels[xx] == 0:
                              tFP += 1
                        if re[xx] == 0 and labels[xx] == 0:
                              tTN += 1
                        if re[xx] == 0 and labels[xx] == 1:
                              tFN += 1
                        

                    loss = loss_function(outputs, labels)

                    test_loss += loss.item() * inputs.size(0)
 
            print(vTN,vTP,vFN,vFP)
            vAcc = (vTP+vTN)/(vTP+vTN+vFP+vFN)
            try:
              vPrecision = vTP/(vTP+vFP)
            except:
              vPrecision = 0
            vRecall = vTP/(vTP+vFN)
           
            print(tTN,tTP,tFN,tFP)
            tAcc = (tTP+tTN)/(tTP+tTN+tFP+tFN)
            try:
              tPrecision = tTP/(tTP+tFP)
            except:
              tPrecision = 0
            tRecall = tTP/(tTP+tFN)
            
            avg_train_loss = train_loss / len(train_data)
            avg_train_acc = train_acc / len(train_data)

            avg_valid_loss = valid_loss / len(valid_data)
            avg_valid_acc = valid_acc / len(valid_data)

            avg_test_loss = test_loss / len(test_data)
            avg_test_acc = test_acc / len(test_data)
            # 将每一轮的损失值和准确率记录下来
            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            if best_acc < avg_valid_acc:
                best_acc = avg_valid_acc
                best_epoch = epoch + 1

            epoch_end = time.time()
            # 打印每一轮的损失值和准确率，效果最佳的验证集准确率
            print(
                "Epoch: {:03d},Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation: Loss: {:.4f}, Accuracy: {:.4f}%, vPrecision: {:.4f}%, vRecall: {:.4f}%, Test: Loss: {:.4f}, Accuracy: {:.4f}%, tPrecision: {:.4f}%, tRecall: {:.4f}%".format(
                    epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                    vPrecision * 100, vRecall * 100,avg_test_loss, avg_test_acc * 100, tPrecision * 100, 
                    tRecall * 100
                ))
            f.write("Epoch: {:03d},Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation: Loss: {:.4f}, Accuracy: {:.4f}%, vPrecision: {:.4f}%, vRecall: {:.4f}%, Test: Loss: {:.4f}, Accuracy: {:.4f}%, tPrecision: {:.4f}%, tRecall: {:.4f}%".format(
                    epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                    vPrecision * 100, vRecall * 100,avg_test_loss, avg_test_acc * 100, tPrecision * 100,
                    tRecall * 100
                ))
            f.write('\n')
            f.flush()

            print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    return history




# 开始训练
history = train_and_valid(net, loss_function, optimizer, num_epochs)

