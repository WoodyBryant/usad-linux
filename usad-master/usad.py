import torch
import torch.nn as nn

from utils import *
device = get_default_device()
##nn.Linear主要用来设置全连接层，全连接层的输入是二维张量(batch_size,size)
##nn.Linear(in_features,out_features)，第一个参数是输入的二维张量的大小，即(bartch_size,size)中的size,
##第二个参数是指输出二维张量的大小，即(bartch_size,out_size)中的out_size，也代表了全连接层的神经元的个数。
##标准的神经网络的结构
class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
##nn.Module是所有神经网络的基类，所有神经网络模型都要继承这个类
##UsadModel定义了一个编码器encoder和两个解码器decoder，他们都是由全连接层组成的。他们组成了两个自编码器
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
  ##两个自编码器的训练过程
  ##AE1的loss1要最小化AE1的重组误差和AE2(AE1(W))和W的重组误差
  ##AE2的loss2要最小化AE2的重组误差和最大化AE2(AE1(W))和W的重组误差
  ##但是计算loss这块有点过于简单
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    ##这里用来返回验证集的最大异常分数
    anomaly_score = 0.5*torch.mean((batch-w1)**2)+0.5*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2,'anomaly_score':anomaly_score}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    ##torch.stack,将若干个张量在维度上连接，比如原来有几个2维张量，连接后就可以得到3维张量
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    
    batch_anomaly_score = [x['anomaly_score'] for x in outputs]
    epoch_anomaly_score = torch.stack(batch_anomaly_score).max()
    
    ##item是为了得到元素张量里面的元素值，就是将一个零维张量转换为浮点数，特别是计算loss和accuracy的时候
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item(),'anomaly_score':epoch_anomaly_score.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)
##训练，优化函数采用了Adam，它使用了梯度的一阶矩估计和二阶矩估计
def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    ##将两个自编码器的参数都加入优化队列，构成两个优化队列
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            ##将每一个batch存入设备
            batch=to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            ##梯度反向传播
            loss1.backward()
            ##更新所有的参数
            optimizer1.step()
            ##情况过往梯度
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        ##得到单次迭代后验证集的loss1和loss2
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
##验证阶段，两个误差加起来求异常分数
def testing(model, test_loader, alpha=.5, beta=.5):
    results=[]
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        results.append(alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1))
    return results