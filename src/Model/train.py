from torchvision.models import alexnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
import math

from loader import path, create_batches


checkpoint_path = "..//..//models/checkpoints/"
log_train_path = "..//..//logs//train"
log_val_path = "..//..//logs//val"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = alexnet(num_classes=3)
model.features[0] = nn.Conv2d(4, 64, kernel_size=(
    11, 11), stride=(4, 4), padding=(2, 2))

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
train_writer = SummaryWriter(log_dir=log_train_path)
val_writer = SummaryWriter(log_dir=log_val_path)
metric = MulticlassAccuracy(num_classes=3).to(device)
batch_size = 32

model.to(device)

train_loader = create_batches('train', batch_size=batch_size)
val_loader = create_batches('val', batch_size=batch_size)


train_losses = []
train_accuracies = []
train_batch_no = 0

eval_losses = []
eval_accuracies = []
eval_batch_no = 0

for e in range(10):
    print(f'\n\n********EPOCH {e} ********')
    model.train()
    initial = True
    b = 0
    b_eval = 0
    for x, y in iter(train_loader):
        x = x.float().to(device)
        y_pred = model.forward(x)
        loss = criterion(y_pred, y.to(device).type(torch.int64))
        optim.zero_grad()
        loss.backward()
        optim.step()
        b += 1
        train_batch_no += 1
        acc_batch = metric(y_pred, y.to(device).type(torch.int64))
        if initial:
            print(
                f'Epoch {e}\tbatch {b} / {len(train_loader)}\tLoss:{loss}\tAccuracy:{acc_batch}')
            initial = False
        train_writer.add_scalar(
            "Loss_batch/train", loss.item(), train_batch_no)
        train_writer.add_scalar("Loss_accuracy/train",
                                acc_batch.item(), train_batch_no)
        if b % 100 == 0:
            print(
                f'Epoch {e}\tbatch {b} / {len(train_loader)}\tLoss:{loss.item()}\tAccuracy:{acc_batch.item()}')
        if b % 1000 == 0:
            PATH = os.path.join(checkpoint_path, str(e) +
                                '_'+str(b)+'_'+'model.pt')
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss,
            }, PATH)
    for g in optim.param_groups:
        g['lr'] = max(0.001, g['lr']*0.5)
    train_writer.add_scalar("Loss_epoch/train", loss.item(), e)
    train_writer.add_scalar("Accuracy_epoch/train", acc_batch.item(), e)
    PATH = os.path.join(checkpoint_path, str(e)+'_'+str(b)+'_'+'model.pt')
    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'loss': loss,
    }, PATH)
    print(f'\n********Training for epoch {e} complete********')
    print(f'Epoch: {e}\tLoss:{loss.item()},\tAccuracy:{acc_batch.item()}')
    train_losses.append(loss.item())
    train_accuracies.append(acc_batch.item())
    model.eval()
    with torch.no_grad():
        for x, y in iter(val_loader):
            x = x.float().to(device)
            y_pred = model.forward(x)
            eval_loss = criterion(y_pred, y.to(device).type(torch.int64))
            eval_batch_no += 1
            eval_acc_batch = metric(y_pred, y.to(device).type(torch.int64))
            val_writer.add_scalar(
                "Loss_batch/eval", eval_loss.item(), eval_batch_no)
            val_writer.add_scalar("Loss_accuracy/train",
                                  eval_acc_batch.item(), eval_batch_no)
            if b_eval % 100 == 0:
                print(
                    f'Eval Epoch {e}\tbatch {b} / {len(val_loader)}\tLoss:{eval_loss.item()}\tAccuracy:{eval_acc_batch.item()}')
    val_writer.add_scalar("Loss_epoch/eval", eval_loss.item(), e)
    val_writer.add_scalar("Accuracy_epoch/eval", eval_acc_batch.item(), e)
    print(f'\n********Validation for epoch {e} complete********')
