from loader import create_batches
import numpy as np
from torchvision.models import alexnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
import math
import time
import pickle
from sklearn.metrics import f1_score

test_log_path = "logs//test"
results_path = "data//results"
writer = SummaryWriter(log_dir=test_log_path)


checkpoint_path = "models/checkpoints/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = torch.nn.CrossEntropyLoss()
metric = MulticlassAccuracy(num_classes=3).to(device)

test_loader = create_batches('test', batch_size=64)

alex_model = alexnet(num_classes=3)
alex_model.features[0] = nn.Conv2d(
    4, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
alex_model.to(device)


def setup_model(model_name, model):
    model_details = torch.load(os.path.join(checkpoint_path, model_name))
    model.load_state_dict(model_details['model_state_dict'])
    del model_details
    model.eval()
    print('File loaded successfully')
    return model


def test(model_name, model, loader):
    with torch.no_grad():
        y_preds, y_acts, losses, sample_lengths, accuracies = [], [], [], [], []
        test_batch_no = 0
        for x, y in iter(loader):
            test_batch_no += 1
            x = x.float().to(device)
            y_pred = model.forward(x)
            y_preds.append(torch.argmax(y_pred, axis=1).cpu().numpy())
            y_acts.append(y.numpy())
            loss = criterion(y_pred, y.to(device).type(torch.int64))
            acc_batch = metric(y_pred, y.to(device).type(torch.int64))
            losses.append(loss.item())
            # If the last sample has less than 64 samples, it gets captured here
            sample_lengths.append(len(y_acts))
            accuracies.append(acc_batch.item())
            writer.add_scalar(f"{model_name}/Loss_batch/test",
                              loss.item(), test_batch_no)
            writer.add_scalar(
                f"{model_name}/Accuracy_batch/test", acc_batch.item(), test_batch_no)
    return y_preds, y_acts, losses, sample_lengths, accuracies


model_evaluations = {}
for file in os.listdir(checkpoint_path):
    print(f'Model {file} initiated')
    model = setup_model(file, alex_model)
    st_time = time.time()
    outputs = test(file, model, test_loader)
    end_time = time.time()
    model_evaluations[file] = (outputs, end_time-st_time)
    del model, outputs
with open(os.path.join(results_path, 'model_evaluations.pkl'), 'wb') as f:
    pickle.dump(model_evaluations, f)


new_results_dict = {}
for model in model_evaluations:
    real_results = model_evaluations[model]
    y_preds, y_acts, losses, sample_lengths, accuracies = real_results[0]
    metrics = metric(torch.tensor(np.hstack(y_acts)),
                     torch.tensor(np.hstack(y_preds)))
    f1_scores = f1_score(
        np.hstack(y_acts), np.hstack(y_preds), average='macro')
    new_results_dict[model] = (metrics.item(), f1_scores)


sorted_scores = sorted(new_results_dict.items(),
                       reverse=True, key=lambda item: item[1][1])
print(dict(sorted_scores))

preferred_model = so
