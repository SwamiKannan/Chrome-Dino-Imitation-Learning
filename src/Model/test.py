from loader import create_batches
import numpy as np
from torchvision.models import alexnet
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
import time
import pickle

from sklearn.metrics import f1_score
from sklearn.metrics._classification import classification_report

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
    '''Setting up the model with all the weights loaded and setting the model mode to eval
    Args:
        model_name (string): Name of the checkpoint whose model weights need to be loaded
        model (torchvision.model): Base torchvision model being used for testing
    Returns:
        model (torchvision.model): Torchvision model with weights loaded and set to eval mode
    '''
    model_details = torch.load(os.path.join(checkpoint_path, model_name))
    model.load_state_dict(model_details['model_state_dict'])
    del model_details
    model.eval()
    print('File loaded successfully')
    return model


def test(model_name, model, loader):
    '''
    Run the model in test model.
    For each 64-sample batch, calculate the loss and the accuracies 
    Return the same along with the predicted values and actual values
    Args:
        model_name (str) : Name of the file which has the model states
        model (torch.model) : The actual model to be loaded
        loader (torch.DataLoader): The PyTorch DataLoader object which maps the test data for loading in batches
    Returns:
        y_preds (list) : List of Torch tensors containing the predicted values of the batch
        y_acts (list) : List of Torch tensors containing the actual values of the batch
        losses (list) : List of Torch tensor variable that represents the loss of each batch
        sample_lengths (list) : List of the batch_size. Should be 'batch_size' parameter given to the loader object for all batches except the last one (Unless the total number of test samples is an exact multiple of the batch_size)
        accuracies (list) : List of Torch tensor variables; each representing the calculated accuracies of one batch
    '''
    model.eval()
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


# For each model get the accuracy for all datapoints
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

# Get f1 scores for all models in the checkpoint folder
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

# Get classification report of preferred model
preferred_model = sorted_scores[0][0]
preferred_model_results = model_evaluations[preferred_model]


y_preds, y_acts, losses, sample_lengths, accuracies = preferred_model_results[0]
final_y_preds = np.hstack(y_preds)
final_ys = np.hstack(y_acts)
print(accuracies)

print(classification_report(final_ys, final_y_preds))

# The output diagram 'heatmap.png' in the repo was generated by the code here: https://stackoverflow.com/a/34304414
