# Performance Logs
### Errata: Loss accuracy is an incorrect title for the second chart. It is "Accuracy/batch"
<p align="center">
<a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/train_logs/images/Accuracy.png"><img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/train_logs/train_accuracy.png"></a>
<a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/train_logs/images/Loss.png"><img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/train_logs/train_loss.png">
</p>

# Validation logs
### Errata: Loss accuracy is an incorrect title for the second chart. It is "Accuracy/batch"
<p align="center">
<a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/val_logs/images/Accuracy.png">
  <img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/val_logs/val_accuracy.png">
</a>

<a href="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/val_logs/images/Loss.png">
  <img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/val_logs/val_loss.png">
</a>
</p>

# Testing logs (for all models)

### Note:
During training, checkpoints for the model weights were created every 1000 batches and at the end of each epoch. Hence, the model name syntax is:
<b><epoch_name>_<batch_name>_model.pt </b><br>
e.g. <br>
0_1000_model.pt refers to the model parameters on the 0th epoch after a 1000 batches were trained.
<p align="center">
  <img src="https://github.com/SwamiKannan/Chrome-Dino-Imitation-Learning/blob/main/logs/test_logs/test_all_models.png">
</p>

