# Checkpoints

When the main.py trains, it produces checkpoint directories.

Each directory contains:

 - `loss.txt` (maybe), contains the loss tracking data of the last run.
 - `wiki.json` (maybe), points how much wikipedia has been trained.
 - `config.json`, the dimensions needed to know how to run this model.
 - `training.txt`, log of what has been trained to this model, and how much, when, how much time it took.
 - `parameters.pt` - parameters of the model.

Checkpoint directory is created in the beginning of the run.
The loss.txt is being written immediately while the training progresses.
When the checkpoint is saved, the other files are written into it.
