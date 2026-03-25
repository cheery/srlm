# Training

Training proceeds in a single loop that interleaves (round robin) different training programs.
Each training program is instantiated from a class that has `__call__` -method that contains a training step.

Training ends when all of the training programs are completed.

Types of training programs:

 * Kalevala - takes a randomized sample from kalevala and trains with it.
              usually runs indefinitely but can be set to run fixed number of steps.

 * Wikipedia - loads wikipedia content and trains with it.
               runs for a fixed number of steps, but can also run for an epoch.

 - global step is counted, and every nth. global step a checkpoint is saved.
 - loss.txt gets every training step into it.
 - Every nth. step the loss is being averaged and reported.
