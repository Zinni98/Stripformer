gopro_dir = "./GOPRO_Large"
saved_models_dir = "./saved_models"
# Put here the directory of the repo if you are running in colab
colab_dir = "/content/gdrive/My Drive/Stripformer/"
epochs = 20
batch_size = 1
max_lr = 1e-4
min_lr = 1e-7
use_wandb = True
# Performing gradient accumulation to save memory
# optimization step is performed every (batch * acc_step) samples
accumulation_steps = 8
