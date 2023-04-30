# flake8: noqa
gopro_dir = "/media/dataset/GOPRO_Large"
saved_models_dir = "/media/checkpoint/run4.tar"  # Where to save the model during training. Set to None If don't want to save
colab_dir = "/content/gdrive/My Drive/Stripformer/"  # Put here the directory of the repo if you are running in colab
load_dir = None  # "./saved_models/run.tar"  # Where to find already trained model
pretrain = False
pre_train_epochs = 20
epochs = 1000
batch_size = 1
accumulation_steps = 8  # Performing gradient accumulation to save memory optimization step is performed every (batch * acc_step) samples
max_lr = 1e-4
min_lr = 1e-7
pre_train_img_size = 256
train_img_size = 256
use_wandb = True  # Whether to log training status on Weights and biases, if unsure what this is, set to False
