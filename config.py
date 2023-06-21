# flake8: noqa
gopro_dir = "/media/dataset/GOPRO_Large"
save_models_dir = "/media/checkpoint/run6.tar"  # Where to save the model during training. Set to None If don't want to save
load_dir = "/media/checkpoint/run4.tar"  # Where to find already trained model
colab_dir = "/content/gdrive/My Drive/Stripformer/"  # Put here the directory of the repo if you are running in colab
pretrain = False
pre_train_epochs = 3000
epochs = 1000
batch_size = 1
accumulation_steps = 8  # Performing gradient accumulation to save memory optimization step is performed every (batch * acc_step) samples
max_lr = 1e-4
min_lr = 1e-7
pre_train_img_size = 256
train_img_size = 512
test_only = False
use_wandb = False  # Whether to log training status on Weights and biases, if unsure what this is, set to False