# Stripformer
Implementation of the paper: [Stripformer: Strip Transformer for Fast Image Deblurring](https://arxiv.org/abs/2204.04627)

## Get started

Clone the repository:
```
git clone https://github.com/Zinni98/Stripformer.git
```
Enter the directory containing the repository:
```
cd Stripformer
```
Make a new virtual environment (Optional):
```
conda create -n stripformer python=3.8
conda activate stripformer
```
Finally install all the dependecies:
```
pip install -r requirements.txt
```

## Configuration parameters
Here is a list of the configuration parameters and their meaning in the [config.py](config.py)
- ```gopro_dir``` expects a string containing the absolute or relative path (make sure that it is starting with "./" if relative) to the GoPro dataset.
- ```save_models_dir``` Directory to save model during training make sure that it ends with ".tar" extension. If the filename passed is something like "/path/to/models/run.tar", during training you will find that two files will be created instead of one:
  -  "/path/to/models/run_best.tar"
  -  "/path/to/models/run_last.tar"

  This saves the best model so far and the model after each epoch. It is important to notice that also information about optimizer, scheduler and last epoch are saved, so the training can be resumed if stopped. Set to ```None``` if you don't want to save the model
- ```load_dir```Directory to be specified if you want to load a already trained (or partially trained) model. Set to ```None``` if you don't want to load the model.
- ```colab_dir``` Directory where the repository is located in google drive when using Google Colab.
- ```pretrain``` Boolean specifying if pretraining should be done. In the paper they pretrain for 3000 epochs with an img_size of 256x256 and train for 1000 epochs with an image size of 512x512
- ```pre_train_epochs``` Number of pretraining epochs
- ```epochs``` Number of training epochs
- ```batch_size``` Batch size **(Suggestion: leave it to 1)**.
- ```accumulation_steps``` Gradient accumulation to save memory optimization step is performed every (batch * acc_step) samples. Can be viewed as the "batch size", i.e. after how many samples the network is updated **(Suggestion: leave it to 8)**.
- ```max_lr``` Maximum learning rate for the scheduler **(Suggestion: leave it to default value)**
- ```min_lr``` Minimum learning rate for the scheduler **(Suggestion: leave it to default value)**
- ```pre_train_img_size```
- ```train_img_size```
- ```test_only``` If True, it only runs one test_stp without training. Recommended to use in conjunction with load_dir
- ```use_wandb``` If True it logs on wandb

## Training:
Before starting training you should tweak the parameters in the [config.py](config.py):
- ⚠️⚠️**Make sure that ```test_only``` is set to false to train the system**⚠️⚠️
- Change the ```gopro_dir``` value to match the path where the dataset is stored. If needed set ```save_models_dir```, ```load_dir``` and ```colab_dir``` (This last one is needed only if running on colab. See configuration parameters above for more details).
- Set the other parameters as desired (Default is fine)
- Run ```python3 main.py```

## Testing
To test the network:
- Change the ```gopro_dir``` value to match the path where the dataset is stored. Set ```load_dir``` and ```colab_dir``` (This last one is needed only if running on colab. See configuration parameters above for more details).
- Set ```test_only``` to **True**.
- Run ```python3 main.py```

# Citations

```
@inproceedings{Tsai2022Stripformer,
  author    = {Fu-Jen Tsai and Yan-Tsung Peng and Yen-Yu Lin and Chung-Chi Tsai and Chia-Wen Lin},
  title     = {Stripformer: Strip Transformer for Fast Image Deblurring},
  booktitle = {ECCV},
  year      = {2022}
}
```
