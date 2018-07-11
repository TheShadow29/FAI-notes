Notebooks Done:
1. `Installation.ipynb`: A small guide to setting up fastai and solutions to two common problems.
2. `Using-Pretrained-Pytorch-Models.ipynb`: Using pretrained models with the FastAI library. In particular I explore how to load models from the pretrained repository here https://github.com/Cadene/pretrained-models.pytorch
3. `Using-Forward-Hook-To-Save-Features.ipynb`: Apart from getting the features from the models I also explore how to save them in a bcolz file and to retrieve it after that.
4. `Using-CAM-for-CNN-Visualization.ipynb`: Using Class Activation Maps for visualization and localization. It assumes that the model has an adaptive average pooling layer followed by a linear layer and then a softmax.
5. `Visdom-With-FastAI.ipynb`: Includes how to port visdom with fastai using callbacks. Also refer to https://github.com/Pendar2/fastai-tensorboard-callback for tensorboardX visualization. 
6. `Using-Sampler-For-Class-Imbalance.ipynb`: A small dive into how the sampler works in pytorch. Not tightly related to fastai in particular though.
