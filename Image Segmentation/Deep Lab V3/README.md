# DeepLab_V3 Image Semantic Segmentation Network

Implementation of the Semantic Segmentation DeepLab_V3 CNN as described at [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf).

# Dataset
Training is done on augmemted Pascal VOC dataset available [here](http://home.bharathh.info/pubs/codes/SBD/download.html) 
The dataset contains 8498 train images and 2857 validation images. The custom_train.txt file contains the name of the images selected for training

# TF Records
Run CreateTfRecord.ipynb to create the tf-record to train the model

# Training
Run Train.ipynb to train th model. Modify the optimizer and epoch suitable for training.I have used a batch size of 8 for training. After the model has been trained, weights get saved.

# Prediction
Run Predict.ipynb to predict on trained model
