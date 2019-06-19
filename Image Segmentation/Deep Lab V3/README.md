# DeepLab_V3 Image Semantic Segmentation Network

Implementation of the Semantic Segmentation DeepLab_V3 CNN as described at [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf).

# Dataset
Training is done on augmemted Pascal VOC dataset available [here.](http://home.bharathh.info/pubs/codes/SBD/download.html) 
The dataset contains 8498 train images and 2857 validation images. The custom_train.txt file contains the name of the images selected for training for benchmarking. In this implementation however, training is done on the split provided in the augmented Pascal VOC dataset. 

# TF Records
Run CreateTfRecord.ipynb to create the tf-record to train the model

# Training
Run Train.ipynb to train th model. Modify the optimizer and epoch suitable for training. Batch size of 8 is used for training. DeepLab V3 takes image dimension of 513 x 513. After the model has been trained, weights get saved as a checkpoint file.

# Prediction
Run Predict.ipynb to predict on trained model
