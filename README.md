# Toxic Comment Detection Using Transfer Learning Based on BERT
In this project, I propose four strategies to fine tune BERT Model to detect toxic comments in social media. In all models, convolution layers is 1 dimensional and output dim of LSTM layers is the same as input dim. Number of kernels of CNN layers if 64 whereever did not mentioned.
## Model 1: BERT+LSTM-CNN
in this model, I give the last hidden layar of BERT to a LSTM and then a 1D-CNN layer. Detecting some toxic comments need sequential learning and some of them just need feature extraction. I propose this and BERT+CNN-LSTM (model 2) models to find both. Reusltof this model is better than BERT+CNN-LSTM  
<p align="center">
<img src="./Pictures/BERT-LSTM-CNN.png" height=500 width=400/>
 </p>
 
## Model 2: BERT+CNN-LSTM
<p align="center">
<img src="./Pictures/BERT-CNN-LSTM.png" height=500 width=400/>
 </p>
 
## Model 3: BERT+CNN-1D
<p align="center">
<img src="./Pictures/BERT-CNN-1D.png" height=500 width=400/>
 </p>
 
## Model 4: BERT+2CNN-1D
<p align="center">
<img src="./Pictures/BERT-2CNN-1D.png" height=500 width=400/>
 </p>
