# Toxic Comment Detection Using Transfer Learning Based on BERT
## Abstract

## Realted works

## Models
In this project, I propose four strategies to fine tune BERT Model to detect toxic comments in social media. I did not use the coresponding output of [CLS] token and just used represenattion of the words in related comment that BERT gives us (last hidden layer). In all models, convolution layers is 1 dimensional and output dim of LSTM layers is the same as input dim. Number of kernels of CNN layers if 64 whereever did not mentioned. I set maximum length of sentecne to 36 for in tokenization phase.

### Model 1: BERT+LSTM-CNN
In this model, I give the last hidden layar of BERT to a LSTM and then a 1D-CNN layer. Detecting some toxic comments need sequential learning and some of them just need feature extraction. I propose this and BERT+CNN-LSTM (model 2) models to find both. Reusltof this model is better than BERT+CNN-LSTM.
<p align="center">
<img src="./Pictures/BERT-LSTM-CNN.png" height=500 width=400/>
 </p>
 
### Model 2: BERT+CNN-LSTM
The motivation of this model is the same as model 1, But instead of that, sequential learning starts after feature extraction. 
<p align="center">
<img src="./Pictures/BERT-CNN-LSTM.png" height=500 width=400/>
 </p>
 
### Model 3: BERT+CNN-1D
This model gives the last hidden layer of BERT to CNN for feature extraction. we know to detect almost toxic comments, we hust need feature extraction, but BERT represenation of words in lexicon-semantic vector space can seperate toxic words(phrases) from normal words and these embeddings can help CNN to extracing the features. 
The motivation of proposing this model is the same as BERT+CNN in Related Works section using representation of all layers of BERT and this cause a lot of memory usage. 
<p align="center">
<img src="./Pictures/BERT-CNN-1D.png" height=500 width=400/>
 </p>
 
### Model 4: BERT+2CNN-1D
As you know, I just used last hidden layer of BERT (not all layers) and it means we need to moew explore this embedding matrix to figure out what comment is toxic or not. for this reason I propose model4, in which pooling layer of first CNN layer do not work on whole of corresponding sequence (Current sentence). It works with kernel size. with the help of this technique after one CNN layer we have a matrix and can apply a CNN on it again. 
<p align="center">
<img src="./Pictures/BERT-2CNN-1D.png" height=500 width=400/>
 </p>
