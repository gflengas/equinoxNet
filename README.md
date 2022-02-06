# Convolution Neural Network Training in C 
This is the 2nd part of my thesis work and continuation of [Convolution-Neural-Network-Training](https://github.com/gflengas/Convolution-Neural-Network-Training).

The current network for CNN training with Mini-Batch Gradient Descent is developed on C and 
trained with the MNIST data set. It consists of a Convolutional layer with 12 3×3 filters 
initialized with He initialization using Relu as activation function, a Maxpool layer with pool 
size 2, and a fully connected layer using SoftMax as activation function. To measure the 
accuracy of the network, the Categorical Cross-Entropy Loss function is used. The training 
parameters that were used for the results presented are: batch size= 4, learning rate=0.01, 
momentum 0.9, decay 0.00005, epoch 20 and iterations 2000.

![network image](https://github.com/gflengas/equinoxNet/blob/main/flow%20of%20simple%20cnn.jpg)

Small cnn training running on MNIST 
Results:

------Epoch :1-----
average loss: 0.300562 accuracy: 90.56 %
Conv - avg_val = -0.003535
fc - avg_val = 0.000005
------Epoch :2-----
average loss: 0.197186 accuracy: 93.87 %
Conv - avg_val = -0.008266
fc - avg_val = 0.000005
------Epoch :3-----
average loss: 0.150811 accuracy: 95.36 %
Conv - avg_val = -0.015109
fc - avg_val = 0.000005
------Epoch :4-----
average loss: 0.122404 accuracy: 96.25 %
Conv - avg_val = -0.018762
fc - avg_val = 0.000005
------Epoch :5-----
average loss: 0.102462 accuracy: 96.88 %
Conv - avg_val = -0.019427
fc - avg_val = 0.000005
------Epoch :6-----
average loss: 0.087829 accuracy: 97.35 %
Conv - avg_val = -0.016657
fc - avg_val = 0.000005
------Epoch :7-----
average loss: 0.077108 accuracy: 97.69 %
Conv - avg_val = -0.017337
fc - avg_val = 0.000005
------Epoch :8-----
average loss: 0.068257 accuracy: 97.97 %
Conv - avg_val = -0.015872
fc - avg_val = 0.000005
------Epoch :9-----
average loss: 0.061224 accuracy: 98.19 %
Conv - avg_val = -0.013994
fc - avg_val = 0.000005
------Epoch :10-----
average loss: 0.055467 accuracy: 98.37 %
Conv - avg_val = -0.013158
fc - avg_val = 0.000005

Time elapsed: 313.28600
