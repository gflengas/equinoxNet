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

average loss: 0.370688 accuracy: 88.64 %

Conv - avg_val = -0.010797

fc - avg_val = 0.000005

------Epoch :2-----

average loss: 0.271359 accuracy: 91.69 %

Conv - avg_val = -0.020622

fc - avg_val = 0.000005

------Epoch :3-----

average loss: 0.220453 accuracy: 93.33 %

Conv - avg_val = -0.029476

fc - avg_val = 0.000005

------Epoch :4-----

average loss: 0.193414 accuracy: 94.13 %

Conv - avg_val = -0.032782

fc - avg_val = 0.000005

------Epoch :5-----

average loss: 0.174565 accuracy: 94.73 %

Conv - avg_val = -0.035303

fc - avg_val = 0.000005

------Epoch :6-----

average loss: 0.158270 accuracy: 95.21 %

Conv - avg_val = -0.037884

fc - avg_val = 0.000005

------Epoch :7-----

average loss: 0.147352 accuracy: 95.55 %

Conv - avg_val = -0.042872

fc - avg_val = 0.000005

------Epoch :8-----

average loss: 0.136984 accuracy: 95.87 %

Conv - avg_val = -0.039950

fc - avg_val = 0.000005

------Epoch :9-----

average loss: 0.129570 accuracy: 96.13 %

Conv - avg_val = -0.047734

fc - avg_val = 0.000005

------Epoch :10-----

average loss: 0.122864 accuracy: 96.33 %

Conv - avg_val = -0.050384

fc - avg_val = 0.000005

Time elapsed: 313.28600
