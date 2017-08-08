# Hello Tensorflow

This is a small tensorflow project designed to be as simple as possible to demonstrate some of the tensorflow concepts.

The problem it's solving is the classic `XOR` problem, which is the "Hello World" of neural networks.

## About The Code

There is a function called `generate_xor` which will generate inputs and outputs for an XOR gate, this will allow us to train and test our model.

The model will take a 1 x 2 input, eg. [0,1]. There will be a hidden layer of size 2 x 10, and then an output layer which will produce a 1 x 1 output. The activation function that will be used is a `sigmoid function`. This is the best activation function to use for binary classification.

The cost that we are minimizing is the absolute difference between the output and the truth. You might want to alter the code to use cross entropy instead.
