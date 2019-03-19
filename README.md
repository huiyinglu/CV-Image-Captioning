# CV-Image-Captioning
Combine CNN and RNN networks to automatically produces captions, given an input image.

In this project, we design a CNN-RNN encoder/decoder model for automatically generating captions for given input images. The dataset we are using is the Microsoft C*ommon *Objects in COntext (MS COCO) dataset - it's a large-scale dataset for scene understanding. The dataset is commonly used to train and benchmark object detection, segmentation, and captioning algorithms. Each image in the dataset comes with a set of captions such that we can feed them into our CNN and RNN network separately.

The encoder in this project uses the pre-trained ResNet-50 architecture (a CNN network- with the final fully-connected layer removed) to extract features from a batch of pre-processed images. The output is then flattened to a vector, before being passed through a Linear layer to transform the feature vector to have the same size as the word embedding.

The decoder RNN network starts with a word embedding layer which takes the embedded image feature vector from encoder CNN network and the captions corresponding with the image, followed by an LSTM layer, then Linear layer which generates the predicted score (the probability of being a specific word in our vocabulary) for the output caption word.

The training set size is over 3,000, and there are over 9,000 tokens in our vocabulary.

Our model does pretty good job in predicting captions on testing image.
