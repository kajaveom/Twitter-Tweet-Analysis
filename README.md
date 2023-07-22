# Twitter-Tweet-Analysis
This competition is available on kaggle .The link of the same is-https://www.kaggle.com/competitions/nlp-getting-started/overview

The team members are:-

Ashish Choudhary-20BRS1234

Om Kajave-20BRS1130

Pulak Jain-20BRS1126

Rishabh Jain-20BRS1065



METHODOLOGY 

1)Data Description and Pre-Processing

The general preprocessing steps are:

A.Converting the text to lowercase
B. Removing stop words, punctuations, URLs, numeric numbers, and special characters from
the tweets
C. Tokenize each text
Tokenize is essentially splitting a phrase, sentence, paragraph, or entire text document
into smaller units, such as individual words or terms
D. Applying stemming
E.Encoding is performed on the data
F. Converting the text into a sequence and performing padding

2)Deep Learning algorithms

The following Deep learning algorithms are applied and then they are compared to find which performs better.

A)TCN

Temporal Convolutional Networks, or simply TCN is a variation over Convolutional Neural Networks for sequence modeling tasks.

<img width="463" alt="image" src="https://user-images.githubusercontent.com/125439405/227759445-04fc0bd9-f948-4785-b414-ebf5e38cdcc3.png">

The encoded & padded data is fed into the model through the input layer which is further connected to the EMBEDDING layer. The Embedding layer converts data of high
dimension into lower dimension allowing the network to learn more about the relationship between inputs and to process the data more efficiently. SpatialDropout1D helps in promoting independence between feature maps, thus enhancing the regularization of activations.TCNs are able to extract long-term patterns using dilated causal convolutions and residual blocks. TCN also allows for parallel computation of outputs which can also be more efficient in terms of computation time. The dilations of [1,2,4] are used to enlarge the receptive field while maintaining resolution. The output of two stacked TCN layers is concatenated after performing global average & global max pooling in order to extract max and average features from the feature map.The concatenation layer output is produced as an input to the hidden dense network which compromises L2 regularization with 0.1 as the shrinkage coefficient in order to avoid overfitting. The dropout layer with an optimal value of 0.4 is also used for the same. Another Dense hidden layer with 16 neurons is used with an activation function as relu, followed by a dropout layer and output layer. The output layer provides the probability of classes using the sigmoid activation function. The model was trained on a batch size of 128 with a learning rate of 0.0000001. Adam was used to optimize the model. The kernel size of 3 is recognized as better. The model achieved an accuracy of 60% on test data. The learning was improved by decreasing the learning rate.

B)BERT

BERT, which stands for Bidirectional Encoder Representations from Transformers, is based on Transformers, a deep learning model in which every output element is connected to every input element, and the weightings between them are dynamically calculated based on their connection.

<img width="457" alt="image" src="https://user-images.githubusercontent.com/125439405/227759538-8cd94bfa-e159-45be-8fcb-4d596956c47d.png">

The BERT architecture consists of several layers of Transformers. The input to the model is a sequence of tokens, such as words or sub-words, and the output is a sequence of hidden states, which capture the contextual information of each token.The input tokens(text related to disaster) are first mapped to embedding vectors using a combination of token embeddings and segment embeddings. Token embeddings represent the meaning of each individual token, while segment embeddings represent which sentence the token belongs to in the case of multi-sentence input. The input embeddings are passed through a stack of Transformer layers/Encoder Layers. Each Transformer layer consists of a self-attention mechanism and a feed-forward neural network. The self-attention mechanism allows the model to capture the relationships between different tokens in the input sequence, while the feed-forward network allows the model to process the information learned from the attention mechanism. The dropout is applied to encoded output as dropout can prevent overfitting by randomly dropping out neurons during training, which helps to prevent the
network from relying too much on specific neurons and their connections. This encourages the network to learn more robust features and reduces the risk of overfitting. Therefore it is passed to the output layer with sigmoid as an activation function for binary classification of whether a tweet is pertaining to disaster or not.

C)LSTM

LSTM stands for long short-term memory networks, it is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, especially in sequence prediction problems. LSTM has feedback connections, i.e., it is capable of processing the entire sequence of data, apart from single data points such as images.

<img width="352" alt="image" src="https://user-images.githubusercontent.com/125439405/227759616-05c90d5b-436a-4ada-8343-56a3d0cb1846.png">

Here before the LSTM network, PCA is used to reduce the dimensionality of the word
embeddings. PCA works by finding the principal components that explain the most variance
in the data and projecting the data onto these components. The number of principal
components to keep can be determined by analyzing the variance explained by each
component and choosing a suitable threshold. The components calculated here for each of the
tweets are text length, capital letter rate, digit rate, and different attributes for special
characters that occurred in tweet texts. After transforming the components, for each training
and testing, they are stored in a separate variable. The original train text first undergoes a
vectorization process, after which they are sequenced. Then these are padded to maintain the
length of the train sequences of the same size.
The pre-processed text is now concatenated with the training data of components
carried out by PCA, similarly with test data. The training data is divided further for training
and validation. An embedding matrix is created using a glove file, which s used in LSTM to
give the input data a better representation, which can enhance the LSTM's performance.
Words with similar meanings have vectors that are closer together in the vector space as a
result of the embedding, which converts each word into a high-dimensional vector. As a
result, the LSTM can catch semantic associations between words more accurately and
generalize to words it hasn't encountered before.
The glove branch and input component branch are created separately. The embedding
matrix is given as weight for glove embedding, where the input components are connected to
a dense layer with LeakyRelu and a dropout. Finally, the output dense layer concatenates the
LSTM network and component layer. A model is created using these inputs and the output
layer respectively. Adam optimizer is used and f1Score as a metric. Also callbacks such as
Early-Stopping and Reduce Learning Rate on Plateau are used while fitting the model to
avoid overfitting. Once the model is fitted, evaluation is performed on the testing dataset with
a threshold of 0.5 to classify a tweet.

3)Results and Discussions

The summarized results of our proposed method based on different models are
presented .

<img width="402" alt="image" src="https://user-images.githubusercontent.com/125439405/227759664-d3e4e502-afa1-440e-9194-a0f61792ca79.png">

According to the accuracy graphs, the LSTM model has the highest accuracy with an estimated score of 0.80815. Thus, we draw the conclusion that the LSTM model is the most appropriate for categorising tweets as disaster-free or disaster-related
