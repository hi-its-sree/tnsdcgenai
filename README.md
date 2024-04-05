Fake News Detection using LSTM
This repository contains code for detecting fake news using Long Short-Term Memory (LSTM) networks. The LSTM model is implemented using TensorFlow and Keras. The model is trained on a dataset containing news articles labeled as either real or fake.

Dataset
The dataset used for training the model is stored in a CSV file named train.csv. It contains two columns:

news: This column contains the text of the news articles.
label: This column contains the labels indicating whether the news article is real or fake.
Before running the code, ensure that you have the dataset stored at the specified location or modify the code to load the dataset from a different location.

Dependencies
Ensure you have the following dependencies installed before running the code:

Pandas
TensorFlow
NLTK (Natural Language Toolkit)
You can install these dependencies using pip:

Copy code
pip install pandas tensorflow nltk
Running the Code
Load the dataset:

python
Copy code
import pandas as pd
df = pd.read_csv('/content/train.csv')
Preprocess the text data:

python
Copy code
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

ps = PorterStemmer()
corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['news'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
Convert text data to numerical representations:

python
Copy code
from tensorflow.keras.preprocessing.text import one_hot

voc_size = 5000
messages = corpus.copy()
Build the LSTM model:

python
Copy code
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=20))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
Train the model:

python
Copy code
import numpy as np
X_final=np.array(one_hot(messages,voc_size))
y_final=np.array(df['label'])
model.fit(X_final,y_final,epochs=10,batch_size=64)
Evaluate the model or make predictions on new data.

Acknowledgments
The dataset used in this project is sourced from [provide source if applicable].
This project is for educational purposes and aims to demonstrate the implementation of an LSTM model for fake news detection.
Feel free to experiment with the code, dataset, and model architecture to improve the accuracy of fake news detection. If you have any questions or suggestions, please feel free to contact [author/contact information].





