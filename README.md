# Quora Insincere Questions Classification

This repository presents my submisison for the Kaggle challenge titled 'Quora Insincere Questions Classification'. The goal of this projecy is to build an efficient and reliable model that identifies toxic content on quora. Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers [1].


# Requirements:
1. **Data**: Download train, test data from https://www.kaggle.com/c/quora-insincere-questions-classification/data and save in the main directory.
2. **Embeddings**: Download the 'glove.840B.300d' embeddings from https://www.kaggle.com/c/quora-insincere-questions-classification/data and unzip, save in a new directory called 'embeddings'.
    Your directory structure should look like:  
    * README.md
    * clean_question.py  
    * final_submission.ipynb  
    * embeddings  
        * glove.840B.300d
            * glove.840b.300d.txt
    * train.csv
    * test.csv

3. **Packages**: numpy, keras, pandas, BeautifulSoup, nltk, contractions, ftfy, sklearn, ipython3 notebook.

# File Description:
1. **final_submission.ipynb**: Jupyter Notebook file, that cleans, pre-processes data, builds embedding matrix, trains and tests a Bi-LSTM Neural Network.
2. **clean_question.py**: A class of functions that perform essential pre-processing techniques such as removal of unicode, HTML, tags, etc., on dataframes.

# How to run?
Use Jupyter Notebook to open file 'final_submission.ipynb' and run entire file to build a model, train and classify the cases in 'test.csv'. Classifications are saved as 'submission.csv'.

# References:
1. https://www.kaggle.com/c/quora-insincere-questions-classification
