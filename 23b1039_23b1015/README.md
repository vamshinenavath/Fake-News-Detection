# Instructions To Run the Code


## Directory Structure

|-encoded(dir)  
|-models(dir)  
|-vocabs(dir)  
|-datasets(dir)  
|  
|  
|-README  
|-all other code files


Create the directories **models** and **datasets** , Download and place these from the google drive "https://drive.google.com/drive/folders/14Har_LxejVHaMf_sxMpqTThqUUR9CcSU?usp=drive_link"

## mulstage_model.py

This python file has the classes of "CNN+BiLSTM" and a "Pure Fully Connected Neural Network" which are used by other files.

## primary_classifier.ipynb
This is to train the CNN+BiLSTM for binary classification of data.

## secondary_classifer.ipynb
This file does the Multi-Class classification given the news is fake

## test_primary_classifier.ipynb and test_secondary_classifier.ipynb


# Libraries Used

torch version: 2.3.0+cpu  
torchtext version: 0.18.0+cpu  
numpy  
pandas  
matplotlib.pyplot
scikit-learn  
seaborn
pickle
tqdm

