# Sarcasm Detection

## Contents

- [Introduction](#Introduction)
- [Data](#Data)
- [Notebook Descriptions](#Notebook-Descriptions)
- [Results and Conclusions](#Results-and-Conclusions)
- [Future Enhancements](#Future-Enhancements)
- [Executive Summary](#Executive-Summary)
- [References](#References)

## Introduction

Sarcasm detection is a very specialized study subject in NLP, a type of sentiment analysis where the focus is on recognizing sarcasm rather than detecting a sentiment throughout the entire spectrum. As a result, the goal of this field is to determine whether or not a particular text is sarcastic.
    
The first issue we have is that, unlike sentiment analysis, where sentiment categories are clearly defined , the limits of sarcasm are not so well defined. It's also critical to understand what sarcasm is before attempting to detect it.
    
[Cambridge dictionary](https://dictionary.cambridge.org/us/dictionary/english/sarcasm) defines it as: 
>the use of remarks that clearly mean the opposite of what they say, made in order to hurt  someone's feelings or to criticize something in a humorous way.
    
Sarcasm is subjective. Non-native speakers/readers may not get it. So it is a use case for sarcasm detection and was my motive to take up this project.

## Project Directory

project-6
|__ code
|   |__ 01_Musicals_Collection.ipynb
|   |__ 02_Synopsis_Collection.ipynb
|   |__ 03_Summary_Collection.ipynb
|   |__ 04_Data_Concatenation.ipynb
|   |__ 05_EDA_and_Cleaning.ipynb
|   |__ 06_Sentiment_Analysis.ipynb
|   |__ 07_Recommender_System.ipynb
|__ assets
|   |__ spacy_architecture.png
|   |__ spacy_cosine_similarity.png
|   |__ musicals.jpg
|   |__ prompt_1_input.png
|   |__ prompt_1_output.png
|   |__ prompt_2_input.png
|   |__ prompt_2_output.png
|__ data
|   |__ musical_names.csv
|   |__ musical_synopses.csv
|   |__ musical_summaries.csv
|   |__ musical_data.csv
|   |__ musical_vectors.csv
|   |__ dbscan_labels.csv
|   |__ kmeans_labels.csv
|   |__ musical_data_vectors_labels.csv
|   |__ musical_sentiments.csv
|   |__ musical_for_app.csv
|__ static
|   |__ css
|   |__ |__ style.css
|__ templates
|   |__ form.html
|   |__ results.html
|__ app.py
|__ showmetunes.py
|__ capstone_presentation.pdf
|__ README.md
## Notebook Descriptions

1. 01_EDA and Preprocessing of headlines.ipynb : Carried out EDA and preprocessing in this notebook
2. 02_Vectorization(NLP).ipynb : vectorised the preprocessed headlines in this notebook using CountVectorizer()
3. 03_Model-Building_Select_K_Best.ipynb : Modeled randomforestclassifier and logistic regression algorithm on the processed data in this notebook by slecting top  100 features
4. 04_Model_Building_with_PCA.ipynb : Modeled randomforestclassifier on data in this notebook after transforming it with PCA
5. 05_BERT.ipynb : implemented a basic BERT model in this notebook

**Software Recommendations**
Pandas, Plotly, Sci-Kit Learn, Numpy, MatPlotLib, Seaborn,Keras
 
## Results and Conclusions 

1. Baseline is at around 56%.
2. RandomForestClassifier gave an accuracy of 65%
3. Logistic regression gave an accuracy of 62.5%
4. num_punctuations as identified earlier was an important predictor of sarcasm level of a text.
5. BERT model is contextual and hence apt for text classification and performed best.
6. PCA is not really useful for text classification task.

## Future Enhancements

I would like to explore more and better deep learning methods more to increase the prediction accuracy and work on generalizing the results to any text. 

## Executive Summary

A dataset consisting of news headlines was used to model in this project. There are two sources of the headlines-theonion and huffingtonpost. Since these are written by professionals, spelling errors are likely to be very less. Theonion is known for their sarcastic headlines here too all sarcastic headlines came from theonion and rest came from huffingtonpost. An exploratory data analysis was done on the dataset and certain new metafeatures were created to be used for 
modeling. Then the headlines were vectorized and then used in two models namely-RandomForestClassifier and Logistic regression. The scores for classical ML methods are quite low at around 62%- 65%. So I used PCA on the vectorized data to check for the possibility of better prediction accuracy. However, not much was achieved
in this process. This is because these methods are not accounting for the context in the text. So using a method which takes into account the
context would likely give better accuracy scores. So I implemented a basic BERT model. BERT is bidirectional and this characteristic allows the model to learn the context of a word based on all of its surroundings. BERT produced best scores among all models. So it is recommended as the production model.

## Data Collection

I am going to the news headlines dataset for this task which was found on Kaggle https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection.
Each record consists of three attributes:

-`is_sarcastic`: 1 if the record is sarcastic otherwise 0
- `headline`: the headline of the news article
- `article_link`: link to the original news article. Useful in collecting supplementary data

## EDA
It was found that the dataset is fairly balanced but not perfectly balanced so we will need to stratify for target as seen in the following graph.

![](./images/source.png)

## Data Processing
The data to be processed is textual. Various NLP libraries were used to process the data namely,
- [spaCy](https://spacy.io/): most of the processing (extracting important words, word vectorization/embedding)
- [Regex](https://docs.python.org/3/library/re.html): removing punctuation
- [nltk](https://www.nltk.org/): removing stopwords, tokenizing
- [Sci-kit learn](https://scikit-learn.org/): CountVectorizer

spaCy had POS(Parts of Speech) functionality to offer, which was used to create metafeatures representing the proportions of various POS in a sentence. 
Regex was used to find patterns of text and 

## References

1. https://medianism.org/2015/01/08/sarcasm-marks-we-dont-do-no-stinking-sarcasm/m
2. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270    
3. https://www.analyticsvidhya.com/blog/2021/12/text-classification-using-bert-and-tensorflow/ 
4. https://towardsdatascience.com/multi-label-text-classification-using-bert-and-tensorflow-d2e88d8f488d


