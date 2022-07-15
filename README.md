# Sarcasm Detection

## Contents

- [Introduction](#Introduction)
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

**I am going to use a very popular news headlines dataset for this task.**

## Notebook Descriptions

1. EDA and Preprocessing of headlines.ipynb : Carried out EDA and preprocessing in this notebook
2. Vectorization(NLP).ipynb : vectorised the preprocessed headlines in this notebook using CountVectorizer()
3. Model-Building_Select_K_Best.ipynb : Modeled randomforestclassifier and logistic regression algorithm on the processed data in this notebook by slecting top        100 features
4. Model_Building_with_PCA.ipynb : Modeled randomforestclassifier on data in this notebook after transforming it with PCA
5. BERT.ipynb : implemented a basic BERT model in this notebook

**Software Recommendations**
Pandas, Sci-Kit Learn, Numpy, MatPlotLib, Seaborn,Keras
 
## Results and Conclusions 

1. Baseline is at around 56%.
2. RandomForestClassifier gave an accuracy of 65%
3. Logistic regression gave an accuracy of 62.5%
4. num_punctuations as identified earlier was an important predictor of sarcasm level of a text.
5. BERT model is contextual and hence apt for text classification and performed best.
6. PCA is not really useful for text classification task.

## Future Enhancements

I would like to explore deep learning methods more to increase the prediction accuracy and work on generalizing the results to any text. 

## Executive Summary

A dataset consisting of news headlines was used to model in this project. There are two sources of the headlines-theonion and huffingtonpost. Since these are written by professionals, spelling errors are likely to be very less. Theonion is known for their sarcastic headlines here too all sarcastic headlines came from theonion and rest came from huffingtonpost. An exploratory data analysis was done on the dataset and certain new metafeatures were created to be used for 
modeling. Then the headlines were vectorized and then used in two models namely-RandomForestClassifier and Logistic regression. The scores for classical ML methods are quite low at around 62.5%- 65%. So I used PCA on the vectorized data to check for the possibility of better prediction accuracy. However, not much was achieved
in this process. This is because these methods are not accounting for the context in the text. So using a method which takes into account the
context would likely give better accuracy scores. So I will try to implement a basic BERT. BERT is bidirectional and this characteristic allows the model to learn the context of a word based on all of its surroundings. BERT produced best scores among all models. So I recommend this as the production model.

## References

1. https://medianism.org/2015/01/08/sarcasm-marks-we-dont-do-no-stinking-sarcasm/m
2. https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270    
3. https://www.analyticsvidhya.com/blog/2021/12/text-classification-using-bert-and-tensorflow/ 
4. https://towardsdatascience.com/multi-label-text-classification-using-bert-and-tensorflow-d2e88d8f488d


