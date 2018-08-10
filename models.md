---
layout: page
title: Models
---
**Model Preparation and Fitting**

## 0. Data Preparation
### Reading and Cleaning Data

```python
#clean tweet
df['cleantweet'] = df['text'].copy()
df['cleantweet'] = df['cleantweet'].replace(r'\n',' ', regex=True)
df['cleantweet'].replace(to_replace=r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", value='',inplace=True,regex=True)

#add sentiment
df['sentiment'] = [TextBlob(str(row)).sentiment.polarity for row in df['cleantweet']]
```
![Image](images/adaboost_scores.png)


### PCA Analysis

We reviewd our data using PCA analysis with two dimensions. These first two dimensions explain 47.1% of the variance in our data. There are interesting sections, but the bots and humans are competely inter-mixed.

![Image](images/PCA_Analysis.png)
 
 
### Baseline
Based on the users we gathered, 2.3% were bots. Our models could guess human 100% of the time and still be 97.7% correct.

### 1) Logistic Regression
### 2) Random Forest
### 3) Ada Boost
### 4) Ensemble
