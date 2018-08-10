---
layout: page
title: Models
---
**Model Preparation and Fitting**

## 0. Data Preparation
### Reading and Cleaning Data

#### Cleaning Tweets
First we removed special characters and line returns from our tweets.

```python
#clean tweet
df['cleantweet'] = df['text'].copy()
df['cleantweet'] = df['cleantweet'].replace(r'\n',' ', regex=True)
df['cleantweet'].replace(to_replace=r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", 
    value='',inplace=True,regex=True)
```

Then we add a sentiment score for each tweet.

```python
#add sentiment
df['sentiment'] = [TextBlob(str(row)).sentiment.polarity for row in df['cleantweet']]
```

Most significantly, we had to un-nest any sub-ojects we wanted for a tweet. Each tweet contained a json object with all attributes of the user, retweet, and quote tweet. Each retweet and quote tweet also had user objects. Each tweet (original, retweet, or quote tweet) had entities (which breaks out in hashtags, urls, media), places, and coordinates. We found the places and coordinates to be predominately empty.

```python
#break out nested objects
users = data['user'].apply(pd.Series)

retweets = data['retweeted_status'].apply(pd.Series)
retweet_users = retweets['user'].apply(pd.Series)

quotes = data['quoted_status'].apply(pd.Series)
quote_users = retweets['user'].apply(pd.Series)

entities = data['entities'].apply(pd.Series)
hashtags = entities['hashtags'].apply(pd.Series)
urls = entities['urls'].apply(pd.Series)

place = data['place'].apply(pd.Series)
coordinates = data['coordinates'].apply(pd.Series)

```


### PCA Analysis

We reviewd our data using PCA analysis with two dimensions. These first two dimensions explain 47.1% of the variance in our data. There are interesting sections, but the bots and humans are competely inter-mixed.

![Image](images/PCA_Analysis.png)
 
 
### Baseline
Based on the users we gathered, 2.3% were bots. Our models could guess human 100% of the time and still be 97.7% correct.

### 1) Logistic Regression
### 2) Random Forest
### 3) Ada Boost

![Image](images/adaboost_scores.png)
### 4) Ensemble
