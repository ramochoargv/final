---
layout: page
title: Introduction and EDA
---

## 1. Introduction and Description of Data

We used the search function of the Twitter API to create a comprehensive dataset of tweet samplings, which we save to a PostgreSQL database. 

We took our sampling of tweets and compiled the most recent 20 tweets for all users. Having more data from each user allowed us to initally have better analytics for user-specific behavior (e.g., what percent of a user’s tweets are retweets). That data set ended up being quite large, so we chose the tweets of 1,000 users for our data set.

We then retrieved a botornot score from the botometer api for our 1,000 users.


**Bold**
_Italic_ 
`Code` text
```markdown
Syntax highlighted code block

```

### 2) Description of Raw Data

This data is one row per tweet, though objects such as users, retweets, and quote tweets are nested within a tweet as a dictionary. This is the biggest challenge of this data. It is naturally nested in structure, since a tweet has many objects, and can originate from a retweet, which also has many objects. Most of the data preparation time was spent flattening out the data. 

We also cleaned the tweets for special characters in order to create a good sentiment analysis.

A heatmap was produced from our dataframe variables showing spearman correlations between known bot and non bot samples.  Many strong correlations were found. This can especially be seen between favorites_count and follower_count. Other predictors also have very strong negative correlations with the exception of listed_count. 

![Image](images/heatmap.png)
[Link](images/heatmap.png)

### 3) Inital Data Analysis

#### Tweets
We analyzed tweet and retweet quantity, frequency and timing. We looked the status count for a user on a per day basis (based on the age of the account in days). Most users were between 0 and 200, but there were some obvious outliers, with one extreme user who had 1200 per day.

We also examined tweets per minute, which we think will be an especially good predictor of bots. Approximately 4% of users had a tweet rate greater than 1 per minute. We briefly looked at the interval between tweets by checking if the time between tweets was on a whole minute. It looked like approximately 1.5% of the users had odd, whole minute timing between tweets.

#### Hashtags
We analyzed hashtag topics and topic diversity. Below is a chart showing our top hashtags by sentiment. We see that overall the top hashtags have more positive tweets than negative. We did not find any abnormalities regarding number of hashtags per tweet; there were at max 10, sloping down to 4 or less pretty quickly, and the average was 1.36.


![Image](images/Top_Hashtags_by_Sentiment.png)


#### Text Content
In our data approximately 26% of tweets contained links, and of those tweets, 74% contained pictures. Tweets that contained links were predominantly retweets.

For the data overall, 52% of tweets were neutral, 33% were positive, and 15% were negative. Below is a distribution of sentiments across our data set, colored by whether the tweet was an original tweet or a retweet.


![Image](images/Sentiment_Distribution.png)


####  Tweet Networks

It was easy to determine the tweet origin, since the original tweet information is stored with every tweet. However, this did not allow us to trace a tweet back through multiple users. We were able to look at tweet duration by sentiment, but there was no significant correlation. 

#### Location

Standardized location data was generally sparse. A user has location on their profile, but it is free-form text entry. Each tweet can be at a "place", but most tweets have no location data.

### 4) Additional Feature Engineering

For many elements, we created binary columns to aid our analysis. These include “Has Link”, "Has Hashtag", “Is Retweet”, "Is Quote Tweet", and "Has JPEG". We also add the percent of total calculations for each of these (e.g., "Retweet Percent" is the count of retweets over all tweets).

In addition, we converted all created dates to date types in order to accommodate duration logic. With these date we create an "Account Age" column, and also a "Tweet Life Days" column, which is the duration between the create of the orignal tweet (the retweet object's created date) and the current tweet.

We branched out and tried some calculations that appeared interesting in our preliminary EDA. First, the count of retweets a user has divided by the number of distinct users that user is retweeting. In our preliminary EDA, we found that in general, the more a user retweets, the more distinct users those tweets originate from. These expected points are represented by the lighter dots where the retweet-to-original-user ratio is near one. However, some accounts stand out because they only ever retweet from one original account, even as their number of retweets increase. This is shown by the y=x line of darker points. This line is clearly diverging from the rest of the data.


![Image](images/Tweets_per_Tweet-Originating_User.png)


We looked at tweets by account age, which generally increased positively together as you’d expect. When we looked at the same chart by the total number of retweets a user has of its tweets, we saw some outliers stand out. Therefore we added a feature defined as: a users' tweets' total retweets divided by account age (in days). This can be seen on the chart below, in which each dot represents a user. You can see one user whose account is less that 500 days old, but has had his/her tweets retweeted over 8 million times.


![Image](images/Total_Retweets_by_Account_Age.png)

### 5) Further Data Manipulation 

#### Hashtag tranformation

Hashtags were a problem because they are at a lower grain than even the tweet data. We needed a way to summarize hashtag activity by users. The hashtag object came in as many columns, which we unpivoted (using melt) down to a single column. First by user id, and then second by just the hastag itself. In doing this we were able to get a hashtage count, the hashtag senitment, and a hashtag percent popularity (hashtag count divided by count of all hashtags) on a per-user basis. By averaging these in our design matrix, we inherentely created a weighted average hashtag metrics that express how often a user is using hashtags and how popular the hashtags s/he is using are.

```python

    hashtags['hashtag_count'] = hashtags.count(axis=1)
    
    #melt by userid to unpivot the many hashtag columns
    hashtags_melt = pd.melt(hashtags.iloc[:, hashtags.columns != 'hashtag_count'], 
                        id_vars = 'user_id',var_name='hashtag_num', value_name='hashtag')
    hashtags_melt = hashtags_melt[hashtags_melt['hashtag'].notnull()]
    
    #clean text 
    hashtags_melt['text'] = hashtags_melt['hashtag'].apply(pd.Series)['text'].str.strip().str.lower()
    
    #group by each individual hashtag, count, and rename count column
    hashtags_count = hashtags_melt.groupby('text').agg({'text': 'count'}, as_index=False)
    hashtags_count.columns = ['mentions']
    
    #reset the index
    hashtags_count.reset_index(level=0, inplace=True, drop=False)
    
    #add hashtag sentiment
    hashtags_count['hash_sentiment'] = [TextBlob(str(row)).sentiment.polarity for row in hashtags_count['text']]
    
    #calculate hashtag's use as percent of total
    hashtags_count['hash_percent'] = hashtags_count['mentions'] / hashtags_count['mentions'].sum()
    
    #join per hashtag data back to per user data
    hashtags_melt = hashtags_melt.merge(hashtags_count, left_on='text', right_on='text', how='inner')
    
    #aggregate all hashtags for a user, average will result in a weighted average of hashtag stats
    user_hash_sum = hashtags_melt.groupby('user_id').agg({'mentions': np.mean, 
                                        'hash_sentiment': np.mean,'hash_percent': np.mean})
```

### 6) Standardization

We did standardize the data, but we found it made little difference in our models.
