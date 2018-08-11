---
layout:page
title: Results and Conclusion
---

## 1. Summary

We analyzed our data, identified features, and engineered features in order to predict bots. We found which features were most important and fit several different types of models. Those models predicted bots correctly approximately 97% percent of time. The Ada Boost model had the highest accuracy score on the tuning set.

We were able to analyze sentiments, hashtags, and topics, but we were not able to logically group those into high-level interpretable topics. We were however able to integrate these scores, counts, and popularity percentages into our model. We were not able to discern diversity of location due to the sparsity of the data.

 
## 2. Results

For our one-thousand users, we had a sample size of 227 bot tweets and 9551 human tweets, which means 2.38% of our tweets were made by bots. For those tweets, bots did have more retweets per tweet, 8208 compared to 6419 retweets per human tweet. Bots are only slightly more effective than humans. Bots were favorited less than humans, with .25 favorites per tweet compared to 1.2 favorites per tweet for humans.

After our analysis, we did see that trending hashtags had more positive sentiments than negative. As for topics, there was no statistically significant evidence that bots are more negative or positive. For the bots in our data set, we found average sentiment for the bot topics was -.002, whereas humans averaged .099. This difference is not statistically significant; the standard deviations were approximately .26 and .3 respectively. 

The precision in our models seemed to be effective at finding approximately the same percentage of bots in our tune and test data as our training data had. A scatter matrix was used to view the data for collinearity to reduce over fitting our data. ROC scores and cross validation scores also showed that our test and training data were very close, showing little signs of overfitting the data. A precision-recall plot could help guide future recall accuracy. 


## 3. Conclusion & Future Work

Overall we found that bots are having their voices heard, but we have no evidence to show that they are more successful than humans, nor that they, on average, lean positively or negatively in sentiment.

In our analysis, we were able to predict bots at approximately the percentage we saw in our training data, but we are not confident that the bots we are predicting is correct. This is because of the rarity of our successes; the models could be inaccurate and still return high-seeming accuracy scores. 

For the future, more work needs to be done to better account for the rare event bias. We would also need a better method than overall score to test that our predictions are accurate. And finally, we need to find a way to map our topics to an overall topic so we could group our bots and tweet sentiment by these topics and better analyze influence.

It would also be interesting focus on sentiment-outlier bots specifically (rather than grouping all bots on average). You could bin bots by overall sentiment so you could focus on those highly negative or highly positive bots only. Then you could better try to judge impact of those bots. 

Ideally, you would have a lot more data with botometer scores than we had. This was due to botometer limitations only. We had collected over 1.6 million tweets, but it took days to score them. This type of analysis would be interesting as an unsupervised learning project, where we would not need to rely on the output others' models in order to train our own.


