---
layout:page
title: Results and Conclusion
---

## 1. Summary

We analyzed our data, identified features, and engineered features in order to predict bots. We found which features were most important and fit several different types of models. Those models predicted approximately 97% percent of time.

We were able to analyze sentiments, hashtags, and topics, but we were not able to logically group those into high-level interpretable topics. We were able to integrate these scores, counts, and popularity percenetages into our model. We were not able to discern diversity of location due to the sparsity of the data.

 
## 2. Results

For our one-thousand users, we had a sample size of 227 bot tweets and 9551 human tweets, which means 2.38% of our tweets were made by bots. For those tweets, bots did have more retweets per tweet, 8208 compared to 6419 retweets per human tweet. Bots are only slightly more effective than humans. Bots were favorited less than humans, with .25 favorites per tweet compared to 1.2 favorites per tweet for humans.

After our analysis, we did saw that trending hashtags had more positive sentiments than negative. As for topic, there was no statistically significant evidence that bots are more negative or positive. For the bots in our data set, we found average sentiment for the bot topics was -.002, whereas humans averaged .099. This difference is probably not actually statistically significant; the standard deviations werw approx .26, .3 respectively. 

The precision in our models proved to be very effective in finding twitter bots. A scatter matrix was used to view the data for collinearity to reduce over fitting our data. ROC scores and cross validation scores also showed that our test data and out training data were very close and showed little signs of overfitting the data. 


## 3. Conclusion & Future Work

Overall we found that bots are getting their messages out there, but we have no evidence to show that they are more successful than humans, nor that they, on average, lean positively or negatively in sentiment.

In our analysis, we were able to predict bots at approximately the percentage we saw in our training data, but we are not confident that the bots we are prediciting is correct. This is because of the rarity of our successes; the models could be inaccurate and still return higth-seeming accuracy scores. 

For the future, more work needs to be done to better account for the rare event bias. We would also need a better method than the score to test that our predictions are accurate. And finally we need to find a way to map our topics to an overall topic so we could group our bots and tweet sentiment by these topics and better analyse influence.
