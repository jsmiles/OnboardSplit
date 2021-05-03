# OnboardSplit
A machine learning model to split customers at the onboarding stage

# Context
__Buckesque__ is a fake neobank servicing a niche market. They face a challenge because their onboarding process contains several manual steps. As a result they have an extensive backlog. It is noticed that a number of new customers drop out of the process because of the delay. In the long term the company will fix the onboarding process. However, in the short term the data science team are tasked with helping to classify customers in the onboarding queue. Their goal is to identify which customers have the highest potential profitability using only onboarding information. 

# Summary
Utilising a modified version of my [DataGen](https://github.com/jsmiles/DataGen) script I built a fake dataset to represent the onboarding data of __Buckesque's__ existing customer base. I have used this data to train a machine learning model to grapple with the above problem. We have available ten thousand records of the companies customers with a metric __top__ dividing customers into two classes depending upon profitability. The challenge is to use their onboarding demographic information to train a classifier to recognise those customers. This classifier could then be operationalised into an API so the customer support team can prioritise the manual stage of the onboarding process accordingly.

The independent variables available amount to what you would have at the early onboarding stage in a bank. Variables include name, email, age, id type/country, nationality and information related to session and device. In the real world you could easily imagine certain hypotheses to be tested. A good example might be that Apple phones are expensive. As such, we might expect customers with Apple devices to have a higher profitability propensity. 


# Evaluation
Both the Random Forest model here and a Logistic regression model had similar results. This model has an __accuracy score__ of 0.61, nothing special. As you can see below the __confusion matrix__ follows the same lines. This is not impressive with both precision and recall being low. The reason this model seems to be performing so poorly is because of the input data used. It is not real world data and so there is little signal for the model to find. 



| 1194 | 306 |
|:-------------:|:-------------:|
| __678__ | __322__ |
