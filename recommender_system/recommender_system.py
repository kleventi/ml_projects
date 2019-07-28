
# coding: utf-8

# In this project, as input we have some dummy data of a e-shop log including customer ids (regular customer ids start with 'CC' prefix, product ids (in the form of categoryId_itemId), and the total duration that each customer spent when clicking on an item. We can use the original duration in seconds (from file customer_product_clickDuration.csv) or the scaled duration (in range 0-1) (from file customer_product_clickDuration_scaled.csv). Moreover, we want to inform the customers for marketing reasons about a product via email, but there is no such information in the input data. So, we will create dummy emails for the sake of this project.
# 
# We will use Surprise library which is a python library for machine learning purposes, to run a series of prediction algorithms (SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering) and choose the one with the lowest RMSE. We will then make a list of regular customers that are recommended to be most probably interested in a specific product (e.g so that they can be informed if the e-shop wishes to promote this specific product).
# 
# Then we will use Turicreate library in order to run content-based popularity model and collaborative filtering predictive models (cosine similarity and pearson coefficient). For the specific data the cosine similarity model seems to give the best results (lower RMSE and precision-recall closer to 1), so it will be chosen to create a list of the ten most suitable products to recommend to each customer.

# ## 1. Import modules

# In[1]:

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import pandas as pd
import numpy as np
import csv

import sys
sys.path.append("..")

# for fake emails
import random
import string
import progressbar

import turicreate as tc
from sklearn.model_selection import train_test_split as t_train_test_split
import time


# In[2]:

from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import SVD, SVDpp, SlopeOne, NMF, NormalPredictor
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import BaselineOnly, CoClustering
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as s_train_test_split


# ## 2. Import and manipulate data

# In[3]:

orig_data = pd.read_csv('../customer_product_clickDuration.csv', header=None, names=['Customer', 'Product', 'Duration']) 
scaled_data = pd.read_csv('../customer_product_clickDuration_scaled.csv', header=None, names=['Customer', 'Product', 'Duration'])


# In[4]:

orig_data.head()


# In[5]:

orig_data.shape


# In[6]:

# Create fake emails
'''
Creates a random string of digits between 1 and 20 characters alphanumeric and adds it to a fake domain and fake 
extension
Most of these emails are completely bogus (eg - gmail.gov) but will meet formatting requirements
'''
def makeEmail():
    extensions = ['com','net','org','gov']
    domains = ['gmail','yahoo','comcast','verizon','charter','hotmail','outlook','frontier']

    winext = extensions[random.randint(0,len(extensions)-1)]
    windom = domains[random.randint(0,len(domains)-1)]

    acclen = random.randint(1,20)

    winacc = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(acclen))

    finale = winacc + "@" + windom + "." + winext
    return finale

# create custEmails dataframe with regular customers' ids and emails (below)
custEmails = pd.DataFrame(orig_data[orig_data['Customer'].str.startswith('CC')]['Customer'].drop_duplicates())

#save count to var howmany (only regular customers)
howmany = len(custEmails)

counter = 0      #counter for While loop
emailarray = []  #empty array for loop

print "Creating email addresses..."

prebar = progressbar.ProgressBar(maxval=int(howmany))

for i in prebar(range(howmany)):
    while counter < howmany:
        emailarray.append(str(makeEmail()))
        counter = counter+1
        prebar.update(i)
    
print "Email creation completed."

bar = progressbar.ProgressBar(maxval=int(howmany))
custEmails['Email'] = emailarray

# save regular customer Ids and Emails in custEmails.csv
custEmails.to_csv('../custEmails.csv',index = None, header=False)

custEmails.head()


# In[7]:

custEmails.shape


# ## 3. Recommendation based on highest click duration (highest number of seconds of a product visit)

# In[29]:

# find the 10 most interested customers in a specific product based on the time they spent on product visit
orig_data[orig_data['Product']=='0_10'].sort_values(by='Duration', ascending=False).head(10)


# ## 4. Predictions with Surprise library

# In[8]:

# find maximum Duration value in order to set the range of rating_scale below
max(orig_data['Duration'])


# In[9]:

reader = Reader(rating_scale=(0, 489))

# load data as Dataset for surprise library
data = Dataset.load_from_df(orig_data[['Customer', 'Product', 'Duration']], reader)


# In[10]:

benchmark = []
# Iterate over all algorithms
for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
    # Perform cross validation
    print('Executing' + str(algorithm))
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)
    
pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')    


# ### Start procedure with the best algorithm (according to the above results, choose the algo with the lowest rmse)

# #### Run cross-validation with the best algo (in the specific case SlopeOne) 

# In[14]:

# SlopeOne algorithm gave us the best rmse, therefore, we will train and predict with SlopeOne 

print('Executing SlopeOne')
algo = SlopeOne()
cross_validate(algo, data, measures=['RMSE'], cv=3, verbose=False)


# #### Train and test the chosen algorithm

# In[15]:

trainset, testset = s_train_test_split(data, test_size=0.25)
predictions = algo.fit(trainset).test(testset)
accuracy.rmse(predictions)


# #### Get detailed results for predictions/recommendations

# In[16]:

#  inspect our predictions in details

def get_Iu(uid):
    """ return the number of items clicked by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items clicked by the user
    """
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):
    """ return number of users that have clicked given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have clicked the item.
    """
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0
    
df = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df['Iu'] = df.uid.apply(get_Iu)
df['Ui'] = df.iid.apply(get_Ui)
df['err'] = abs(df.est - df.rui)
best_predictions = df.sort_values(by='err')[:10]
worst_predictions = df.sort_values(by='err')[-10:]


# In[27]:

# print the 10 best predictions
best_predictions


# In[28]:

# print the 10 worst predictions
worst_predictions


# In[ ]:

# test example: regular customers that are recommended to be most probably interested in product 0_10
listOfCustomers = pd.DataFrame(df[(df['iid'] == '0_10') & (df['uid'].str.startswith('CC'))]['uid'])


# ## 5. Predictions with Turicreate library

# #### Define train and test subsets

# In[30]:

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = t_train_test_split(data, test_size = .25)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# In[31]:

train_data, test_data = split_data(orig_data)


# In[32]:

# define the variables to use in the models
# custEmails.csv has the ids and emails of the regular customers

customersEmails = pd.read_csv('../custEmails.csv', header=None, names=['Customer', 'Email']) 
user_id = 'Customer'
item_id = 'Product'
users_to_recommend = list(customersEmails['Customer'])
n_rec = 10 # number of items to recommend
n_display = 30 # to display the first few rows in an output dataset


# In[34]:

# function for all models using Turicreate

def model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data, 
                                                    user_id=user_id, 
                                                    item_id=item_id, 
                                                    target=target, 
                                                    similarity_type='pearson')
        
recom = model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)
return model


# #### Train popularity, cosine and pearson models with trainset

# In[35]:

# Content-based popularity model

name = 'popularity'
target = 'Duration'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[36]:

# Collaborative Filtering Model - Cosine similarity

name = 'cosine'
target = 'Duration'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[37]:

# Collaborative Filtering Model - Pearson similarity

name = 'pearson'
target = 'Duration'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[38]:

# Model Evaluation

models_w_counts = [popularity, cos, pear]
names_w_counts = ['Popularity Model on Click Duration', 'Cosine Similarity on Click Duration', 'Pearson Similarity on Click Duration']

# compare all the models we have built based on RMSE and precision-recall
eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)


# In[39]:

# Final Output (cosine model seems to give best results)

final_model = tc.item_similarity_recommender.create(tc.SFrame(orig_data), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='Duration', similarity_type='cosine')
recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)

