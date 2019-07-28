
# coding: utf-8

# In this project, as input data we have some dummy data (in sqlite format) that refer to a three month log of the clicks in a e-shop. The data include customer ids, item and category ids, and the duration (in seconds) that each customer spent when clicking on an item, each time. Regular customers have ids starting with "CC" prefix. 
# 
# Using this data, we will create a list where we will recommend ten products to each customer that would most probable interest them. 
# 
# For the recommendation list we will apply different models to the original duration data, the dummy data ('1' if a customer clicked on an item) and the scaled duration data. The models we will use are: content-based popularity model and collaborative filtering (cosine similarity and pearson coefficient) and we will choose the one which gives the best results.
# 
# In addition, we will recommend the ten most suitable items to a specific given customer.
# 
# Last, we will produce a list of the customers that would be interested in items of a specific category (e.g. so that they can be informed if the e-shop wishes to promote a new or existing item of this category).

# ## 1. Import modules

# In[1]:

import pandas as pd
import sqlite3


# In[2]:

get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')

import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")


# ## 2. Load data

# In[1]:

# Load the data from all .sqlite files in one dataframe for further analysis and manipulation


# In[3]:

months = {3:'March', 4:'April', 5:'May'}

month_df = pd.DataFrame()

for k,v in months.items():
    for i in range(1,31+k%2,1):
        cnx = sqlite3.connect(r'../Data'+'//'+ v+'//2019_' + v + '_' + str(i)+'.sqlite')
        #print(r'../Data'+'//'+ v+'//2019_' + v + '_' + str(i)+'.sqlite')
        #print("Connection Successful",cnx)
        df1 = pd.read_sql('select * from Data',cnx)
        cnx.close()
        # add a column to indicate the day
        df1['Day'] = str(i)
        # add a column to indicate the month
        df1['Month'] = str(k)
        month_df = month_df.append(df1)

print(month_df.shape)


# In[4]:

df = month_df

# split the information from the Sess_Info column
df[['Customer', 'Item', 'Category', 'SecNum', 'Sec']] = month_df['Sess_Info'].str.split('_',expand=True)

# columns Sec and Sess_Info are of no use
df = df.drop('Sec',axis=1)
df = df.drop('Sess_Info',axis=1)

# remove the prefix 'item' from column Item and the prefix 'cat' from column Category
df = df.apply(lambda S:S.str.strip('item'))
df = df.apply(lambda S:S.str.replace('cat',''))

# add a column with the Number of seconds as floats for later use
df['SecNumF'] = df['SecNum'].astype(float)


# In[5]:

print(df.shape)
#df.info()
#df.describe()
df.columns
#df.head


# ## 3. Data preparation

# #### 3.1. Create data with user, item, and target field (duration of clicks in seconds

# In[6]:

# I will use only the customers, the items, the categories and the number of seconds of each click.

df2 = df[['Customer', 'Item', 'Category', 'SecNumF']]


# In[7]:

# Since each specific product is determined by a combination of an ItemId and a CategoryId, 
# I will create a unique productId as a concatenation of the category and the item id.

# Product = CategoryId_ItemId

df3 = df2.assign(Product = df2.Category.astype(str) + '_' + df2.Item.astype(str)).drop(['Item', 'Category'], axis=1)
df3.head()


# In[8]:

# Calculate the total sum of the time each customer spent in each product.

data = df3.groupby(['Customer','Product'])['SecNumF'].sum().reset_index()

data.head()
#data.columns


# #### 3.2. Create dummy

# In[9]:

# If a customer has clicked on a product, then SecNumF_Dummy is marked as 1

def create_data_dummy(data):
    data_dummy = data.copy()
    data_dummy['SecNumF_Dummy'] = 1
    return data_dummy

data_dummy = create_data_dummy(data)


# #### 3.3. Normalize item values across users

# In[10]:

# create a user-item matrix

df_matrix = pd.pivot_table(data, values='SecNumF', index='Customer', columns='Product')


# In[11]:

# normalize time of clicks of each product across customers
# from 0–1 (with 1 being the highest number of seconds for an item and 0 being 0 time for that item)

df_matrix_norm = (df_matrix-df_matrix.min())/(df_matrix.max()-df_matrix.min())


# In[12]:

# create a table for input to the modeling  
d = df_matrix_norm.reset_index() 
d.index.names = ['Scaled_SecNumF'] 
data_norm = pd.melt(d, id_vars=['Customer'], value_name='Scaled_SecNumF').dropna()
print(data_norm.shape)
data_norm.head()


# ## 4. Split train and test set

# In[15]:

# We use 80:20 ratio for our train-test set size.
# Our training portion will be used to develop a predictive model, while the other to evaluate the model’s performance.

def split_data(data):
    '''
    Splits dataset into training and test set.
    
    Args:
        data (pandas.DataFrame)
        
    Returns
        train_data (tc.SFrame)
        test_data (tc.SFrame)
    '''
    train, test = train_test_split(data, test_size = .2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# In[16]:

train_data, test_data = split_data(data)
train_data_dummy, test_data_dummy = split_data(data_dummy)
train_data_norm, test_data_norm = split_data(data_norm)


# ## 5. Define Models using Turicreate library

# In[18]:

# define our variables to use in the models
# constant variables to define field names include:

user_id = 'Customer'
item_id = 'Product'
users_to_recommend = list(data[user_id])
n_rec = 10       # number of items to recommend
n_display = 30   # to display the first few rows in an output dataset


# In[19]:

# define a function for all models

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


# ## 6. Popularity Model

# In[20]:

# Using number of seconds SecNumF

name = 'popularity'
target = 'SecNumF'
popularity = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[21]:

# Using SecNumF dummy

name = 'popularity'
target = 'SecNumF_Dummy'
pop_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[22]:

# Using scaled number of seconds

name = 'popularity'
target = 'Scaled_SecNumF'
pop_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# ## 7. Collaborative Filtering Model

# #### 7.1. Cosine similarity

# In[23]:

# Using number of seconds

name = 'cosine'
target = 'SecNumF'
cos = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[24]:

# Using SecNumF dummy

name = 'cosine'
target = 'SecNumF_Dummy'
cos_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[25]:

# Using scaled number of seconds

name = 'cosine' 
target = 'Scaled_SecNumF' 
cos_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# #### 7.2. Pearson similarity

# In[26]:

# Using number of seconds

name = 'pearson'
target = 'SecNumF'
pear = model(train_data, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[27]:

# Using SecNumF dummy

name = 'pearson'
target = 'SecNumF_Dummy'
pear_dummy = model(train_data_dummy, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# In[28]:

# Using scaled number of seconds

name = 'pearson'
target = 'Scaled_SecNumF'
pear_norm = model(train_data_norm, name, user_id, item_id, target, users_to_recommend, n_rec, n_display)


# ## 8. Model Evaluation

# In[29]:

# create initial callable variables for model evaluation:

models_w_counts = [popularity, cos, pear]
models_w_dummy = [pop_dummy, cos_dummy, pear_dummy]
models_w_norm = [pop_norm, cos_norm, pear_norm]

names_w_counts = ['Popularity Model on SecNumF', 'Cosine Similarity on SecNumF', 'Pearson Similarity on SecNumF']
names_w_dummy = ['Popularity Model on SecNumF Dummy', 'Cosine Similarity on SecNumF Dummy', 'Pearson Similarity on SecNumF Dummy']
names_w_norm = ['Popularity Model on Scaled SecNumF', 'Cosine Similarity on Scaled SecNumF', 'Pearson Similarity on Scaled SecNumF']


# In[30]:

# Compare all the models we have built based on RMSE and precision-recall characteristics:

eval_counts = tc.recommender.util.compare_models(test_data, models_w_counts, model_names=names_w_counts)
eval_dummy = tc.recommender.util.compare_models(test_data_dummy, models_w_dummy, model_names=names_w_dummy)
eval_norm = tc.recommender.util.compare_models(test_data_norm, models_w_norm, model_names=names_w_norm)


# ### From the above summary, we select the Cosine similarity on scaled number of seconds approach as our final model, because this combination gives the best results (the desirable outcome has low RMSE and precision-recall close to 1).

# ## 9. Final Output

# In[32]:

# data_norm


# In[31]:

# rerun the model using the whole dataset, as we came to a final model using train data and evaluated with test set.

final_model = tc.item_similarity_recommender.create(tc.SFrame(data_norm), 
                                            user_id=user_id, 
                                            item_id=item_id,
                                            target ='Scaled_SecNumF',
                                            similarity_type = 'cosine' )

recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)


# #### 9.1. CSV output file

# In[32]:

df_rec = recom.to_dataframe()
print(df_rec.shape)
df_rec.head()


# In[33]:

# create the desired output

def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe().drop_duplicates()
    df_rec['recommendedProducts'] = df_rec.groupby([user_id])[item_id]         .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['Customer', 'recommendedProducts']].drop_duplicates()         .sort_values('Customer').set_index('Customer')
    if print_csv:
        df_output.to_csv('../recommendation_list.csv')
        print("An output file was created with the name 'recommendation_list.csv'")
    return df_output


# In[34]:

# setprint_csv to true to print out the output in a csv file

df_output = create_output(pear_norm, users_to_recommend, n_rec , print_csv=True)
print(df_output.shape)
df_output.head()


# #### 9.2. Customer recommendation function

# In[35]:

# return a recommendation list of ten products for a given customer ID

def customer_recomendation(customer_id):
    if customer_id not in df_output.index:
        print('Customer not found.')
        return customer_id
    return df_output.loc[customer_id]


# In[36]:

# test example
customer_recomendation('CC20190310')


# ## 10. Given a category Id, get a list of customers that are probably interested

# In[12]:

# returns a list of target Customers that are probably interested in items of category "catId"

def findTargetCustomers(catId):
    
    # load the list of recommendations to customers
    customers = pd.read_csv('../recommendation_list.csv') 
    
    # Create a separate row for each one of the multiple values in column "recommendedProducts"
    def splitDataFrameList(df,target_column,separator):
        def splitListToRows(row,row_accumulator,target_column,separator):
            split_row = row[target_column].split(separator)
            for s in split_row:
                new_row = row.to_dict()
                new_row[target_column] = s
                row_accumulator.append(new_row)
        new_rows = []
        df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
        new_df = pd.DataFrame(new_rows)
        return new_df

    df = splitDataFrameList(customers,'recommendedProducts','|')
    
    # split the information from the recommendedProducts column in two new columns: Category and Item
    df[['Category', 'Item']] = df['recommendedProducts'].str.split('_',expand=True)
    df = df.drop('recommendedProducts',axis=1)
    
    # keep only the rows that refer to the given Category and to regular customers (begining with CC)
    df = df[(df['Category'] == catId) & (df['Customer'].str.startswith('CC'))]
    
    # Keep only the customer ids
    df = df['Customer']

    tarCustLst = df.drop_duplicates()
    tarCustLst.to_csv('../tarCustLst.csv',index = None, header=False)
    
    return tarCustLst


# In[13]:

# test example
tarCustLst = findTargetCustomers('1')

