#!/usr/bin/env python
# coding: utf-8

# #### In this project we will get data from a table in a web page, manipulate it, save it to a dataframe and finally to a csv file for future use.
# 
# ##### The data is about "What Women Earn by Race/Ethnicity in the USA" and are taken from the below link: 
# https://statusofwomendata.org/explore-the-data/employment-and-earnings/additional-state-data/what-women-earn-by-race-ethnicity/

# In[1]:


# Import modules

import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup


# In[2]:


# Download the web page containing the data.
# Must define a User-Agent, otherwise the web page rejects the request (error 403)

url = "https://statusofwomendata.org/explore-the-data/employment-and-earnings/additional-state-data/what-women-earn-by-race-ethnicity/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:68.0) Gecko/20100101 Firefox/68.0'}
page = requests.get(url, headers=headers)
#print(page.content.decode())


# In[3]:


# Create a BeautifulSoup class to parse the page.

soup = BeautifulSoup(page.content, 'html.parser')


# In[4]:


# After using the code inspector from our browser, we located the desired data inside the area of the article 
# with id post-748. We will find it and assign to table_contents (article id="post-748" class="post")

table_contents = soup.find(id = "post-748")
# table_contents


# In[5]:


# Inside table_contents, we find each individual table row. table_rows is a resultset with all table rows

table_rows = table_contents.find_all('tr')


# In[6]:


# We then save the beautifulsoup resulset to a pandas dataframe pd_table

l = []
for tr in table_rows:
    td = tr.find_all('td')
    row = [tr.text for tr in td]
    l.append(row)

pd_table = pd.DataFrame(l, columns=["State", "White", "W_Ratio", "Hispanic", "H_Ratio", "Black", "B_Ratio", "Asian/Pacific islander", "A_Ratio", "Native American", "N_Ratio", "Other Race or Two or More Races", "O_Ratio", "All women", "Ratio to all men"])    


# In[7]:


# Drop the unnecessary rows and columns

pd_table = pd_table.drop(["W_Ratio", "H_Ratio", "B_Ratio", "A_Ratio", "N_Ratio", "O_Ratio", "Ratio to all men"],axis=1)
pd_table = pd_table.drop([0, 1, 2, 55, 56])
pd_table = pd_table.reset_index().drop('index',axis=1)


# In[8]:


# Clear unwanted characters
pd_table = pd_table.applymap(lambda x: str(x).strip('\n'))
pd_table = pd_table.applymap(lambda x: str(x).lstrip('$'))

# Replace "," with "." in wages (in order to convert from strings to floats later)
pd_table = pd_table.apply(lambda x: x.str.replace(',','.'))


# In[9]:


# Check for empty cells
pd_table.isnull().values.any()


# In[10]:


# Convert wage columns from strings to floats for future manipulation

cols = pd_table.columns.drop('State')
pd_table[cols] = pd_table[cols].apply(pd.to_numeric, errors='coerce')


# In[11]:


# Fill the NaN values with the mean value of each column 
# (we could fill them with 0 or delete the rows containing NaN values or replace them with different value (such as the median))

pd_table.fillna(pd_table.mean(), inplace=True)


# In[12]:


# Add a column with the mean value of each row

pd_table['Mean'] = pd_table.mean(axis=1)


# In[13]:


# For the shake of this project we will assume that in reality, all states that have a mean wage below the total mean wage 
# belong to cluster 1 whereas the rest states belong to cluster 2 
# We made this assumption in order to help us evaluate our clustering models at the end, by comparing predictions with true cluster values

pd_table['Cluster'] = np.where(pd_table['Mean'] <= pd_table['Mean'].mean() , 1, 2)


# In[14]:


pd_table.head()


# In[17]:


# Save dataframe to file for future use
pd_table.to_csv(r'C:\YourPath\data.csv')


# In[ ]:




