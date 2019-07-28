
# coding: utf-8

# ## Import modules

# In[1]:

import pandas as pd
import sqlite3


# ### Load data

# In[2]:

# Load the data from all .sqlite files in one dataframe for further analysis and manipulation 

months = {3:'March', 4:'April', 5:'May'}

month_df = pd.DataFrame()

for k,v in months.items():
    for i in range(1,31+k%2,1):
        cnx = sqlite3.connect(r'../Data'+'//'+ v+'//2019_' + v + '_' + str(i)+'.sqlite')
        df1 = pd.read_sql('select * from Data',cnx)
        cnx.close()
        # add a column to indicate the day
        df1['Day'] = str(i)
        # add a column to indicate the month
        df1['Month'] = str(k)
        month_df = month_df.append(df1)

print(month_df.shape)


# In[3]:

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


# In[4]:

print(df.shape)
#df.info()
#df.describe()
df.columns


# ### Data Analysis

# #### What was the most popular item category in our online shop in: March

# In[5]:

df[(df['Month']=='1')]['Category'].value_counts().sort_values(ascending=False).head(1)


# #### What was the most popular item category in our online shop in: April

# In[6]:

df[(df['Month']=='2')]['Category'].value_counts().sort_values(ascending=False).head(1)


# #### What was the most popular item category in our online shop in: May

# In[7]:

df[(df['Month']=='3')]['Category'].value_counts().sort_values(ascending=False).head(1)


# #### What was the most popular item category in our online shop in: Overall

# In[8]:

df['Category'].value_counts().sort_values(ascending=False).head(1)


# #### What are the top 5 items customers were interested in ?

# In[9]:

# I will group by item and category, since different categories can have items with the same item number

res = df.groupby(['Item', 'Category']).size().sort_values(ascending=False).head(5)
res


# #### What was an average time they spent on one item/click?

# In[10]:

print(df['SecNumF'].mean())


# #### Assuming that interested customer spent more than 10 sec on the item, what was the most and least popular item in March?

# In[11]:

pop1 = df[(df['SecNumF'] > 10.0) & (df['Month'] == '1')].groupby(['Item', 'Category']).size().sort_values(ascending=False)
print(pop1.head(1))  # most popular
print(pop1.tail(1))  # least popular


# #### Assuming that interested customer spent more than 10 sec on the item, what was the most and least popular item in April?

# In[12]:

pop2 = df[(df['SecNumF'] > 10.0) & (df['Month'] == '2')].groupby(['Item', 'Category']).size().sort_values(ascending=False)
print(pop2.head(1))
print(pop2.tail(1))


# #### Assuming that interested customer spent more than 10 sec on the item, what was the most and least popular item in May?

# In[13]:

pop3 = df[(df['SecNumF'] > 10.0) & (df['Month'] == '3')].groupby(['Item', 'Category']).size().sort_values(ascending=False)
print(pop3.head(1))
print(pop3.tail(1))


# #### Assuming that interested customer spent more than 10 sec on the item, what was the most and least popular item overall?

# In[14]:

pop4 = df[df['SecNumF'] > 10.0].groupby(['Item', 'Category']).size().sort_values(ascending=False)
print(pop4.head(1))
print(pop4.tail(1))


# #### Please find the top 3 categories and the 5 clients (if applicable) who would be the best bet to offer new products from those categories.

# In[15]:

# I will filter only the clicks with > 10 secs. Then I will calculate the top 3 categories 
# (with the highest sum of secs of all items per category)

t = df[df['SecNumF'] > 10.0].groupby(['Category']).sum().sort_values(by='SecNumF', ascending=False).head(3)
t


# In[21]:

# I will calculate the top 10 Customers with the highest sum of seconds in items of category 3 (only for clicks with > 10 secs)

cust1 = df[(df['Category']=='3')& (df['SecNumF'] > 10.0)].groupby(['Customer']).sum().sort_values(by='SecNumF', ascending=False).head(10)


# In[22]:

# I will calculate the top 10 Customers with the highest sum of seconds in items of category 2 (only for clicks with > 10 secs)

cust2 = df[(df['Category']=='2')& (df['SecNumF'] > 10.0)].groupby(['Customer']).sum().sort_values(by='SecNumF', ascending=False).head(10)


# In[23]:

# I will calculate the top 10 Customers with the highest sum of seconds in items of category 4 (only for clicks with > 10 secs)

cust3 = df[(df['Category']=='4')& (df['SecNumF'] > 10.0)].groupby(['Customer']).sum().sort_values(by='SecNumF', ascending=False).head(10)


# In[ ]:

# I will find the customers that are common in the three categories and have the highest sums of seconds


# In[24]:

c1 = pd.merge(cust1, cust2, how='inner', on=['Customer']).sort_values(by=['SecNumF_x', 'SecNumF_y'],ascending=False)


# In[25]:

c2 = pd.merge(c1, cust3, how='inner', on=['Customer']).sort_values(by=['SecNumF_x', 'SecNumF_y', 'SecNumF'],ascending=False).head(5)
c2


# Above are the 5 customers for categories 2, 3 and 4.
