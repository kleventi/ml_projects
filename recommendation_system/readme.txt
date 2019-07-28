In this project, as input data we have some dummy data (in sqlite format) that refer to a three month log of the clicks in a e-shop.
The data include customer ids, item and category ids, and the duration (in seconds) that each customer spent when clicking on an 
item, each time. Regular customers have ids starting with 'CC' prefix.

Using this data, we will create a list where we will recommend ten products to each customer that would most probable interest them.

For the recommendation list we will apply different models to the original duration data, the dummy data ('1' if a customer clicked
on an item) and the scaled duration data. The models we will use are: content-based popularity model and collaborative filtering 
(cosine similarity and pearson coefficient) and we will choose the one which gives the best results.

In addition, we will recommend the ten most suitable items to a specific given customer.

Last, we will produce a list of the customers that would be interested in items of a specific category (e.g. so that they can be 
informed if the e-shop wishes to promote a new or existing item of this category).
