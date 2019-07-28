In this project, as input we have some dummy data of a e-shop log including customer ids (regular customer ids start with 'CC' 
prefix, product ids (in the form of categoryId_itemId), and the total duration that each customer spent when clicking on an item. 
We can use the original duration in seconds (from file customer_product_clickDuration.csv) or the scaled duration (in range 0-1) 
(from file customer_product_clickDuration_scaled.csv). Moreover, we want to inform the customers for marketing reasons about a 
product via email, but there is no such information in the input data. So, we will create dummy emails for the sake of this project.

We will use Surprise library, which is a python library for machine learning purposes, to run a series of prediction algorithms 
(SVD, SVDpp, SlopeOne, NMF, NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering) and 
choose the one with the lowest RMSE. We will then make a list of regular customers that are recommended to be most probably 
interested in a specific product (e.g so that they can be informed if the e-shop wishes to promote this specific product).

Then we will use Turicreate library in order to run content-based popularity model and collaborative filtering predictive models 
(cosine similarity and pearson coefficient). For the specific data the cosine similarity model seems to give the best results 
(lower RMSE and precision-recall closer to 1), so it will be chosen to create a list of the ten most suitable products to recommend
to each customer.
