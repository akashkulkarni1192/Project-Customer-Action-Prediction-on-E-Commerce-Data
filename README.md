Project-Customer-Action-Prediction-on-E-Commerce-Data
=====================================================
Implemented using ANN model

Goal
----
Predicting the action of the user on a website given the user's data.

Dataset contains following features per user
--------------------------------------------

1. Is user accessing the website on mobile platform or not
2. Number of products the user has viewed 
3. Duration for which the user has stayed on the site
4. Is the user a returning visitor or new one
5. Time of the day at which the user has visited the site

User's action can be one of the following four categories
---------------------------------------------------------
1. Bounce: Customer exits the site without placing any order.
2. Add to cart: Cutomer adds products to cart but doesnt buy them.
3. Begin check out: Customer begins to check out but doesnt finish the transaction.
4. Finish check out: Customer makes the payment and buys the product.


Implementation
--------------
As there were only two numerical columns i.e number of products viewed and duration for which user stayed on site and all other columns were either boolean or categorical, I had to normalize these two columns before training the model. I, then, used a neural network with 3 hidden layers with sigmoid as activation function at hidden layer and softmax as activation function at output layer.
