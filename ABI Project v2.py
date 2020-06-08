#!/usr/bin/env python
# coding: utf-8

# ![Capture.PNG](attachment:Capture.PNG)

# # Recommender System Using Python

# ## Analytics for BI Project
# ### By Deepal Rathod
# ### RUID: 196000794

# ## Contents
# 
# 1. What is Recommender System?
# 2. Types of recommendation engines
# 3. Dataset
# 4. Exploratory Data Analysis
# 5. Item based recommender system
# 6. Recommendation output

# # 1. What is Recommender System?

# A recommender system is a simple algorithm whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. 
# 
# An example of recommendation in action is when you visit Amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you. They are also used by Music streaming applications such as Spotify and Deezer to recommend music that you might like.

# # 2. Types of recommendation engines

# The most common types of recommendation systems are:
# 
#   1) Collaborative filtering recommender systems
# 
#   2) Content based filtering recommender systems

# ## 1) Collaborative filtering recommender systems:
# In collaborative filtering the behavior of a group of users is used to make recommendations to other users. Recommendation is based on the preference of other users. A simple example would be recommending a movie to a user based on the fact that their friend liked the movie.
# 
# There are two types of collaborative models: 
# 
#       1) Memory-based methods
#       2) Model-based methods
# 
# ### 1.1) Memory-based methods:
# 
# Advantages of memory-based techniques:
#    1. They are simple to implement 
#    2. The resulting recommendations are often easy to explain
# 
# They are divided into two:
# 
#    a) User-based collaborative filtering: 
#     In this model products are recommended to a user based on the fact that the products have been liked by users similar to the other user. For example if User1 and User2 like the same movies and a new movie comes out that User1 likes then we can recommend that movie to User2 because User1 and User2 seem to like the same movies.
# 
#    b) Item-based collaborative filtering: 
#     These systems identify similar items based on users’ previous ratings. For example if users A,B and C gave a 5 star rating to books X and Y then when a user D buys book Y they also get a recommendation to purchase book X because the system identifies book X and Y as similar based on the ratings of users A,B and C.
# 
# ### 1.2) Model-based methods:
# 
# Model-based methods are based on matrix factorization and are better at dealing with sparsity. They are developed using data mining, machine learning algorithms to predict users’ rating of unrated items. In this approach techniques such as dimensionality reduction are used to improve the accuracy. Examples of such model-based methods include decision trees, rule-based models, Bayesian methods and latent factor models.

# ## 2) Content based systems:
# Content based systems use meta data such as genre, producer, actor, musician to recommend items say movies or music. Such a recommendation would be for instance recommending movie that featured particular actor because someone watched and liked his other movies. Similarly you can get music recommendations from certain artists because you liked their music. Content based systems are based on the idea that if you liked a certain item you are most likely to like something that is similar to it.

# # 3. Approach

# ## Here is a very simple illustration of how recommender systems work in the context of an e-commerce site.
# 
# Two users buy the same items A and B from an ecommerce store.
# 
# 
# When this happens the similarity index of these two users is computed. 
# 
# 
# Depending on the score the system can recommend item C to the other user because it detects that those two users are similar in terms of the items they purchase.

# ![Recommender%20system.png](attachment:Recommender%20system.png)

# # 4. Dataset

# We have used the MovieLens Dataset.
# 
# We are using 3 files:
# 1. Users dataset
# 2. Movie ratings
# 3. Movies List
# 
# These files contain 1,000,209 anonymous ratings of approximately 3,900 movies made by 6,040 MovieLens users who joined MovieLens in 2000.

# 1. USERS FILE DESCRIPTION
# 
# ================================================================================
# 
# User information is in the file "users.csv" and is in the following
# format:
# 
# UserID::Gender::Age::Occupation::Zip-code
# 
# - Gender is denoted by a "M" for male and "F" for female
# 
# 
# - Age is chosen from the following ranges:
# 
# 	*  1:  "Under 18"
# 	* 18:  "18-24"
# 	* 25:  "25-34"
# 	* 35:  "35-44"
# 	* 45:  "45-49"
# 	* 50:  "50-55"
# 	* 56:  "56+"
# 
# 
# - Occupation is chosen from the following choices:
# 
# 	*  0:  "other" or not specified
# 	*  1:  "academic/educator"
# 	*  2:  "artist"
# 	*  3:  "clerical/admin"
# 	*  4:  "college/grad student"
# 	*  5:  "customer service"
# 	*  6:  "doctor/health care"
# 	*  7:  "executive/managerial"
# 	*  8:  "farmer"
# 	*  9:  "homemaker"
# 	* 10:  "K-12 student"
# 	* 11:  "lawyer"
# 	* 12:  "programmer"
# 	* 13:  "retired"
# 	* 14:  "sales/marketing"
# 	* 15:  "scientist"
# 	* 16:  "self-employed"
# 	* 17:  "technician/engineer"
# 	* 18:  "tradesman/craftsman"
# 	* 19:  "unemployed"
# 	* 20:  "writer"

# 2. RATINGS FILE DESCRIPTION
# 
# ================================================================================
# 
# All ratings are contained in the file "ratings.csv" and are in the
# following format:
# 
# UserID::MovieID::Rating::Timestamp
# 
# - UserIDs range between 1 and 6040 
# - MovieIDs range between 1 and 3952
# - Ratings are made on a 5-star scale (whole-star ratings only)
# - Timestamp is represented in seconds since the epoch as returned by time(2)
# - Each user has at least 20 ratings

# 3. MOVIES FILE DESCRIPTION
# 
# ================================================================================
# 
# Movie information is in the file "movies.csv" and is in the following
# format:
# 
# MovieID::Title::Genres
# 
# - Titles are identical to titles provided by the IMDB (including year of release)
# 
# 
# - Genres are pipe-separated and are selected from the following genres:
# 
# 	* Action
# 	* Adventure
# 	* Animation
# 	* Children's
# 	* Comedy
# 	* Crime
# 	* Documentary
# 	* Drama
# 	* Fantasy
# 	* Film-Noir
# 	* Horror
# 	* Musical
# 	* Mystery
# 	* Romance
# 	* Sci-Fi
# 	* Thriller
# 	* War
# 	* Western

# ## 4.1 Importing required Python libraries

# In[85]:


#import pandas and numpy

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ## 4.2 Loading Dataset

# In[86]:


# read Users data

Users = pd.read_csv('C:\\Users\\Deepal\\Desktop\\ABI Project\\Users.csv')
Users.head(10)


# In[87]:


# read Movie Ratings data

MovieRatings = pd.read_csv('C:\\Users\\Deepal\\Desktop\\ABI Project\\Movie ratings.csv')
MovieRatings.head(10)


# In[88]:


# read Movie List data

MovieList = pd.read_csv('C:\\Users\\Deepal\\Desktop\\ABI Project\\Movies List.csv', encoding='latin-1')
MovieList.head(10)


# ## 4.3 Merge two dataset

# Since the MovieID columns are the same we can merge these datasets on this column.

# In[89]:


# merge two dataset

Movie_Ratings = pd.merge(MovieRatings, MovieList, on='MovieID')
Movie_Ratings.head(10)


# In[90]:


Movie_Ratings.describe()


# We can see that the average rating is 3.52 and the max is 5. We also see that the dataset has 100003 records.

# ## 4.4 Data preparation

# ### 4.4.1 Creating a dataframe with the average rating for each movie and the number of ratings. We are going to use these ratings to calculate the correlation between the movies.

# In[91]:


ratings = pd.DataFrame(Movie_Ratings.groupby('MovieName')['Ratings'].mean())
ratings.head(10)


# ### 4.4.2 Checking number of ratings for each movie. 
# 
# This is important so that we can see the relationship between the average rating of a movie and the number of ratings the movie got. It is very possible that a 5 star movie was rated by just one person. It is therefore statistically incorrect to classify that movie has a 5 star movie. We will therefore need to set a threshold for the minimum number of ratings as we build the recommender system.

# In[92]:


ratings['Number of ratings'] = Movie_Ratings.groupby('MovieName')['Ratings'].count()
ratings.head(10)


# # 5. Exploratory Data Analysis

# ## 5.1 Techniques we have used : Correlation
# 
# •	Correlation is a statistical measure that indicates the extent to which two or more variables fluctuate together.
# 
# 
# •	Movies that have a high correlation coefficient are the movies that are most similar to each other. 
# 
# 
# •	In our case we will be using the Pearson correlation coefficient. This number will lie between -1 and 1. 
# 
# 
#      • 1 indicates a positive linear correlation
#      • -1 indicates a negative correlation
#      • 0 indicates no linear correlation. Therefore, movies with a zero correlation are not similar at all.

# ## 5.2 Plotting a Histogram using pandas plotting functionality to visualize the distribution of the ratings with respect to movies

# In[93]:


sns.distplot(ratings['Ratings'], hist=True, kde=False, bins=100, hist_kws={'edgecolor':'#000000'})


# We can see that most of the movies are rated between 2.5 and 4.

# ## 5.3 Plotting a Histogram to visualize the distribution of the Number of ratings with respect to movies

# In[94]:


sns.distplot(ratings['Number of ratings'], hist=True, kde=False, bins=60, hist_kws={'edgecolor':'#000000'})


# From the above histogram it is clear that most movies have few ratings. Movies with most ratings are those that are most famous.

# ## 5.4 Plotting relationship between the rating of a movie and the number of ratings.

# In[95]:


import seaborn as sns
sns.jointplot(x='Ratings', y='Number of ratings', data=ratings)


# From the diagram we can see that their is a positive relationship between the average rating of a movie and the number of ratings. The graph indicates that the more the ratings a movie gets the higher the average rating it gets. This is important to note especially when choosing the threshold for the number of ratings per movie.

# # 6. Item based recommender system

# Assume that a user has watched Air Force One (1997) and Contact (1997). 
# 
# 
# Our recommender system will recommend movies to this user based on this watching history. 
# 
# 
# The goal is to look for movies that are similar to Contact (1997) and Air Force One (1997) which our system will recommend to this user. 
# 
# 
# We can achieve this by computing the correlation between these two movies’ ratings and the ratings of the rest of the movies in the dataset.

# ## 6.1 Converting dataset into a matrix
# 
# Converting dataset into a matrix with the movie titles as the columns, the UserID as the index and the ratings as the values.
# 
# 
# We will get output as dataframe with the columns as the movie titles and the rows as the user ids. 
# 
# 
# Each column represents all the ratings of a movie by all users. 
# 
# 
# The rating appear as NaN where a user didn't rate a certain movie. 
# 
# 
# Use of this matrix is to compute the correlation between the ratings of a single movie and the rest of the movies in the matrix.

# In[96]:


movie_matrix = Movie_Ratings.pivot_table(index='UserID', columns='MovieName', values='Ratings')
movie_matrix.head(5)


# ## 6.2 Most rated movies
# 
# Checking the most rated movies and choose two of them to work with our recommender system.

# In[97]:


ratings.sort_values('Number of ratings', ascending=False).head(10)


# ## 6.3 Implementing Correlation

# In[98]:


AFO_user_rating = movie_matrix['Air Force One (1997)']
contact_user_rating = movie_matrix['Contact (1997)']


# We now have the dataframes showing the user_id and the rating they gave the two movies.

# In[99]:


AFO_user_rating.head()
contact_user_rating.head()


# In order to compute the correlation between two dataframes we use pandas corrwith functionality. Corrwith computes the pairwise correlation of rows or columns of two dataframe objects.
# 
# ### 6.3.1 Correlation between each movie's rating and the ratings of the Air Force One movie.

# In[100]:


similar_to_air_force_one=movie_matrix.corrwith(AFO_user_rating)
similar_to_air_force_one.head(10)


# ### 6.3.2 Correlation between Contact (1997) ratings and the rest of the movies ratings.

# In[101]:


similar_to_contact = movie_matrix.corrwith(contact_user_rating)
similar_to_contact.head(10)


# ## 6.4 Challenges

# ### 6.4.1 Dropping Null values
# 
# As our matrix has many missing values since not all the movies were rated by all the users. 
# 
# 
# We therefore drop those null values and transform correlation results into dataframes to make the results look more appealing.

# In[102]:


corr_contact = pd.DataFrame(similar_to_contact, columns=['Correlation'])
corr_contact.dropna(inplace=True)
corr_contact.head()


# In[103]:


corr_AFO = pd.DataFrame(similar_to_air_force_one, columns=['correlation'])
corr_AFO.dropna(inplace=True)
corr_AFO.head()


# These two dataframes above show us the movies that are most similar to Contact (1997) and Air Force One (1997) movies respectively.
# 
# ### 6.4.2 Less Number of ratings
# 
# We have a challenge here that some of the movies have very few ratings and may end up being recommended simply because one or two people gave them a 5 star rating. We can fix this by setting a threshold for the number of ratings.

# In[105]:


corr_AFO = corr_AFO.join(ratings['Number of ratings'])
corr_contact = corr_contact.join(ratings['Number of ratings'])
corr_AFO.head()
corr_contact.head()


# # 7. Recommendation output
# 
# ## 7.1
# Now our recommendor system gives the list of movies that are most similar to Air Force One (1997) by limiting them to movies that have at least 100 reviews. We then sort them by the correlation column and view the first 10.

# In[106]:


corr_AFO[corr_AFO['Number of ratings'] > 100].sort_values(by='correlation', ascending=False).head(10)


# We notice that Air Force One (1997) has a perfect correlation with itself, which is not surprising. 
# 
# The next most similar movie to Air Force One (1997) is Major League Back to the Minors (1998) with a correlation of 0.529. 
# 
# Clearly by changing the threshold for the number of reviews we get different results from the previous way of doing it. 
# 
# Limiting the number of rating gives us better results and we can confidently recommend the above movies to someone who has watched Air Force One (1997).

# ## 7.2
# Applying same rule for Contact (1997) movie and check the movies that are most correlated to it.

# In[107]:


corr_contact[corr_contact['Number of ratings'] > 100].sort_values(by='Correlation', ascending=False).head(10)


# The most similar movie to Contact (1997) is Girlfight (2000) with a correlation coefficient of 0.585 with 127 ratings. 
# 
# So if somebody liked Contact (1997) we can recommend the above movies to them.

# # Conclusion
# This is exactly the kind of disruptive innovation that is the reason for success of companies like Amazon, Walmart, Google. All these websites have their recommender system which they customize to each and every visitor.

# # Citations

# https://www.kaggle.com
# 
# 
# https://en.wikipedia.org/wiki
# 
# 
# https://seaborn.pydata.org/
# 
# 
# https://python-graph-gallery.com/

# # Thank you!!
