#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

movies=pd.read_csv("movies.csv")


# In[2]:


movies


# In[3]:


# removal of special characters from the title

import re

def clean_title(title):
    return re.sub("[^a-zA-Z0-9]", " ", title)


# In[4]:


movies["clean_title"]=movies["title"].apply(clean_title)


# In[5]:


movies


# In[6]:


# to build the tfidf matrix and get the unique word

from sklearn.feature_extraction.text import TfidfVectorizer

# ngram_range is used to not only check the words individually but also as a group 
# for example TOY STORY this will not only check toy and story individually but also chek them
# as how many times TOY and STORY have occured together.

vectorizer= TfidfVectorizer(ngram_range=(1,2))

tfidf= vectorizer.fit_transform(movies["clean_title"])


# In[7]:


# here we use cosine similarity to find the similarity between the vector of the inserted movie name
# and the tfidf that is the clean title 

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title=clean_title(title)
    query_vec=vectorizer.transform([title])
    similarity=cosine_similarity(query_vec, tfidf).flatten()
    indices=np.argpartition(similarity,-5)[-5:]
    # the above code gives the top 5 realted movies which is in reverse order
    # so we use [::-1] to reveerse back the indices 
    results=movies.iloc[indices][::-1]
    return results


# In[8]:


import ipywidgets as widgets
from IPython.display import display

movie_input=widgets.Text(
    value="",
    description="Enter the title",
    disabled=False
)

movie_list=widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        # here we have the dictionary value and the new field will give us the new
        # value that was entered in the input widget
        title=data["new"]
        if len(title)>5:
            display(search(title))
            
movie_input.observe(on_type,names='value')

display(movie_input, movie_list)

            

            
    


# In[9]:


ratings= pd.read_csv("ratings.csv")
ratings


# In[10]:


ratings.dtypes


# In[11]:


# here we try to find out user who have given ratings 4 and above to the movie and like the movie asuming that 4 
#is the rating for liking the movie
movie_id=1
similar_users=ratings[(ratings["movieId"] == movie_id) & (ratings["rating"]>=4)]["userId"].unique()


# In[12]:


similar_users


# In[13]:


# here we find different movies that the user has liked (users from the similar_users)
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>=4)]["movieId"]


# In[14]:


similar_user_recs


# In[15]:


# we are going to look for the movies from the similar users only till 10% that are similar to our movie as well

similar_user_recs= similar_user_recs.value_counts() /len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs> 0.1]


# In[16]:


similar_user_recs


# In[17]:


all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]
    
rec_percentages


# In[18]:


def find_similar_movies(movie_id):
    # here we try to find out user who have given ratings 4 and above to the movie and like the movie asuming that 4 
    #is the rating for liking the movie
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    
    # here we find different movies that the user has liked (users from the similar_users) 
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
     # we are going to look for the movies from the similar users only till 10% that are similar to our movie as well
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# In[19]:


import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)


# In[ ]:





# In[ ]:




