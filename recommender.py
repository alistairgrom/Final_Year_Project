#!/usr/bin/env python
# coding: utf-8

# ## 964398 - Content Based Movie Recommender
#  ### To run this program go to: Run -> Run All Cells
#  #### This will then run at the bottom of the program

# In[1]:


#To work with dataframes and allow for simple reading of csv files 
import pandas as pd
import numpy as np

#Import TfIdf to extract important features from overview
from sklearn.feature_extraction.text import TfidfVectorizer
#use to calculate the similarity values between movies based on the tfidf
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

#import CountVectorizer to create the count matrix
from sklearn.feature_extraction.text import CountVectorizer
# compute cosine similarity matrix based upon the count matrix
from sklearn.metrics.pairwise import cosine_similarity
import random


# Data prep

# In[2]:


metadata = pd.read_csv('movies_metadata.csv', low_memory=False)
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

pd.set_option('display.max_columns', 10)
percent_sample = 20
percent_sample_tmp = 1 - (percent_sample)/100
m = metadata['vote_count'].quantile(percent_sample_tmp)
metadata = metadata.copy().loc[metadata['vote_count'] >= m]
print(f"Dataset sample size: {metadata.shape[0]}, Top {percent_sample}% of full dataset")


# #### Generate a cosine similarity matrix
# Generate a cosine similarity matrix using the features extracted form the TF-IDF Vectorizer. This is using stop words to eliminate the 'common' words in english that do not give us any relevant info about the movies.

# In[3]:


# create the td-idf vectorizer
# use stop words to remove redundant words
tfidf = TfidfVectorizer(stop_words='english')

# replace empty fields with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# create the tfidf matrix and fit to the overview data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# array mapping from feature integer indices to feature name.
tfidf.get_feature_names()[5000:5010]

# cosine similarity matrix for the tfidf
cosine_sim_tfidf = linear_kernel(tfidf_matrix, tfidf_matrix)

# map the movie titles to their respective indexes
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()


# ### Get Recommendation Function
# This function allows for the user to search via the Title of the movie and recieve X amount of similar movies.

# In[4]:


def get_recommendations(title, cosine_sim=cosine_sim_tfidf):
    # find the index for the film searched by title
    idx = indices[title]

    # get similarity values of movies with current movie
    sim_values = list(enumerate(cosine_sim[idx]))

    # sort by highest sim value
    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)

    # get top 10 most similar
    # 1 is going to be the movie its self
    sim_values = sim_values[1:11]

    # get the movie indices
    movie_indices = [i[0] for i in sim_values]

    # return the titles of the 10 most sim movies
    return metadata['title'].iloc[movie_indices]

def get_recommendations_by_index(idx, cosine_sim=cosine_sim_tfidf):   
    movies = []

    # get similarity values of movies with current movie
    sim_values = list(enumerate(cosine_sim[idx]))

    # sort by highest sim value
    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)

    # get top 10 most similar
    # 1 is going to be the movie its self
    sim_values = sim_values[1:11]

    # get the movie indices
    movie_indices = [i[0] for i in sim_values]

    # return the titles of the 10 most sim movies
    return metadata[['title', 'genres']].iloc[movie_indices]


def get_recommendations_top_5(title, cosine_sim=cosine_sim_tfidf):
    # find the index for the film searched by title
    idx = indices[title]

    
    # get similarity values of movies with current movie
    sim_values = list(enumerate(cosine_sim[idx]))

    # sort by highest sim value
    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)

    # get top 5 most similar
    # 1 is going to be the movie its self
    sim_values = sim_values[1:6]

    # get the movie indices
    movie_indices = [i[0] for i in sim_values]
    
    movies = []
    
    searched_movie = idx
    
    picked_movies.append(searched_movie)
    
    print()
    print("Your List")
    for i in picked_movies:
        print(metadata[(metadata['index'] == i)]['title'].to_string(index=False))
    print(f"\nMovies similar to {title}")
    
    # return the titles of the 5 most sim movies
    return (metadata['title'].iloc[movie_indices].to_string(index=False))


# In[5]:


# convert ids to ints, to merge datasets
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
metadata['id'] = metadata['id'].astype('int')

# use merge to add the credits and keywords datasets into the main dataframe
metadata = metadata.merge(credits, on='id')
metadata = metadata.merge(keywords, on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)


# In[6]:


def get_director(data):
    for i in data:
        if i['job'] == 'Director':
            return i['name']
    return np.nan


# In[7]:


def get_list(data):
    if isinstance(data, list):
        names = [i['name'] for i in data]
        #more than 3 elements exist, return only first three. else then get them all
        if len(names) > 3:
            names = names[:3]
        return names

    return []


# In[8]:


# apply data cleaning functions
metadata['director'] = metadata['crew'].apply(get_director)

features = ['cast', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)


# In[9]:


def clean_data(data):
    if isinstance(data, list):
        return [str.lower(i.replace(" ", "")) for i in data]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(data, str):
            return str.lower(data.replace(" ", ""))
        else:
            return ''
        
# clean the features data
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)


# In[10]:


pd.set_option('display.max_colwidth', None)

def create_bag_of_words(data):
    return ' '.join(data['keywords']) + ' ' + ' '.join(data['cast']) + ' ' + data['director'] + ' ' + ' '.join(data['genres'])

metadata['bag_of_words'] = metadata.apply(create_bag_of_words, axis=1)


# ### Second iterations feature extraction and cosine matrix function

# In[11]:


# using CountVectorizer to get the counts of each keyword
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['bag_of_words'])

# generate similarity matrix
cosine_sim_count_matrix = cosine_similarity(count_matrix, count_matrix)

# reset indexes of main DataFrame and map the indexes like before
metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])


# #### User input handler

# In[12]:


def user_input():
    #cont is true for the program to keep looping 
    cont = True
    while cont:
        print(f"Selection")
        user_input = input("Enter a movie (type 'done' when finished): ")
        #when the user is done they can end the loop and recieve output for all their preferences
        if (user_input.lower() == 'done'):
            cont = False
        else:
            #gives the top 5 most similar and adds the searched for movie
            print(f"{get_recommendations_top_5(user_input, cosine_sim_count_matrix)}\n")


# In[13]:


def getrec(index):
    print(f"{get_recommendations_by_index(index, cosine_sim_count_matrix).to_string(index=False)}")


# In[14]:


def main():
    while True:
        try:
            user_input()
            print()
        except KeyError:
            print('Oops this movie is not in the database.\n')
            continue
        except ValueError:
            print('There has been an error with getting this movie.\n')
            continue
        break


# ### Movies picked by the user.
# 
# This data is used to form the profile of the user, decisions made i.e. 'clicking' on the film feed back to the algorithm, optimizing it with every 'click.'
# This block will cause the entire program to run and produce output

# In[15]:


#resetting the users selections before the program is run again
picked_movies = []

main()

joint_bag_of_words = []

print("-------------------------------------------------------------")
print()
print(f"Your List, generating recommendations from the following...")
for i in picked_movies:
    joint_bag_of_words.append(metadata[metadata['index'] == i].bag_of_words.item())
    print(f"    {metadata[metadata['index'] == i].title.item()}")
    
joint_bag_of_words_str = " ".join(joint_bag_of_words)

this_index = metadata.shape[0]

new_row = {'index': this_index, 'title':'USER', 'bag_of_words':joint_bag_of_words_str}
metadata_2 = metadata.append(new_row, ignore_index=True)

def get_movie_by_index(index):
    movie = metadata[(metadata['index'] == index)]
    return movie

count_2 = CountVectorizer(stop_words='english')
count_matrix_2 = count.fit_transform(metadata_2['bag_of_words'])

cosine_sim_3 = cosine_similarity(count_matrix_2, count_matrix_2)


# reset indexes of main DataFrame and map the indexes like before
indices_2 = pd.Series(metadata_2.index, index=metadata_2['title']).drop_duplicates()

def get_recommendations_for_user(cosine_sim=cosine_sim_3):
    # find the index for the film searched by title
    idx = indices_2['USER']

    # get similarity values of movies with current movie
    sim_values = list(enumerate(cosine_sim[idx]))

     # sort by highest sim value
    sim_values = sorted(sim_values, key=lambda x: x[1], reverse=True)

    # get top 15 most similar
    # 1 is going to be the movie its self
    sim_values = sim_values[1:16]
    
    # get the movie indices
    movie_indices = [i[0] for i in sim_values]
    
    # return the titles of the 5 most sim movies
    return (metadata_2[['title', 'genres']].iloc[movie_indices]).to_string(index=False)

if (len(picked_movies) > 1):
    print()
    print("-------------------------------------------------------------\n")
    print("Recommendations for You based on User Profile")
    print(get_recommendations_for_user(cosine_sim_3))

#Because you liked, the individual movies most similar
for i in range(len(picked_movies)):
    print()
    print("-------------------------------------------------------------\n")
    print(f"Because you liked{get_movie_by_index(picked_movies[i])['title'].to_string(index=False)}")
    getrec(picked_movies[i])

print("\n-------------------------------------------------------------")    

