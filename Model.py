#Try

"""
In this machine learning project,
we build a recommendation system from the ground up to suggest movies to the user based on his/her preferences.
"""

#Starting with the most important libraries

import pandas as pd
import numpy as np
import ast


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies.head(1)
credits.head(1)


movies = movies.merge(credits,on='title')

movies.head(1)


#genres
#id
#keywords
#title
#overview
#cast
#crew

movies=movies[['movie_id', 'title', 'overview','genres', 'keywords', 'cast', 'crew']]


movies.head(1)

movies.isnull().sum()


movies.dropna(inplace=True)#Removing the missing values movie as the number is very less


movies.duplicated().sum()


movies.iloc[0].genres

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres']=movies['genres'].apply(convert)


movies.head()


movies['keywords']=movies['keywords'].apply(convert)


movies.head()


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L



movies['cast'].apply(convert3)



movies['cast']=movies['cast'].apply(convert3)


movies.head()

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break
    return L



movies['crew']=movies['crew'].apply(fetch_director)


movies.head()


movies['overview']=movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies.head()


new_df=movies[['movie_id','title','tags']]


new_df



new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


new_df.head()


import nltk



from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()




def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)



new_df['tags'][0]


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())



new_df.head()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')



vectors=cv.fit_transform(new_df['tags']).toarray()



vectors


vectors[0]




from sklearn.metrics.pairwise import cosine_similarity


similarity=cosine_similarity(vectors)


similarity

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


recommend('Avatar')

import pickle

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


pickle.dump(similarity,open('similarity.pkl','wb'))







