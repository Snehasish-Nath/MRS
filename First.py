#Try

"""
In this machine learning project,
we build a recommendation system from the ground up to suggest movies to the user based on his/her preferences.
"""

#Starting with the most important libraries

import pandas as pd
import numpy as np

#Reading the files using read_csv function and storing them in different variables_
movies=pd.read_csv("tmdb_5000_movies.csv")

credits=pd.read_csv("tmdb_5000_credits.csv")

movies.head()
credits.head()

movies=movies.merge()

