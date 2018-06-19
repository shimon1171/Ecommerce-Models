from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
import Timer


class Competition:

    def __init__(self, data, movie_data):
        self.data = pd.read_csv(data)
        self.movies_data = pd.read_csv(movie_data)
        self.users = self._generate_users()
        self.movies = self._generate_movies()
        #self.incomes = self._generate_income_dict()
        self.movie_index = self._generate_movie_index()
        self.user_index = self._generate_user_index()

    def _generate_users(self):
        users = sorted(set(self.data['user']))
        return tuple(users)

    def _generate_movies(self):
        movies = sorted(set(self.data['movie']))
        return tuple(movies)

    def _generate_income_dict(self):
        income_dict = defaultdict(float)
        movies = sorted(set(self.movies_data['movie']))
        for m in movies:
            movie_income = int(self.movies_data[self.movies_data['movie'] == m]['income'])
            income_diff = movie_income * 10e-10
            income_dict[m] = income_diff
        return income_dict

    def _generate_user_index(self):
        user_index = defaultdict(int)
        for i, u in enumerate(self.users):
            user_index[u] = i
        return user_index

    def _generate_movie_index(self):
        movie_index = defaultdict(int)
        for i, m in enumerate(self.movies, start=len(self.users)):
            movie_index[m] = i
        return movie_index


    def calcualte_user_sim(self):

        df = self.data.copy()
        dic = {}
        for user1 in self.user_index.keys():
             for user2 in self.user_index.keys():
                t =  (user1 , user2)
                t_op = (user2 , user1)

                if (t in dic) or (t_op in dic) :
                    continue

                if user1 == user2:
                    dic[t] = 1
                    continue
                else:
                    udf = df[ (df['user'] == user1) | (df['user'] == user2)].copy()
                    if len(udf) == 0:
                        dic[t] = 0
                        continue
                    movies = list(set(udf[(df['user'] == user1)]['movie']) |  set(udf[(df['user'] == user2)]['movie']))
                    udf = udf[ udf['movie'].isin(movies) ]

                    user1_mean = udf[udf['user'] == user1]['rate'].mean()
                    udf_user1 = df[df['user'] == user1]
                    udf_user1['rating_adjusted1'] = udf_user1['rate'] - user1_mean
                    user1_val = np.sqrt(np.sum(np.square(udf_user1['rating_adjusted1'])))


                    user2_mean = udf[udf['user'] == user2]['rate'].mean()
                    udf_user2 = df[df['user'] == user2]
                    udf_user2['rating_adjusted2'] = udf_user2['rate'] - user2_mean
                    user2_val = np.sqrt(np.sum(np.square(udf_user2['rating_adjusted2'])))

                    udf['vector_product']=(udf['rating_adjusted1']*udf['rating_adjusted2'])
                    sum = udf['vector_product'].sum()
                    sim = float(sum) / (user1_val * user2_val)

                dic[t] = sim

        return dic



if __name__ == '__main__':
    data = Competition('dataset.csv', 'movie_data.csv')
    data.calcualte_user_sim()
