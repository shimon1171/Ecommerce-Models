from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from scipy.sparse.linalg import inv
import Timer


class ModelData:
    """The class reads 5 files as specified in the init function, it creates basic containers for the data.
    See the get functions for the possible options, it also creates and stores a unique index for each user and movie
    """

    def __init__(self, train_x, train_y, test_x, test_y, movie_data):
        """Expects 4 data set files with index column (train and test) and 1 income + genres file without index col"""
        self.train_x = pd.read_csv(train_x, index_col=[0])
        self.train_y = pd.read_csv(train_y, index_col=[0])
        self.test_x = pd.read_csv(test_x, index_col=[0])
        self.test_y = pd.read_csv(test_y, index_col=[0])
        self.movies_data = pd.read_csv(movie_data)
        self.users = self._generate_users()
        self.movies = self._generate_movies()
        self.incomes = self._generate_income_dict()
        self.movie_index = self._generate_movie_index()
        self.user_index = self._generate_user_index()
        self._matix_a_index()


    def _matix_a_index(self):

        movies = sorted(set(self.train_x['movie']) | set(self.test_x['movie']))
        users = sorted(set(self.train_x['user']) | set(self.test_x['user']))

        index = 0
        self.matix_a_index_movie = {}
        for movie in movies:
            self.matix_a_index_movie[movie] = index
            index += 1
        self.matix_a_index_users = {}
        for user in users:
            self.matix_a_index_users[user] = index
            index += 1

        index = 0
        self.matix_a_index_users_with_income = {}
        for user in users:
            self.matix_a_index_users_with_income[user] = index
            index += 1

    def _generate_users(self):
        users = sorted(set(self.train_x['user']))
        return tuple(users)

    def _generate_movies(self):
        movies = sorted(set(self.train_x['movie']))
        return tuple(movies)

    def _generate_income_dict(self):
        income_dict = defaultdict(float)
        average_income = np.float64(self.movies_data['income'].mean())
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

    def get_users(self):
        """:rtype tuples of all users"""
        return self.users

    def get_movies(self):
        """:rtype tuples of all movies"""
        return self.movies

    def get_movie_index(self, movie):
        """:rtype returns the index of the movie if it exists or None"""
        return self.movie_index.get(movie, None)

    def get_user_index(self, user):
        """:rtype returns the index of the user if it exists or None"""
        return self.user_index.get(user, None)

    def get_movie_income(self, movie):
        """:rtype returns the income of the movie if it exists or None"""
        if self.incomes.get(movie, None) is None:
            print(movie)
        return self.incomes.get(movie, None)

    def get_movies_for_user(self, user):
        return self.train_x[self.train_x['user'] == user]['movie'].values

    def get_users_for_movie(self, movie):
        return self.train_x[self.train_x['movie'] == movie]['user'].values

"""Construct the coefficients matrix A"""
def create_coefficient_matrix(train_x, data: ModelData = None):
    matrix_timer = Timer.Timer('Matrix A creation')
    # TODO: Modify this function to return the coefficient matrix A as seen in the lecture (slides 24 - 37).
    movies = data.get_movies()
    users = data.get_users()
    number_of_colunm = max(data.matix_a_index_users.values()) + 1
    number_of_rows = len(train_x)

    dic = {}
    matix_a_index_movie = data.matix_a_index_movie
    matix_a_index_users = data.matix_a_index_users
    row_index = 0
    for index, row in train_x.iterrows():
        movie = row['movie']
        j = matix_a_index_movie[movie]
        tuple = (row_index,j)
        dic[tuple] = 1

        user = row['user']
        j = matix_a_index_users[user]
        tuple = (row_index, j)
        dic[tuple] = 1
        row_index += 1

    for i in range(number_of_colunm):
        tuple = (0, i)
        if tuple in dic:
            continue
        else:
            dic[tuple] = 0.001 * np.random.random()

    row = []
    col = []
    data_values = []
    for tuple in dic.keys():
        value = dic[tuple]
        row.append(tuple[0])
        col.append(tuple[1])
        data_values.append(value)
        if tuple[1] > number_of_colunm:
            print(tuple[1])
            print(tuple)

    matrix_a = csr_matrix((data_values, (row, col)), shape=((number_of_rows,number_of_colunm)))

    matrix_timer.stop()

    return matrix_a


def create_coefficient_matrix_with_income(train_x, data: ModelData = None):
    matrix_timer = Timer.Timer('Matrix A creation')
    # TODO: Modify this function to return the coefficient matrix A as seen in the lecture (slides 24 - 37).
    users = data.get_users()
    number_of_colunm =  len(users) + 1
    number_of_rows = len(train_x)

    dic = {}
    matix_a_index_users = data.matix_a_index_users_with_income
    incomes = data.incomes
    row_index = 0
    for index, row in train_x.iterrows():
        user = row['user']
        j = matix_a_index_users[user]
        tuple = (row_index, j)
        dic[tuple] = 1
        movie = row['movie']
        income = incomes[movie]
        tuple = (row_index, number_of_colunm-1)
        dic[tuple] = income
        row_index += 1

    for i in range(number_of_colunm):
        tuple = (0, i)
        if tuple in dic:
            continue
        else:
            dic[tuple] = 0.001 * np.random.random()


    row = []
    col = []
    data_values = []
    for tuple in dic.keys():
        value = dic[tuple]
        row.append(tuple[0])
        col.append(tuple[1])
        data_values.append(value)

    matrix_a = csr_matrix((data_values, (row, col)), shape=((number_of_rows, number_of_colunm)))

    matrix_timer.stop()

    return matrix_a


"""Construct vector C from the train set as (r_i - r_avg) """ # done
def construct_rating_vector(train_y, r_avg):
    df = train_y.copy()
    df['c'] = df['rate'] - r_avg
    c = list(df['c'])

    row = []
    col = []
    data =  np.array(c, dtype=float)
    for i in range(len(c)):
        row.append(i)
        col.append(0)


    vector_c = csr_matrix((data, (row, col)), shape=(len(c),1))
    return vector_c

"""" this function to return vector b* """
def fit_parameters(matrix_a, vector_c):
    A = matrix_a
    At = A.transpose()
    AtA = At.dot(A)
    AtC = At.dot(vector_c)
    AtA = AtA.toarray()
    AtC = AtC.toarray()
    res = np.linalg.solve(AtA, AtC)
    b = []
    for i in res:
        b.append(i[0])

    return b

"""Calc the basic parameters vector and run the basic model r_hat = r_avg + b_u + b_i""" # done
def calc_parameters(r_avg, train_x, train_y, data: ModelData = None):
    matix_a_index_movie = data.matix_a_index_movie
    matix_a_index_users = data.matix_a_index_users

    b = [0 for _ in range(len(matix_a_index_movie) + len(matix_a_index_users))]

    group_by_user = train_x.groupby(by='user')
    for user_name, user_group in group_by_user:
        indexes = user_group.index.values.tolist()
        rate_df = train_y.loc[indexes]
        r_u_avg = rate_df['rate'].mean()
        b_u = (r_u_avg - r_avg)
        key = matix_a_index_users[user_name]
        b[key] = b_u

    group_by_movie = train_x.groupby(by='movie')
    for movie_name, movie_group in group_by_movie:
        indexes = movie_group.index.values.tolist()
        rate_df = train_y.loc[indexes]
        r_i_avg = rate_df['rate'].mean()
        b_i = (r_i_avg - r_avg)
        key = matix_a_index_movie[movie_name]
        b[key] = b_i
    return b


"""Calculate the average rating from the train set""" # done
def calc_average_rating(train_y):
    r_avg = train_y['rate'].mean()
    return r_avg

"""this function to return the predictions list ordered by the same index as in argument test_x""" # done
def model_inference(test_x, vector_b, r_avg, data: ModelData = None):
    matix_a_index_movie = data.matix_a_index_movie
    matix_a_index_users = data.matix_a_index_users

    predictions_list = []
    for i in test_x.index:
        df = test_x.loc[i]
        user_name = df['user']
        key = matix_a_index_users[user_name]
        user_b = vector_b[key]

        movie_name = df['movie']
        matix_a_index_movie[movie_name]
        movie_b = vector_b[key]

        r = r_avg + user_b + movie_b
        predictions_list += [r]

    return predictions_list


def model_inference_with_income(test_x, vector_b, r_avg, data: ModelData = None):
    # TODO: Modify this function to return the predictions list ordered by the same index as in argument test_x
    # TODO: based on the modified model with income

    matix_a_index_users = data.matix_a_index_users_with_income
    len_b = len(vector_b)
    incomes = data.incomes
    bI = vector_b[len_b - 1]
    predictions_list = []
    for i in test_x.index:
        df = test_x.loc[i]
        user_name = df['user']
        key = matix_a_index_users[user_name]
        user_b = vector_b[key]

        movie_name = df['movie']
        income = incomes[movie_name]
        Im = income * ( 1.0 / pow(10,10) )

        r = r_avg + user_b + bI*Im
        predictions_list += [r]

    return predictions_list


""""Calc the RMSE """ # done
def calc_error_old(predictions_df, test_df):
    sum = 0
    for i in predictions_df.index:
        df = predictions_df.loc[i]
        r_hat = df['r_hat']
        df = test_df.loc[i]
        rate = df['rate']
        sum += ( r_hat - rate ) ** 2
    sum = float(sum) / len(predictions_df)
    rmse = np.sqrt(sum)
    return rmse

def calc_error(predictions_df, test_df):
    indexes = list(predictions_df.index)
    rate_df = test_df.loc[indexes]
    p_df = predictions_df.copy()
    p_df['rate'] = rate_df['rate']
    p_df['power'] = ( p_df['r_hat'] - p_df['rate'] ) ** 2
    sum = p_df['power'].mean()
    rmse = np.sqrt(sum)
    return rmse



""""Calc the avg RMSE for each movie""" # done
def calc_avg_error(predictions_df, test_df):
    m_error = {}
    group_by_movie = predictions_df.groupby(by='movie')
    for movie_name, movie_group in group_by_movie:
        rmse = calc_error(movie_group,test_df )
        number_of_users = len(movie_group)
        m_error[movie_name] = (rmse, number_of_users)
    return m_error


# this function to plot the graph y:(RMSE of movie i) x:(number of users rankings)
def plot_error_per_movie(movie_error):
    list1 = sorted(list(movie_error.values()), key=lambda tup: tup[1])
    x_values = []
    y_values = []
    for tuple in list1:
        x_values.append(tuple[1])
        y_values.append(tuple[0])
    file_name = "Empyt"
    rcParams.update({'figure.autolayout': True})
    plt.title('RMSE as function of rankings')
    plt.xlabel('Number of users rankings')
    plt.ylabel("RMSE", rotation=90)
    plt.plot(x_values, y_values, '-bo')
    plt.subplots_adjust(top=0.8)
    if (file_name != "Empyt"):
        plt.savefig(file_name, format='png', dpi=600)
    plt.close()


###########################################################################
###########################################################################
######### Create Test and train files ####################################

def create_test_and_train_files(data_file_name):
    data_file_name = 'dataset.csv'
    df = pd.read_csv(data_file_name)
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    split_dataset_to_x_and_y(train,'train_x.csv', 'train_y.csv')
    test = df[~msk]
    split_dataset_to_x_and_y(test,'test_x.csv', 'test_y.csv')

def split_dataset_to_x_and_y(df , x_file , y_file):
    x_df = df[['user','movie']]
    y_df = df[['rate']]
    x_df.to_csv(x_file)
    y_df.to_csv(y_file)


###########################################################################
###########################################################################
######### Part 1 ####################################

if __name__ == '__main__':
    x =1
