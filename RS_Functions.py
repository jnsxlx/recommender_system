import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
from scipy.sparse.linalg import svds

from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


### Load data ###

def loadMovieLens(filename='u.data'):
    """Load train data
    """

    str1 = './ml-100k/'

    movie_data = {}
    for line in open(str1 + filename, 'r'):
        (user, movieid, rating, ts) = line.split('\t')
        movie_data.setdefault(user, {})
        movie_data[user][movieid] = float(rating)
    return movie_data


def loadMovieLensTrain(filename='u1.base'):
    """Load train data
    """

    str1 = './ml-100k/'

    movie_data = {}
    for line in open(str1 + filename, 'r'):
        (user, movieid, rating, ts) = line.split('\t')
        movie_data.setdefault(user, {})
        movie_data[user][movieid] = float(rating)
    return movie_data


def loadMovieLensTest(filename='u1.test'):
    """Load test data
    """

    str1 = './ml-100k/'

    movie_data = {}
    for line in open(str1 + filename, 'r'):
        (user, movieid, rating, ts) = line.split('\t')
        movie_data.setdefault(user, {})
        movie_data[user][movieid] = float(rating)
    return movie_data


### Collaborative Filtering ###

def sim_distance(movie_data, p1, p2):
    """
    Returns an Euclidean distance-based similarity score for person1 and person2.
    """

    # Get the list of shared_items
    si = {}
    for item in movie_data[p1]:
        if item in movie_data[p2]:
            si[item] = 1
    # If they have no ratings in common, return 0
    if len(si) == 0:
        return 0
    # Add up the squares of all the differences
    sum_of_squares = sum([pow(movie_data[p1][item] - movie_data[p2][item], 2) for item in
                          movie_data[p1] if item in movie_data[p2]])
    return 1 / (1 + np.sqrt(sum_of_squares))


def sim_pearson(movie_data, p1, p2):
    '''
    Returns the Pearson correlation coefficient for p1 and p2.
    '''

    # Get the list of mutually rated items
    si = {}
    for item in movie_data[p1]:
        if item in movie_data[p2]:
            si[item] = 1
    # If they are no ratings in common, return 0
    if len(si) == 0:
        return 0
    # Sum calculations
    n = len(si)
    # Sums of all the preferences
    sum1 = sum([movie_data[p1][item] for item in si])
    sum2 = sum([movie_data[p2][item] for item in si])
    # Sums of the squares
    sum1Sq = sum([pow(movie_data[p1][item], 2) for item in si])
    sum2Sq = sum([pow(movie_data[p2][item], 2) for item in si])
    # Sum of the products
    pSum = sum([movie_data[p1][item] * movie_data[p2][item] for item in si])
    # Calculate r (Pearson score)
    num = pSum - sum1 * sum2 / n
    den = np.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0
    r = num / den
    return r


def getRecommendations(movie_train, person, item, similarity=sim_pearson, round_opt=True):
    """
    Using Collaborative Filtering(weighted average of other users' ratings) to
    give a prediction of current user's rating
    """

    rat_avg_person = sum(
        movie_train[person].values()) / len(movie_train[person])

    totals = 0
    simSums = 0
    other_cnt = 0  # count how many related users have rated the same item
    for other in movie_train:
        # Don't compare me to myself
        if other == person:
            continue

        sim = similarity(movie_train, person, other)
        # Ignore scores of zero or lower
        if sim <= 0:
            continue

        # Only calculate items also rated in train data
        if item not in movie_train[other]:
            continue

        other_cnt += 1
        # Average rating by others
        rat_avg_other = sum(
            movie_train[other].values()) / len(movie_train[other])

        # Similarity * Score
        # The final score is calculated by weighted average
        totals += (movie_train[other][item] - rat_avg_other) * sim

        # Sum of similarities
        simSums += sim

    # rating = round(rat_avg_person + totals / simSums)
    if other_cnt == 0:
        if round_opt:
            rating = round(rat_avg_person)
        else:
            rating = rat_avg_person
    else:
        if round_opt:
            rating = round(rat_avg_person + totals / simSums)
        else:
            rating = rat_avg_person + totals / simSums

    # Rating is 1 - 5
    if rating <= 1:
        rating = 1
    elif rating >= 5:
        rating = 5

    return rating


### Validation ###

def calMSE(predict_data, movie_test):
	"""
	Calculate MSE between predicted ratings and those in test data
	"""
    mse = 0
    max_mse = 0
    for person in movie_test:
        for item in movie_test[person]:
            mse += (movie_test[person][item] - predict_data[person][item]) ** 2
            if (movie_test[person][item] - predict_data[person][item]) ** 2 > max_mse:
                max_mse = (movie_test[person][item] -
                           predict_data[person][item]) ** 2
                max_person = person
                max_item = item
    print('max mse:', max_mse, 'max person:', max_person, 'max item', max_item)
    mse = mse / sum(map(len, movie_test.values()))
    return mse


def rs_cf(similarity):
	"""
	Cross validate 5 sets
	& Output RMSE
	"""
    movie_train_1 = loadMovieLensTrain('u1.base')
    movie_test_1 = loadMovieLensTest('u1.test')
    predicts = {}
    for user in movie_test_1:
        if int(user) % 100 == 1:
            print('user:', user)
        for item in movie_test_1[user]:
            predicts.setdefault(user, {})
            predicts[user][item] = getRecommendations(
                movie_train_1, user, item, similarity)

    mse_cf_1 = calMSE(predicts, movie_test_1)
    print('MSE of first set:', mse_cf_1)

    movie_train_2 = loadMovieLensTrain('u2.base')
    movie_test_2 = loadMovieLensTest('u2.test')
    predicts = {}
    for user in movie_test_2:
        if int(user) % 100 == 1:
            print('user:', user)
        for item in movie_test_2[user]:
            predicts.setdefault(user, {})
            predicts[user][item] = getRecommendations(
                movie_train_2, user, item, similarity)

    mse_cf_2 = calMSE(predicts, movie_test_2)
    print('MSE of second set:', mse_cf_2)

    movie_train_3 = loadMovieLensTrain('u3.base')
    movie_test_3 = loadMovieLensTest('u3.test')
    predicts = {}
    for user in movie_test_3:
        if int(user) % 100 == 1:
            print('user:', user)
        for item in movie_test_3[user]:
            predicts.setdefault(user, {})
            predicts[user][item] = getRecommendations(
                movie_train_3, user, item, similarity)

    mse_cf_3 = calMSE(predicts, movie_test_3)
    print('MSE of third set:', mse_cf_3)

    movie_train_4 = loadMovieLensTrain('u4.base')
    movie_test_4 = loadMovieLensTest('u4.test')
    predicts = {}
    for user in movie_test_4:
        if int(user) % 100 == 1:
            print('user:', user)
        for item in movie_test_4[user]:
            predicts.setdefault(user, {})
            predicts[user][item] = getRecommendations(
                movie_train_4, user, item, similarity)

    mse_cf_4 = calMSE(predicts, movie_test_4)
    print('MSE of fourth set:', mse_cf_4)

    movie_train_5 = loadMovieLensTrain('u5.base')
    movie_test_5 = loadMovieLensTest('u5.test')
    predicts = {}
    for user in movie_test_5:
        if int(user) % 100 == 1:
            print('user:', user)
        for item in movie_test_5[user]:
            predicts.setdefault(user, {})
            predicts[user][item] = getRecommendations(
                movie_train_5, user, item, similarity)

    mse_cf_5 = calMSE(predicts, movie_test_5)
    print('MSE of fifth set:', mse_cf_5)

    rmse_cf = np.sqrt(
        (mse_cf_1 + mse_cf_2 + mse_cf_3 + mse_cf_4 + mse_cf_5) / 5)
    print('RMSE of 5-fold CV by Collaborative Filtering with Pearson distance:', rmse_cf)


### Inspect MovieLens 100K Data ###

def heatmap(datafile):
	"""
	Use Heatmap to check missing pattern
	"""
    r_cols = ['userid', 'movieid', 'rating', 'ts']
    movie_rating = pd.read_csv('ml-100k/' + datafile, sep='\t', names=r_cols,
                               encoding='latin-1')

    table_rating = movie_rating.pivot(
        index='userid', columns='movieid', values='rating').fillna(0)  # pivot rating table
    # table_rating = pd.pivot_table(movie_rating, values='rating', index='userid', columns='movieid', aggfunc=np.sum, fill_value=0) #pivot rating table with aggfunc

    sns.heatmap(table_rating, xticklabels=False, yticklabels=False)
    plt.show()

    return movie_rating


def movie_bd(datafile):
	"""
	MovieLens Data breakdown by different metrics
	"""
    r_cols = ['userid', 'movieid', 'rating', 'ts']
    movie_rating = pd.read_csv('ml-100k/' + datafile, sep='\t', names=r_cols,
                               encoding='latin-1')
    uprf_cols = ['userid', 'age', 'gender', 'occupation', 'zip_code']
    user_profiles = pd.read_csv('ml-100k/u.user', sep='|', names=uprf_cols,
                                encoding='latin-1')

    mprf_cols = ['movieid', 'movietitle', 'rd', 'videord', 'imdb', 'unknown', 'action', 'advt', 'anm', 'chd', 'comedy',
                 'crime', 'dcmt', 'drama', 'fts', 'fmnr', 'hrr', 'msc', 'mst', 'romance', 'scf', 'thriller', 'war', 'wst']
    movie_profiles = pd.read_csv('ml-100k/u.item', sep='|', names=mprf_cols,
                                 encoding='latin-1')

    user_movie_rating = pd.merge(movie_rating, user_profiles)
    user_movie_rating = user_movie_rating.merge(
        movie_profiles, left_on='movieid', right_on='movieid', how='inner')

    # Number of ratings by movie genres
    user_movie_rating_sum = user_movie_rating[['unknown', 'action', 'advt', 'anm', 'chd', 'comedy', 'crime',
                                               'dcmt', 'drama', 'fts', 'fmnr', 'hrr', 'msc', 'mst', 'romance', 'scf', 'thriller', 'war', 'wst']].sum(axis=0)
    sns.set(style="white", context="talk")
    f, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    x = np.array(['unknown', 'action', 'advt', 'anm', 'chd', 'comedy', 'crime', 'dcmt', 'drama',
                  'fts', 'fmnr', 'hrr', 'msc', 'mst', 'romance', 'scf', 'thriller', 'war', 'wst']).T.flatten()
    y1 = np.array(list(user_movie_rating_sum)).flatten()
    sns.barplot(x, y1, palette="BuGn_d", ax=ax1)
    ax1.set_ylabel("Count")
    sns.despine(bottom=True)
    # plt.setp(f.axes, yticks=[])
    plt.tight_layout(h_pad=3)
    plt.show()

    # Movie genres breakdown
    # Action breakdown
    action_data = user_movie_rating.loc[user_movie_rating['action'] == 1]
    action_data = action_data[['userid', 'rating']]
    action_rt = action_data.groupby('rating').count()

    # Comedy breakdown
    comedy_data = user_movie_rating.loc[user_movie_rating['comedy'] == 1]
    comedy_data = comedy_data[['userid', 'rating']]
    comedy_rt = comedy_data.groupby('rating').count()

    # Drama breakdown
    drama_data = user_movie_rating.loc[user_movie_rating['drama'] == 1]
    drama_data = drama_data[['userid', 'rating']]
    drama_rt = drama_data.groupby('rating').count()

    # Romance breakdown
    romance_data = user_movie_rating.loc[user_movie_rating['romance'] == 1]
    romance_data = romance_data[['userid', 'rating']]
    romance_rt = romance_data.groupby('rating').count()

    # Thriller breakdown
    thriller_data = user_movie_rating.loc[user_movie_rating['thriller'] == 1]
    thriller_data = thriller_data[['userid', 'rating']]
    thriller_rt = thriller_data.groupby('rating').count()

    # Plot the distribution
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(
        5, 1, figsize=(8, 6), sharex=True)
    # Generate some sequential data
    x = np.array(list("12345"))
    y1 = np.array(action_rt).reshape(5,)
    sns.barplot(x, y1, palette="BuGn_d", ax=ax1)
    ax1.set_ylabel("Action")

    y2 = np.array(comedy_rt).reshape(5,)
    sns.barplot(x, y2, palette="BuGn_d", ax=ax2)
    ax2.set_ylabel("Comedy")

    y3 = np.array(drama_rt).reshape(5,)
    sns.barplot(x, y3, palette="BuGn_d", ax=ax3)
    ax3.set_ylabel("Drama")

    y4 = np.array(romance_rt).reshape(5,)
    sns.barplot(x, y4, palette="BuGn_d", ax=ax4)
    ax4.set_ylabel("Romance")

    y5 = np.array(thriller_rt).reshape(5,)
    sns.barplot(x, y5, palette="BuGn_d", ax=ax5)
    ax5.set_ylabel("Thriller")

    # Finalize the plot
    sns.despine(bottom=True)
    plt.tight_layout(h_pad=3)
    plt.show()



### Simple SVD ###

"""
Load data for SVD
"""

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users_df = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols,
                       encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv('ml-100k/u1.base', sep='\t', names=r_cols,
                         encoding='latin-1')

test_ratings_df = pd.read_csv('ml-100k/u1.test', sep='\t', names=r_cols,
                              encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(5),
                        encoding='latin-1')

def rs_svd(movies_df, ratings_df, k):
	"""
	Predict unknown ratings with SVD
	"""
    movies_df['movie_id'] = movies_df['movie_id'].apply(pd.to_numeric)

    R_df = ratings_df.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

    mtx_rating = R_df.as_matrix()
    mean_rating = np.mean(mtx_rating, axis=1)
    rating_demeaned = mtx_rating - mean_rating.reshape(-1, 1)

    U, sigma, Vt = svds(rating_demeaned, k)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + mean_rating.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns, index=R_df.index)

    preds = round(preds_df.clip(lower=1, upper=5))
    #
    # preds_df = round(preds_df) * (preds_df > 0)
    # preds_df = preds_df * ((preds_df - 5) < 0)
    return preds


def calMSE_svd(preds_df, movie_test):
	"""
	Calculate MSE for SVD
	"""
    rmse_sum = 0
    max_rmse = 0
    # ratingarr = []
    # useridarr = []
    # movieidarr = []
    for i in range(0, movie_test.shape[0]):
        userid = movie_test['user_id'][i] - 1
        movieid = movie_test['movie_id'][i] - 1
        rating_true = movie_test['rating'][i]
        rmse_sum += (rating_true - preds_df.iat[userid, movieid])**2
        # useridarr.append(userid)
        # movieidarr.append(movieid)
        # ratingarr.append(preds_df.iat[userid, movieid])
    #     if (rating_true - preds_df.iat[userid, movieid])**2 > max_rmse:
    #         max_rmse = (rating_true - preds_df.iat[userid, movieid])**2
    #         max_person = userid
    #         max_item = movieid
    # print('max rmse:', max_rmse, 'max person:', max_person+1, 'max item:', max_item+1, 'predict:', preds_df.iat[userid, movieid])
    rmse = rmse_sum / movie_test.shape[0]

    # est = {'user': np.array(useridarr), 'movie': np.array(movieidarr), 'rating': np.array(ratingarr)}
    return rmse


### Run SVD ###
"""
Not included in the Jupyter notebook
"""

# num_iter = 100
# rmse_min = 10
# for i in range(3, num_iter):
#     preds_df = rs_svd(movies_df, ratings_df, i)
#     rmse_svd = calMSE_svd(preds_df, test_ratings_df)
#     if rmse_svd < rmse_min:
#         min_k = np.copy(i)
#         rmse_min = np.copy(rmse_svd)
# print('Smallest RMSE is:', rmse_min, 'acquired by:', min_k, ' svd')
# preds_df = rs_svd(movies_df, ratings_df, 13)
# rmse_svd = calMSE_svd(preds_df, test_ratings_df)
# print(rmse_svd)


### Repeated Matrix Reconstruction method ###


def pd_read(trainfile, testfile):
	"""
	Load data
	"""
    r_cols = ['userid', 'movieid', 'rating', 'unix_timestamp']
    train = pd.read_csv('ml-100k/' + trainfile, sep='\t',
                        names=r_cols, encoding='latin-1')
    test = pd.read_csv('ml-100k/' + testfile, sep='\t', names=r_cols, encoding='latin-1',
                       usecols=['userid', 'movieid', 'unix_timestamp'])
    test_org = pd.read_csv('ml-100k/' + testfile, sep='\t',
                           names=r_cols, encoding='latin-1')
    return train, test, test_org


def RMRM(set_i, train, test, test_org, round_opt=True, n_compo=15, random_s=42, max_iteration=10):
	"""
	Use Repeated Matrix Reconstruction method to predict ratings
	Output MSE
	"""
    matrix = pd.concat([train, test]).pivot('userid', 'movieid', 'rating')
    movie_means = matrix.mean()
    user_means = matrix.mean(axis=1)
    mzm = matrix - movie_means  # Standardized Original rating matrix
    mz = mzm.fillna(0)  # Add 0s to Original matrix
    mask = -mzm.isnull()  # Original mtx with values

    iteration = 0
    mse_last = 999
    while iteration < max_iteration:
        iteration += 1
        svd = TruncatedSVD(n_components=n_compo, random_state=random_s)
        svd.fit(mz)
        mzsvd = pd.DataFrame(svd.inverse_transform(svd.transform(
            mz)), columns=mz.columns, index=mz.index)  # Run SVD for mz
        mse = mean_squared_error(mzsvd[mask].fillna(0), mzm[mask].fillna(0))
        # print('%i %.5f %.5f' % (iteration, mse, mse_last-mse))
        mzsvd[mask] = mzm[mask]  # Put actual values to the predicted matrix

        mz = mzsvd
        if mse_last - mse < 0.00001:
            break
        mse_last = mse

    m = mz + movie_means
    m = m.clip(lower=1, upper=5)

    if round_opt:
        test['rating'] = round(test.apply(
            lambda x: m[m.index == x.userid][x.movieid].values[0], axis=1))
    else:
        test['rating'] = test.apply(
            lambda x: m[m.index == x.userid][x.movieid].values[0], axis=1)

    # There are some movies who did not have enough info to make prediction, so just used average value for user
    # missing = np.where(test.rating.isnull())[0]
    # test.ix[missing, 'rating'] = user_means[test.loc[missing].userid].values

    mse = np.sum((test_org['rating'] - test['rating'])
                 ** 2) / len(test_org['rating'])
    print('MSE of', set_i + 1, 'set:', mse)
    return mse


def sub_group(max_k):
    """
    Group by number of ratings
    Check if MSE is influenced by the quantity of ratings
    """
    r_cols = ['userid', 'movieid', 'rating', 'unix_timestamp']
    train = pd.read_csv('ml-100k/' + trainfile, sep='\t',
                        names=r_cols, encoding='latin-1')
    test = pd.read_csv('ml-100k/' + testfile, sep='\t', names=r_cols, encoding='latin-1',
                       usecols=['userid', 'movieid', 'unix_timestamp'])
    max_k = 2
    interval_range = [0, 200, 700]
    grp = train.groupby(['userid'])
    grp = grp.count()
    grp_data = pd.cut(grp.movieid, bins=interval_range,
                      labels=list((range(1, max_k + 1))))
    grp_data = grp_data.to_frame().reset_index()
    grp_data.columns = ['userid', 'count']

    train_grp = train.merge(grp_data, left_on='userid',
                            right_on='userid', how='inner')
    test_grp = test.merge(grp_data, left_on='userid',
                          right_on='userid', how='inner')
    test_org_grp = test_org.merge(
        grp_data, left_on='userid', right_on='userid', how='inner')

    train_subgrp = []
    test_subgrp = []
    test_org_subgrp = []
    for j in range(6):
        train_subgrp.append(train_grp.loc[train_grp['count'] == j + 1])
        test_subgrp.append(test_grp.loc[test_grp['count'] == j + 1])
        test_org_subgrp.append(
            test_org_grp.loc[test_org_grp['count'] == j + 1])

    mse = []
    for k in range(max_k):
        mse.append(RMRM(set_i=k, train=train_subgrp[k], test=test_subgrp[k], test_org=test_org_subgrp[k],
                        n_compo=15, random_s=42, max_iteration=1) * len(test_org_subgrp[k]))
        print('Subgroup length:', len(test_subgrp[k]))
    # mse = mse/len(test_org['rating'])

    print('Subgroup MSE:', mse)
    mse_r = sum(mse) / len(test_org_grp['rating'])
    print('The resulting MSE is:', mse_r)

    return mse_r

### Run to check Sub Group MSE###

# r_cols = ['userid', 'movieid', 'rating', 'unix_timestamp']
# train = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
# grp = train.groupby(['userid'])
# grp = grp.count()
# sns.distplot(grp.rating, axlabel="User's number of ratings")
# plt.show()
#
#
#
# grp_mv = train.groupby(['movieid'])
# grp_mv_ct = grp_mv.count()
# grp_mv_mn = grp_mv.mean()
# ax = sns.regplot(x=grp_mv_ct.userid, y=grp_mv_mn.rating, color="g", fit_reg=False)
# plt.show()
#
# # sns.distplot(grp.rating, axlabel="User's number of ratings")
# # plt.show()


# """
# Test part for prediction
# """
# predicts = {}
# user = '5'
#
# for item in movie_test[user]:
#     predicts.setdefault(user, {})
#     predicts[user][item] = getRecommendations(movie_train, user, item, similarity=sim_pearson)
#
# rmse = 0
# max_rmse = 0
# for item in movie_test[user]:
#     rmse += (movie_test[user][item] - predicts[user][item]) ** 2
#     if (movie_test[user][item] - predicts[user][item]) ** 2 > max_rmse:
#         max_rmse = (movie_test[user][item] - predicts[user][item]) ** 2
#         max_person = user
#         max_item = item
# print('max rmse:', max_rmse, 'max person:', max_person, 'max item', max_item, 'predict value', predicts[max_person][max_item])
# rmse = rmse / len(movie_test['1'].values())
# print(rmse)
#
# print(getRecommendations(movie_train, '1', '143', similarity=sim_pearson))


# print(preds_df.iloc[0, 11])
# print(preds_df.idxmax())
# print(preds_df.iat[0, 5])
