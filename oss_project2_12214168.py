'''
이름: 김유현
학과: 인공지능공학과
학번:12214168

데이터셋 출처

Harper, F. Maxwell, and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015),
19 pages. DOI: http://dx.doi.org/10.1145/2827872

'''

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.cluster import KMeans

sample = pd.read_csv('ratings.dat', sep='::', header=None, engine='python',
                     names = ['UserID', 'MovieID', 'Rating', 'Timestamp'])

ratings_pd = sample.pivot(index='UserID', columns='MovieID', values='Rating') # movieID 중 빈 값이 있음

new_index = range(1, ratings_pd.index.max() + 1)
new_columns = range(1, ratings_pd.columns.max() + 1)

ratings_pd_new = pd.DataFrame(0, index=new_index, columns=new_columns)
ratings_pd_new.loc[ratings_pd.index, ratings_pd.columns] = ratings_pd.values

ratings = ratings_pd_new.fillna(0).values # ndarray
# 클러스터링을 위해 nan을 일단 0으로 채움

kmeans = KMeans(n_clusters=3, random_state=0).fit(ratings)
labels = kmeans.labels_

labels = labels + 1 # 라벨 구간을 1에서 3으로 조정

clustered_users = {i: (np.where(labels == i)[0] + 1) for i in range(1, 4)}

ratings = np.where(ratings == 0, np.nan, ratings)
# 다시 0을 nan으로 바꿈

def additive_utilitarian(ratings, grouped_users):
    print("AU")
    for group_num, users in grouped_users.items():
        user_ratings = ratings[users - 1, :]  # 0부터 시작하는 인덱싱 조정
        total_ratings = np.nansum(user_ratings, axis=0)
        # 단순합으로 상위 10개라서 일단 평점의 개수가 많으면 유리하다.

        top_movies = np.argsort(total_ratings)[-10:][::-1] + 1
        print("Group{}의 상위 10개 영화:".format(group_num), top_movies)

        # top_values = total_ratings[np.argsort(total_ratings)[-10:][::-1]]
        # print("해당 인덱스의 total_ratings 값:",end=' ')
        # print(top_values)

    print()

def average(ratings, grouped_users):
    print("AVG")
    for group_num, users in grouped_users.items():
        user_ratings = ratings[users - 1, :]
        mean_ratings = np.nanmean(user_ratings, axis=0)
        mean_ratings[np.isnan(mean_ratings)] = 0
        top_movies = np.argsort(mean_ratings)[-10:][::-1] + 1
        # 단순히 평균으로 상위 10개의 영화를 추천하기 때문에 평점의 개수가 적어 평균 평점이 거의 5점인 영화가 추천된다
        print("Group{}의 상위 10개 영화:".format(group_num), top_movies)

        # top_values = mean_ratings[np.argsort(mean_ratings)[-10:][::-1]]
        # print("해당 인덱스의 mean_ratings 값:",end=' ')
        # print(top_values)

    print()

def simple_count(ratings, grouped_users):
    print("Simple Count")
    for group_num, users in grouped_users.items():
        user_ratings = ratings[users - 1, :]
        simple_count = np.where(np.isnan(user_ratings), 0, 1)
        simple_count_sum = simple_count.sum(axis=0)
        top_movies = np.argsort(simple_count_sum)[-10:][::-1] + 1
        print("Group{}의 상위 10개 영화:".format(group_num), top_movies)

        # print("해당 인덱스의 simple_count_sum 값:",end=' ')
        # top_values = simple_count_sum[np.argsort(simple_count_sum)[-10:][::-1]]
        # print(top_values)

    print()

def approval_voting(ratings, grouped_users):
    print("Approval Voting")
    for group_num, users in grouped_users.items():
        user_ratings = ratings[users - 1, :]
        threshold = 4
        cond = (~np.isnan(user_ratings)) & (user_ratings >= threshold)
        av_count = np.where(cond, 1, 0)
        av_count_sum = av_count.sum(axis=0)
        top_movies = np.argsort(av_count_sum)[-10:][::-1] + 1
        print("Group{}의 상위 10개 영화:".format(group_num), top_movies)

        # print("해당 인덱스의 av_count_sum 값:",end=' ')
        # top_values = av_count_sum[np.argsort(av_count_sum)[-10:][::-1]]
        # print(top_values)

    print()

def borda_count(ratings, grouped_users):
    print("Borda Count")
    for group_num, users in grouped_users.items():
        user_ratings = ratings[users - 1, :]
        df = pd.DataFrame(user_ratings)
        ranks = df.rank(axis=1, na_option='keep')
        ranks = ranks - 1

        ranks_array = ranks.values  # ndarray
        borda_scores = np.nansum(ranks_array, axis=0)
        top_movies = np.argsort(borda_scores)[-10:][::-1] + 1
        print("Group{}의 상위 10개 영화:".format(group_num), top_movies)
        # print("해당 인덱스의 borda_scores 값:", borda_scores[np.argsort(borda_scores)[-10:][::-1]])

    print()

def copeland_rule(ratings, grouped_users):
    pass

# 모든 그룹에 대해 상위 10개 영화 출력
additive_utilitarian(ratings, clustered_users)
average(ratings, clustered_users)
simple_count(ratings, clustered_users)
approval_voting(ratings, clustered_users)
borda_count(ratings, clustered_users)
copeland_rule(ratings, clustered_users)