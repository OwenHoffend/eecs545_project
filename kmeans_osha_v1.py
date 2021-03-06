# -*- coding: utf-8 -*-
"""Kmeans_OSHA_v1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1C3nRztmiP3zYasw6nNt30y-4MbgRvQXe

# Apply K-means to OSHA
"""

#Install packages 
#!pip install pyldavis

data_path = "your_path"
data_file_lst = os.listdir(data_path)

print("data_file_lst: {}".format(data_file_lst))

import pandas as pd
df = pd.read_excel(data_path+'osha 4470_with_additional_metadata.xlsx')
print(len(df))

df.head()

df.dropna(subset=['title', 'SUMMARY'])

print(len(df))

df.head(5)

title = df['title']
summary = df['SUMMARY']

title.head()

summary.head()

"""## Preprocessing: Remove punctuation, lower alphabet

### About Title
"""

import re
title = title.apply(lambda x: x.strip())
# Remove punctuation
title = title.apply(lambda x: re.sub('[.,\!]', '', str(x)))
# Lower the letter
title = title.apply(lambda x: x.lower())
title.head()

"""### About Summary"""

import re
summary = summary.apply(lambda x: x.strip())
# Remove punctuation
summary = summary.apply(lambda x: re.sub('[.,\!]', '', str(x)))
# Lower the letter
summary = summary.apply(lambda x: x.lower())
summary.head()

"""## TF-IDF vectorization

### About Title
"""

import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

import gensim
from gensim.utils import simple_preprocess

stop_words = stopwords.words('english')
extend_lst = ['from', '']

stop_words.extend(extend_lst)
print(stop_words)

title_tolist = title.tolist()
title_tolist

prep_sent_list = []

## Remove Stopwords
for i in range(len(title_tolist)):
  sent = title_tolist[i]
  empty_lst = []
  for word in sent.split(" "): 
    if word not in  stop_words:
      empty_lst.append(word)
  input_str = ' '.join(empty_lst)
  prep_sent_list.append(input_str)

prep_sent_list

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(prep_sent_list)

tf_idf = pd.DataFrame(data = X.toarray(), columns=vectorizer.get_feature_names())

final_df = tf_idf

print("{} rows".format(final_df.shape[0]))
final_df.T.nlargest(5, 0)

from sklearn import cluster
def run_KMeans(max_k, data):
    max_k += 1
    kmeans_results = dict()
    for k in range(2 , max_k):
        kmeans = cluster.KMeans(n_clusters = k
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               #, n_jobs = -1
                               , random_state = 1
                               , algorithm = 'full')

        kmeans_results.update( {k : kmeans.fit(data)} )
        
    return kmeans_results

kmeans = cluster.KMeans(n_clusters = 3
                               , init = 'k-means++'
                               , n_init = 10
                               , tol = 0.0001
                               #, n_jobs = -1
                               , random_state = 1
                               , algorithm = 'full').fit(final_df)
labels = kmeans.labels_

new_dfa = pd.DataFrame(data = final_df)
new_dfa['label_kmeans'] = labels
new_dfa

distortions = []
for i in range(2, 8):
    km = cluster.KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    km.fit(final_df)
    distortions.append(km.inertia_)

# plot
plt.plot(range(2, 8), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

"""### Silhouette score"""

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm

def printAvg(avg_dict):
    for avg in sorted(avg_dict.keys(), reverse=True):
        print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))
        
def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(8, 6)
    ax1.set_xlim([-0.2, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--") # The vertical line for average silhouette score of all the values
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')
    
    y_lower = 10
    sample_silhouette_values = silhouette_samples(df, kmeans_labels) # Compute the silhouette scores for each sample
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i)) # Label the silhouette plots with their cluster numbers at the middle
        y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
    plt.show()
    
        
def silhouette(kmeans_dict, df, plot=False):
    df = df.to_numpy()
    avg_dict = dict()
    for n_clusters, kmeans in kmeans_dict.items():      
        kmeans_labels = kmeans.predict(df)
        silhouette_avg = silhouette_score(df, kmeans_labels) # Average Score for all Samples
        avg_dict.update( {silhouette_avg : n_clusters} )
    
        if(plot): plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg)

# Running Kmeans
k = 8
kmeans_results = run_KMeans(k, final_df)

df = final_df.to_numpy()
avg_dict = dict()
for n_clusters, kmeans in kmeans_results.items():      
    kmeans_labels = kmeans.predict(df)
    silhouette_avg = silhouette_score(df, kmeans_labels) # Average Score for all Samples
    avg_dict.update( {silhouette_avg : n_clusters} )
avg_dict

"""###Cluster analysis"""

def get_top_features_cluster(tf_idf_array, prediction, n_feats):
    labels = np.unique(prediction)
    dfs = []
    for label in labels:
        id_temp = np.where(prediction==label) # indices for each cluster
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores
        features = vectorizer.get_feature_names()
        best_features = [(features[i], x_means[i]) for i in sorted_means]
        df = pd.DataFrame(best_features, columns = ['features', 'score'])
        dfs.append(df)
    return dfs

def plotWords(dfs, n_feats):
    plt.figure(figsize=(8, 4))
    for i in range(0, len(dfs)):
        plt.title(("Most Common Words in Cluster {}".format(i)), fontsize=10, fontweight='bold')
        sns.barplot(x = 'score' , y = 'features', orient = 'h' , data = dfs[i][:n_feats])
        plt.show()

best_result = 5
kmeans = kmeans_results.get(best_result)
import numpy as np
import matplotlib.pyplot  as plt
import seaborn as sns
final_df_array = final_df.to_numpy()
prediction = kmeans.predict(final_df)
n_feats = 20
dfs = get_top_features_cluster(final_df_array, prediction, n_feats)
plotWords(dfs, 13)

"""### Map of words

"""

# Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
def centroidsDict(centroids, index):
    a = centroids.T[index].sort_values(ascending = False).reset_index().values
    centroid_dict = dict()

    for i in range(0, len(a)):
        centroid_dict.update( {a[i,0] : a[i,1]} )

    return centroid_dict

def generateWordClouds(centroids):
    wordcloud = WordCloud(max_font_size=100, background_color = 'white')
    for i in range(0, len(centroids)):
        centroid_dict = centroidsDict(centroids, i)        
        wordcloud.generate_from_frequencies(centroid_dict)

        plt.figure()
        plt.title('Cluster {}'.format(i))
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()

centroids = pd.DataFrame(kmeans.cluster_centers_)
centroids

from wordcloud import WordCloud
centroids.columns = final_df.columns
generateWordClouds(centroids)













