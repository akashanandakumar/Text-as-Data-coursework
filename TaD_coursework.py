#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# Load the dataset
my_df = pd.read_csv("Reviews.csv")

# Display the first 5 rows of the DataFrame
print(my_df.head())


# In[3]:


my_df.shape


# In[4]:


from sklearn.model_selection import train_test_split

# map the ratings to labels
my_df['reviews'] = my_df['Score'].apply(lambda x: 'Negative' if x in [1, 2] else 'Neutral' if x == 3 else 'Positive')

# split the dataset into training, validation, and test sets
train_val_my_df, test_my_df = train_test_split(my_df, test_size=0.2, random_state=42)
train_my_df, val_my_df = train_test_split(train_val_my_df, test_size=0.25, random_state=42)

# calculate label counts for each split
train_label_counts = train_my_df['reviews'].value_counts()
val_label_counts = val_my_df['reviews'].value_counts()
test_label_counts = test_my_df['reviews'].value_counts()

# create table of label counts for each split
counts_table = pd.concat([train_label_counts, val_label_counts, test_label_counts], axis=1, sort=False)
counts_table.columns = ['Train', 'Validation', 'Test']
counts_table.index.name = 'Label'
print(counts_table) 

# Check label distribution across splits
print('Training Set Label Distribution:', train_label_counts/len(train_my_df))
print('Validation Set Label Distribution:', val_label_counts/len(val_my_df))
print('Test Set Label Distribution:', test_label_counts/len(test_my_df))


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Step 0: Vectorise text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(my_df['Text'])

# Step 1: Pick k random "centroids"
k = 5
my_kmeans = KMeans(n_clusters=k, init='random', max_iter=100, n_init=1, random_state=42)

# Step 2 and 3: Assign each vector to its closest centroid and recalculate the centroids based on the closest vectors
my_kmeans.fit(X)



def get_top_terms(cluster_centroid, feature_names_out, n=10):
    # Get the top n terms for a given centroid
    sort_terms = cluster_centroid.argsort()[::-1]
    return [feature_names_out[i] for i in sort_terms[:n]]



# Print the cluster assignments and the top terms in each cluster
cluster_labels = my_kmeans.labels_
cluster_centroids = my_kmeans.cluster_centers_



for i in range(k):
    print("Cluster %d:" % i)
    cluster_documents = [doc for j, doc in enumerate(my_df) if cluster_labels[j] == i]
    print("Number of documents:", len(cluster_documents))
    top_term_cluster = get_top_terms(cluster_centroids[i], vectorizer.get_feature_names_out(), n=10)
    print("Top terms:", top_term_cluster)


# In[6]:


for i in range(k):
    print("Clusters %d:" % i)
    cluster_documents = [doc for j, doc in enumerate(my_df) if cluster_labels[j] == i]
    print("Number of docs:", len(cluster_documents))
    print("Sample docs:")
    for doc in cluster_documents[:5]:
        print(" -", doc)
    top_term_cluster = get_top_terms(cluster_centroids[i], vectorizer.get_feature_names_out(), n=5)
    print("Top terms:", top_term_cluster)


# In[7]:


import pandas as pd

# Create a new DataFrame with cluster assignments and corresponding true labels
my_cluster_df = pd.DataFrame({'Cluster': cluster_labels, 'True_Label': my_df['reviews']})

# Construct the confusion matrix using pd.crosstab()
my_confusion_matrix = pd.crosstab(my_cluster_df['Cluster'], my_cluster_df['True_Label'])

# Print the confusion matrix
print(my_confusion_matrix)


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split the data into training and test sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(my_df['Text'], my_df['reviews'], test_size=0.2, random_state=42)

# Baseline 1: Dummy Classifier with strategy="most_frequent"
dummy_most_freq = DummyClassifier(strategy="most_frequent")
dummy_most_freq.fit(X_train_set, y_train_set)
y_pred_dummy_most_freq = dummy_most_freq.predict(X_test_set)

# Evaluate the classifier using accuracy, precision, recall, and F1-score
print("Baseline 1: Dummy Classifier with strategy=\"most_frequent\"")
print("Accuracy:", accuracy_score(y_test_set, y_pred_dummy_most_freq))
print("Precision:", precision_score(y_test_set, y_pred_dummy_most_freq, average='macro', zero_division=0))
print("Recall:", recall_score(y_test_set, y_pred_dummy_most_freq, average='macro', zero_division=0))
print("F1-score:", f1_score(y_test_set, y_pred_dummy_most_freq, average='macro', zero_division=0))


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Split the data into training and test sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(my_df['Text'], my_df['reviews'], test_size=0.2, random_state=42)

# Baseline 2: Dummy Classifier with strategy="stratified"
dummy_strat = DummyClassifier(strategy="stratified")
dummy_strat.fit(X_train_set, y_train_set)
y_pred_dummy_strat = dummy_strat.predict(X_test_set)

# Evaluate the classifier using accuracy, precision, recall, and F1-score
print("Baseline 2: Dummy Classifier with strategy=\"stratified\"")
print("Accuracy:", accuracy_score(y_test_set, y_pred_dummy_strat))
print("Precision:", precision_score(y_test_set, y_pred_dummy_strat, average='macro'))
print("Recall:", recall_score(y_test_set, y_pred_dummy_strat, average='macro'))
print("F1-score:", f1_score(y_test_set, y_pred_dummy_strat, average='macro'))


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert the 'Text' column to strings
my_df['Text'] = my_df['Text'].astype(str)

# Split the data into training and test sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(my_df['Text'], my_df['reviews'], test_size=0.2, random_state=42)

# Transform the text data into a one-hot encoded matrix
vectorizer = CountVectorizer(binary=True)
X_train_onehot = vectorizer.fit_transform(X_train_set)
X_test_onehot = vectorizer.transform(X_test_set)

# Baseline 3: Logistic Regression with One-hot vectorization
logreg_onehot = LogisticRegression(max_iter=2000)
logreg_onehot.fit(X_train_onehot, y_train_set)
y_pred_logreg_onehot = logreg_onehot.predict(X_test_onehot)

# Evaluate the classifier using accuracy, precision, recall, and F1-score
print("Baseline 3: Logistic Regression with One-hot vectorization")
print("Accuracy:", accuracy_score(y_test_set, y_pred_logreg_onehot))
print("Precision:", precision_score(y_test_set, y_pred_logreg_onehot, average='macro'))
print("Recall:", recall_score(y_test_set, y_pred_logreg_onehot, average='macro'))
print("F1-score:", f1_score(y_test_set, y_pred_logreg_onehot, average='macro'))


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert the 'Text' column to strings
my_df['Text'] = my_df['Text'].astype(str)

# Split the data into training and test sets
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(my_df['Text'], my_df['reviews'], test_size=0.2, random_state=42)

# Transform the text data into a TF-IDF matrix
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train_set)
X_test_tfidf = vectorizer.transform(X_test_set)

# Baseline 4: Logistic Regression with TF-IDF vectorization
logreg_tfidf = LogisticRegression(max_iter=1000)
logreg_tfidf.fit(X_train_tfidf, y_train_set)
y_pred_logreg_tfidf = logreg_tfidf.predict(X_test_tfidf)

# Evaluate the classifier using accuracy, precision, recall, and F1-score
print("Baseline 4: Logistic Regression with TF-IDF vectorization")
print("Accuracy:", accuracy_score(y_test_set, y_pred_logreg_tfidf))
print("Precision:", precision_score(y_test_set, y_pred_logreg_tfidf, average='macro'))
print("Recall:", recall_score(y_test_set, y_pred_logreg_tfidf, average='macro'))
print("F1-score:", f1_score(y_test_set, y_pred_logreg_tfidf, average='macro'))



