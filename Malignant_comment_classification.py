#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import  stopwords

import warnings
warnings.filterwarnings('ignore')


# In[2]:


train=pd.read_csv('malignant_comment_train.csv')
test=pd.read_csv('malignant_comment_test.csv')


# In[4]:


train


# In[5]:


test


# # EDA

# In[8]:


train.info()


# In[41]:


train.iloc[:,2:].sum()


# In[12]:


for columns in train.iloc[:,2:]:
    print(columns)


# In[44]:



# Plot a bar chart using the index (category values) and the count of each category

plt.figure(figsize=(8,4))
ax = sns.barplot(train.iloc[:,2:].sum().index, train.iloc[:,2:].sum().values, alpha=0.8)
plt.title("No. of comments per class")
plt.ylabel('No. of Occurrences', fontsize=12)
plt.xlabel('Type ', fontsize=12)
rects = ax.patches
labels = train.iloc[:,2:].sum().values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.show()


# In[47]:



# Create a bar graph 
sum_mal = train['malignant'].sum() / len(train) * 100
sum_hig = train['highly_malignant'].sum() / len(train) * 100
sum_rude = train['rude'].sum() / len(train) * 100
sum_thr = train['threat'].sum() / len(train) * 100
sum_abu = train['abuse'].sum() / len(train) * 100
sum_loa = train['loathe'].sum() / len(train) * 100

# Initiate a list of 6 values that represent the 6 x-axis values for the categories
ind = np.arange(6)

# Let the ind variable be the x-axis, whereas the % of toxicity for each category be the y-axis.
# Sequence of % have been sorted manually. This method cannot be done if there are large numbers of categories.
ax = plt.barh(ind, [sum_mal, sum_hig, sum_rude, sum_thr, sum_abu, sum_loa])
plt.xlabel('Percentage (%)', size=20)
plt.xticks(np.arange(0, 30, 5), size=20)
plt.title('% of comments in various categories', size=20)
plt.yticks(ind, ('malignant', 'highly_malignant', 'rude', 'threat','abuse', 'loathe' ), size=15)

# Invert the graph so that it is in descending order.
plt.gca().invert_yaxis()
plt.show()


# In[19]:


for columns in train.iloc[:,2:]:
    sns.countplot(train[columns])
    plt.show()
    print(train[columns].value_counts())


# In[24]:


# remove all numbers with letters attached to them
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)

# '[%s]' % re.escape(string.punctuation),' ' - replace punctuation with white space
# .lower() - convert all strings to lowercase 
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())

# Remove all '\n' in the string and replace it with a space
remove_n = lambda x: re.sub("\n", " ", x)

# Remove all non-ascii characters 
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]',r' ', x)

# Apply all the lambda functions wrote previously through .map on the comments column
train['comment_text'] = train['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)


# # Separating the labels

# In[26]:


data_mal = train.loc[:,['id','comment_text','malignant']]
data_hig = train.loc[:,['id','comment_text','highly_malignant']]
data_rude = train.loc[:,['id','comment_text','rude']]
data_thr = train.loc[:,['id','comment_text','threat']]
data_abu = train.loc[:,['id','comment_text','abuse']]
data_loa = train.loc[:,['id','comment_text','loathe']]


# # Wordcloud for visualising most used words

# In[27]:


import wordcloud
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords


# In[28]:


def wordcloud(df, label):
    
    # Print only rows where the toxic category label value is 1 (ie. the comment is toxic)
    subset=df[df[label]==1]
    text=subset.comment_text.values
    wc= WordCloud(background_color="black",max_words=4000)
    wc.generate(" ".join(text))

    plt.figure(figsize=(25,25))
    plt.subplot(221)
    plt.axis("off")
    plt.title("Words frequented in {}".format(label), fontsize=20)
    plt.imshow(wc.recolor(random_state=244), alpha=0.98)


# In[30]:


wordcloud(data_mal,'malignant')


# In[31]:


wordcloud(data_hig,'highly_malignant')


# In[32]:


wordcloud(data_rude,'rude')


# In[33]:


wordcloud(data_thr,'threat')


# In[34]:


wordcloud(data_abu, 'abuse')


# In[29]:


wordcloud(data_loa,'loathe')


# # Making the unbalanced data a balanced One

# In[35]:


data_mal_1 = data_mal[data_mal['malignant'] == 1].iloc[0:5000,:]
data_mal_1.shape


# In[36]:


data_mal_0 = data_mal[data_mal['malignant'] == 0].iloc[0:5000,:]


# In[37]:


data_mal_done = pd.concat([data_mal_1, data_mal_0], axis=0)
data_mal_done.shape


# In[38]:


data_hig[data_hig['highly_malignant'] == 1].count()


# In[48]:


data_hig_1 = data_hig[data_hig['highly_malignant'] == 1].iloc[0:1595,:]
data_hig_0 = data_hig[data_hig['highly_malignant'] == 0].iloc[0:1595,:]
data_hig_done = pd.concat([data_hig_1, data_hig_0], axis=0)
data_hig_done.shape


# In[49]:


data_rude[data_rude['rude'] == 1].count()


# In[50]:


data_rude_1 = data_rude[data_rude['rude'] == 1].iloc[0:5000,:]
data_rude_0 = data_rude[data_rude['rude'] == 0].iloc[0:5000,:]
data_rude_done = pd.concat([data_rude_1, data_rude_0], axis=0)
data_rude_done.shape


# In[51]:


data_thr[data_thr['threat'] == 1].count()


# In[52]:


data_thr_1 = data_thr[data_thr['threat'] == 1].iloc[0:478,:]

# We include 1912 comments that have no threat so that the data with threat (478) will represent 20% of the dataset.
data_thr_0 = data_thr[data_thr['threat'] == 0].iloc[0:1912,:]  
data_thr_done = pd.concat([data_thr_1, data_thr_0], axis=0)
data_thr_done.shape


# In[53]:


data_abu[data_abu['abuse'] == 1].count()


# In[54]:


data_abu_1 = data_abu[data_abu['abuse'] == 1].iloc[0:5000,:]
data_abu_0 = data_abu[data_abu['abuse'] == 0].iloc[0:5000,:]
data_abu_done = pd.concat([data_abu_1, data_abu_0], axis=0)
data_abu_done.shape


# In[67]:


data_loa_1 = data_loa[data_loa['loathe'] == 1].iloc[0:1405,:] # 20%
data_loa_0 = data_loa[data_loa['loathe'] == 0].iloc[0:5620,:] # 80%
data_loa_done = pd.concat([data_loa_1, data_loa_0], axis=0)
data_loa_done.shape


# # Model Building

# In[58]:


# Import packages for pre-processing
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel

# Import tools to split data and evaluate model performance
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, fbeta_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Import ML algos
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


# In[59]:


def cv_tf_train_test(df_done,label,vectorizer,ngram):

    ''' Train/Test split'''
    # Split the data into X and y data sets
    X = df_done.comment_text
    y = df_done[label]

    # Split our data into training and test data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    ''' Count Vectorizer/TF-IDF '''

    # Create a Vectorizer object and remove stopwords from the table
    cv1 = vectorizer(ngram_range=(ngram), stop_words='english')
    
    X_train_cv1 = cv1.fit_transform(X_train) # Learn the vocabulary dictionary and return term-document matrix
    X_test_cv1  = cv1.transform(X_test)      # Learn a vocabulary dictionary of all tokens in the raw documents.
    
    # Output a Dataframe of the CountVectorizer with unique words as the labels
    # test = pd.DataFrame(X_train_cv1.toarray(), columns=cv1.get_feature_names())
        
    ''' Initialize all model objects and fit the models on the training data '''
    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)
    print('lr done')

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_cv1, y_train)

    bnb = BernoulliNB()
    bnb.fit(X_train_cv1, y_train)
    print('bnb done')
    
    mnb = MultinomialNB()
    mnb.fit(X_train_cv1, y_train)
    print('mnb done')
    
    svm_model = LinearSVC()
    svm_model.fit(X_train_cv1, y_train)

    randomforest = RandomForestClassifier(n_estimators=100, random_state=42)
    randomforest.fit(X_train_cv1, y_train)
    print('rdf done')
    
    # Create a list of F1 score of all models 
    f1_score_data = {'F1 Score':[f1_score(lr.predict(X_test_cv1), y_test), f1_score(knn.predict(X_test_cv1), y_test), 
                                f1_score(bnb.predict(X_test_cv1), y_test), f1_score(mnb.predict(X_test_cv1), y_test),
                                f1_score(svm_model.predict(X_test_cv1), y_test), f1_score(randomforest.predict(X_test_cv1), y_test)]} 
                          
    # Create DataFrame with the model names as column labels
    df_f1 = pd.DataFrame(f1_score_data, index=['Log Regression','KNN', 'BernoulliNB', 'MultinomialNB', 'SVM', 'Random Forest'])  

    return df_f1


# In[60]:


df_mal_cv = cv_tf_train_test(data_mal_done, 'malignant', TfidfVectorizer, (1,1))
df_mal_cv.rename(columns={'F1 Score': 'F1 Score(malignant)'}, inplace=True)

df_mal_cv


# In[61]:


df_hig_cv = cv_tf_train_test(data_hig_done, 'highly_malignant', TfidfVectorizer, (1,1))
df_hig_cv.rename(columns={'F1 Score': 'F1 Score(highly_malignant)'}, inplace=True)

df_hig_cv


# In[62]:


df_rude_cv = cv_tf_train_test(data_rude_done, 'rude', TfidfVectorizer, (1,1))
df_rude_cv.rename(columns={'F1 Score': 'F1 Score(rude)'}, inplace=True)

df_rude_cv


# In[63]:


df_thr_cv = cv_tf_train_test(data_thr_done, 'threat', TfidfVectorizer, (1,1))
df_thr_cv.rename(columns={'F1 Score': 'F1 Score(threat)'}, inplace=True)

df_thr_cv


# In[65]:


df_abu_cv = cv_tf_train_test(data_abu_done, 'abuse', TfidfVectorizer, (1,1))
df_abu_cv.rename(columns={'F1 Score': 'F1 Score(abuse)'}, inplace=True)

df_abu_cv


# In[68]:


df_loa_cv = cv_tf_train_test(data_loa_done, 'loathe', TfidfVectorizer, (1,1))
df_loa_cv.rename(columns={'F1 Score': 'F1 Score(loathe)'}, inplace=True)

df_loa_cv


# In[69]:


# Let's combine the dataframes into a master dataframe to compare F1 scores across all categories.
final_all = pd.concat([df_mal_cv, df_hig_cv, df_rude_cv, df_abu_cv, df_thr_cv, df_loa_cv], axis=1)
final_all


# In[71]:


final_all_trp = final_all.transpose()
final_all_trp


# In[74]:


sns.lineplot(data=final_all_trp, markers=True)
plt.xticks(rotation='90', fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best')
plt.title('F1 Score of ML models (TF-IDF)', fontsize=20)

# Repeat this for CountVectorizer as well


# LinearSVM and Random Forest models perform best (purple and brown lines seem to be the highest).
# 
# Test if our code actually works. Probability of the comment falling in various categories should be output.

# In[75]:


data_mal_done.head()


# In[76]:


X = data_mal_done.comment_text
y = data_mal_done['malignant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initiate a Tfidf vectorizer
tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfv.fit_transform(X_train)  # Convert the X data into a document term matrix dataframe
X_test_fit = tfv.transform(X_test)  # Converts the X_test comments into Vectorized format

randomforest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train our SVM model with the X training data converted into Count Vectorized format with the Y training data
randomforest.fit(X_train_fit, y_train)
randomforest.predict(X_test_fit)


# In[78]:


test.head()


# In[79]:


# Sample Prediction
test_vect = tfv.transform(test['comment_text'])
randomforest.predict_proba(test_vect)[:,1]


# LinearSVM and Random Forest models perform best (purple and brown lines seem to be the highest)

# In[ ]:




