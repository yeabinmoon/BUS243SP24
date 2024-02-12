# Put some inputs here.

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('PATH+boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv')
# df = pd.read_csv('PATH+boydstun_nyt_frontpage_dataset_1996-2006_0_pap2014_recoding_updated2018.csv', encoding='latin-1')

train_set, temp = train_test_split(df, test_size=0.4, random_state=1231)
dev_set, test_set = train_test_split(temp, test_size=0.5, random_state=123)

# Consider what this line:
train_set.loc[train_set['title'].isnull(),'title'] = train_set.loc[train_set['title'].isnull(),'summary']

X_train, X_dev, X_test = train_set['title'], dev_set['title'], test_set['title']
y_train, y_dev, y_test = train_set['majortopic'], dev_set['majortopic'], test_set['majortopic']

## Exercise
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

count = TfidfVectorizer(tokenizer = text: TreebankWordTokenizer().tokenize(text), max_features=10**2)
%time bag_of_words = count.fit_transform(X_train)
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(bag_of_words,y_train)
print(log_reg.score(bag_of_words, y_train))
dev_bow = count.transform(X_dev)
print(log_reg.score(dev_bow, y_dev))

