import pickle
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from text_utils import remove_tags, rem_body_space, freq_words, remove_stopwords


class TagExtractionTrain:
    def __init__(self, csv_data):
        self.data = csv_data
        self.data_new = None
        self.new_tags = []
        self.all_tags = None
        self.all_tags_df = None
        self.all_wordas = []
        #nltk.download('stopwords')

    def pre_process_data(self):
        #print(self.data.shape)
        self.data.drop_duplicates(keep=False)
        #print(self.data.shape)
        self.data = self.data.dropna()
        for i in self.data['Tags']:
            self.new_tags.append(i.split())
        self.data['New_Tags'] = self.new_tags
        self.data_new = self.data[~(self.data['New_Tags'].str.len() == 0)]
        #print(self.data.shape, self.data_new.shape)
        #print(self.data_new.head())
        self.all_tags = sum(self.new_tags, [])
        print('Total Tages: ' + str(len(set(self.all_tags))))
        self.all_tags = nltk.FreqDist(self.all_tags)
        self.all_tags_df = pd.DataFrame({'Tags': list(self.all_tags.keys()), 'Count': list(self.all_tags.values())})
        g = self.all_tags_df.nlargest(columns="Count", n=20)
        plt.figure(figsize=(20, len(g)+20))
        ax = sns.barplot(data=g, x="Count", y="Tags")
        ax.set(ylabel='Count')
        plt.show()
        self.data_new['New_Body'] = self.data_new['Body'].apply(lambda x: remove_tags(x))
        self.data_new['New_Body'] = self.data_new['Body'].apply(lambda x: rem_body_space(x))
        freq_words(self.data_new['New_Body'], 100)
        self.data_new['New_Body'] = self.data_new['New_Body'].apply(lambda x: remove_stopwords(x))
        freq_words(self.data_new['New_Body'], 100)
        print('='*30)
        print('Question 1 Orginal')
        print('='*30)
        print(self.data_new['Body'][0])
        print('='*30)
        print('Question 1 Pre-Processed')
        print('='*30)
        print(self.data_new['New_Body'][0])
        self.text_to_features()

    def text_to_features(self):
        multilabel_binarizer = MultiLabelBinarizer()
        multilabel_binarizer.fit(self.data_new['New_Tags'])

        y_train = multilabel_binarizer.transform(self.data_new['New_Tags'])
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
        x_train = self.data_new['New_Body']
        x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
        log_reg = RandomForestClassifier(n_estimators=10)
        one_v_rest = OneVsRestClassifier(log_reg)
        one_v_rest.fit(x_train_tfidf, y_train)
        pickle.dump(one_v_rest, open('classifier.pickle', 'wb'))
        y_pred = one_v_rest.predict(x_train_tfidf)
        print('\n')
        print('='*30)
        print('Predicted_Tags (From 1 To 30)')
        print('='*30)
        for i in multilabel_binarizer.inverse_transform(y_pred):
            print(i)
        print('\n')
        print('='*30)
        print('F1_Score: ', str(f1_score(y_train, y_pred, average='micro')))
        print('='*30)


if __name__ == '__main__':
    csv_file = 'dataset/Train_Sample.csv'
    dataa = pd.read_csv(csv_file)
    txt = TagExtractionTrain(csv_data=dataa)
    txt.pre_process_data()
