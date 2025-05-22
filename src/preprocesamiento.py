import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath, encoding='latin1')
    df['label'] = df['Freshness'].map({'fresh': 1, 'rotten': 0})
    df = df.dropna(subset=['Review'])

    def preprocess(text):
        text = text.lower()
        text = re.sub(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        return ' '.join(tokens)

    df['Review_clean'] = df['Review'].apply(preprocess)
    return df


def vectorize_and_split(df, max_features=5000, test_size=0.2, random_state=42):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df['Review_clean'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    train_df['label'] = y_train.values
    test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())
    test_df['label'] = y_test.values

    return train_df, test_df, vectorizer


def save_csv(train_df, test_df, train_path, test_path):
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def plot_category_counts(df):
    plt.figure(figsize=(6,4))
    sns.countplot(x='label', data=df)
    plt.xticks(ticks=[0,1], labels=['Rotten', 'Fresh'])
    plt.title('Número de reviews por categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad')
    plt.show()


def plot_word_counts(df):
    df['word_count'] = df['Review_clean'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6,4))
    sns.barplot(x=['Rotten', 'Fresh'],
                y=[df[df['label'] == 0]['word_count'].sum(),
                   df[df['label'] == 1]['word_count'].sum()])
    plt.title('Total de palabras por categoría')
    plt.ylabel('Cantidad de palabras')
    plt.show()


def plot_wordclouds(df):
    fresh_text = ' '.join(df[df['label'] == 1]['Review_clean'])
    rotten_text = ' '.join(df[df['label'] == 0]['Review_clean'])

    wordcloud_fresh = WordCloud(width=600, height=400, background_color='white').generate(fresh_text)
    wordcloud_rotten = WordCloud(width=600, height=400, background_color='white').generate(rotten_text)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(wordcloud_fresh, interpolation='bilinear')
    plt.title('WordCloud - Fresh')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(wordcloud_rotten, interpolation='bilinear')
    plt.title('WordCloud - Rotten')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    
    
    filepath = "src/data/rt_reviews.csv"

    df = load_and_preprocess(filepath)

    train_df, test_df, vectorizer = vectorize_and_split(df)

    train_path = "src/data/train_data_clean.csv"
    test_path = "src/data/test_data_clean.csv"
    save_csv(train_df, test_df, train_path, test_path)

    #Graficas
    plot_category_counts(df)
    plot_word_counts(df)
    plot_wordclouds(df)
