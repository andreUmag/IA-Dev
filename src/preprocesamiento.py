import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from collections import Counter

nltk.download('stopwords')

custom_stopwords = {'movie', 'film', 'one', 'character', 'story', 'make', 'even'}
stop_words = set(stopwords.words('english')).union(custom_stopwords)

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

def explore_data(df):
    print("Primeras filas:\n", df.head())
    print("\nDescripción:\n", df.describe(include='all'))
    print("\nValores nulos:\n", df.isnull().sum())

    plt.figure(figsize=(10, 6))
    sns.histplot(df['Review_clean'].apply(lambda x: len(x.split())), bins=30)
    plt.title('Distribución de longitud de reseñas')
    plt.xlabel('Número de palabras')
    plt.ylabel('Frecuencia')
    plt.show()

    corr = df[['label']].copy()
    corr['word_count'] = df['Review_clean'].apply(lambda x: len(x.split()))
    sns.heatmap(corr.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlación entre variables')
    plt.show()

def vectorize_and_split(df, max_features=2000, test_size=0.2, random_state=42):
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

def scale_features(train_df, test_df):
    features = [col for col in train_df.columns if col != 'label']
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    test_df[features] = scaler.transform(test_df[features])
    return train_df, test_df

def save_parquet(train_df, test_df, train_path, test_path):
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

def plot_category_counts(df):
    df['label_str'] = df['label'].map({0: 'Rotten', 1: 'Fresh'})

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='label_str', hue='label_str', palette={'Rotten': '#DF776E', 'Fresh': '#28DEB8'}, legend=False)
    plt.title('Número de reviews por categoría')
    plt.xlabel('Categoría')
    plt.ylabel('Cantidad')
    plt.show()

def plot_word_counts(df):
    df['word_count'] = df['Review_clean'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=['Rotten', 'Fresh'],
        y=[df[df['label'] == 0]['word_count'].sum(),
           df[df['label'] == 1]['word_count'].sum()],
        palette=['#DF776E', '#28DEB8']
    )
    plt.title('Total de palabras por categoría')
    plt.ylabel('Cantidad de palabras')
    plt.show()

def plot_wordclouds(df):
    fresh_text = ' '.join(df[df['label'] == 1]['Review_clean'])
    rotten_text = ' '.join(df[df['label'] == 0]['Review_clean'])

    wordcloud_fresh = WordCloud(width=600, height=400, background_color='white').generate(fresh_text)
    wordcloud_rotten = WordCloud(width=600, height=400, background_color='white').generate(rotten_text)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(wordcloud_fresh, interpolation='bilinear')
    plt.title('WordCloud - Fresh')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(wordcloud_rotten, interpolation='bilinear')
    plt.title('WordCloud - Rotten')
    plt.axis('off')

    plt.show()

def export_frequent_words(df):
    def get_word_counts(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)

    fresh_text = ' '.join(df[df['label'] == 1]['Review_clean'])
    rotten_text = ' '.join(df[df['label'] == 0]['Review_clean'])

    fresh_counts = get_word_counts(fresh_text)
    rotten_counts = get_word_counts(rotten_text)

    fresh_df = pd.DataFrame(fresh_counts.items(), columns=['Word', 'Count'])
    fresh_df['Label'] = 'Fresh'
    fresh_df = fresh_df.sort_values(by='Count', ascending=False)

    rotten_df = pd.DataFrame(rotten_counts.items(), columns=['Word', 'Count'])
    rotten_df['Label'] = 'Rotten'
    rotten_df = rotten_df.sort_values(by='Count', ascending=False)

    fresh_df.to_csv('src/data/frequent_words_fresh.csv', index=False)
    rotten_df.to_csv('src/data/frequent_words_rotten.csv', index=False)

def plot_unique_words_and_export_common(df):
    def get_word_counts(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)

    fresh_text = ' '.join(df[df['label'] == 1]['Review_clean'])
    rotten_text = ' '.join(df[df['label'] == 0]['Review_clean'])

    fresh_counts = get_word_counts(fresh_text)
    rotten_counts = get_word_counts(rotten_text)

    fresh_words = set(fresh_counts.keys())
    rotten_words = set(rotten_counts.keys())

    unique_fresh = fresh_words - rotten_words
    unique_rotten = rotten_words - fresh_words

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=['Fresh Only', 'Rotten Only'],
        y=[len(unique_fresh), len(unique_rotten)],
        palette=['#28DEB8', '#DF776E']
    )
    plt.title('Número de palabras únicas por categoría')
    plt.ylabel('Cantidad de palabras únicas')
    plt.show()

    common_words = fresh_words & rotten_words
    data = []
    for word in common_words:
        data.append({
            'Word': word,
            'Fresh Count': fresh_counts[word],
            'Rotten Count': rotten_counts[word]
        })

    common_df = pd.DataFrame(data)
    common_df = common_df.sort_values(by=['Fresh Count', 'Rotten Count'], ascending=False)
    common_df.to_csv('src/data/common_words_counts.csv', index=False)

if __name__ == "__main__":
    filepath = "src/data/rt_reviews.csv"
    df = load_and_preprocess(filepath)

    explore_data(df)

    train_df, test_df, vectorizer = vectorize_and_split(df)
    train_df, test_df = scale_features(train_df, test_df)

    train_path = "src/data/train_data_clean.parquet"
    test_path = "src/data/test_data_clean.parquet"
    save_parquet(train_df, test_df, train_path, test_path)

    plot_category_counts(df)
    plot_word_counts(df)
    plot_wordclouds(df)

    export_frequent_words(df)
    plot_unique_words_and_export_common(df)
