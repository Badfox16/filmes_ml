# Importando as bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, naive_bayes, metrics
import pickle

# Carregando o dataset
# Certifique-se de que o caminho para o arquivo CSV está correto
df = pd.read_csv('./Datasets/IMDB_Dataset.csv')

# Visualizando as primeiras linhas do dataset
df.head()

# Verificando informações básicas sobre o dataset
df.info()

# Preprocessamento dos dados
# Removendo valores nulos
df.dropna(inplace=True)

# Convertendo os textos para letras minúsculas
df['Comments'] = df['Comments'].str.lower()

# Removendo caracteres especiais
df['Comments'] = df['Comments'].str.replace('[^a-zA-Z]', ' ')

# Verificando a distribuição das classes
df['Reviews'].value_counts()

# Dividindo o dataset em treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(df['Comments'], df['Reviews'], test_size=0.2, random_state=42)

# Convertendo os textos em vetores numéricos usando TF-IDF
tfidf_vect = preprocessing.text.TfidfVectorizer(max_features=5000)
tfidf_vect.fit(X_train)

X_train_tfidf = tfidf_vect.transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# Treinando o modelo Naive Bayes
clf = naive_bayes.MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Avaliando a acurácia do modelo
accuracy = accuracy_score(y_test, clf.predict(X_test_tfidf)) * 100
print(f'Acurácia do modelo: {accuracy:.2f}%')

# Salvando o modelo treinado em um arquivo
pickle.dump(clf, open('nlp_model.pkl', 'wb'))
