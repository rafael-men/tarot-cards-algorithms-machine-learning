import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# utilização de naipes como classes, pois apenas 'minor' e 'major' não são suficientes para classificar as cartas

dataset = 'dataset/cards.json'
nome_col = 'name'
categorias = [ 'planet', 'zodiac', 'element', 'yes_no', 'yes_no_reversed' ]
numericas = [ 'number_numerology' ]
texto_col = [
    "keywords_upright",
    "keywords_reversed",
    "meaning_upright",
    "meaning_reversed",
    "love",
    "career",
    "mood",
    "spiritual",
]

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data,dict):
       df = pd.json_normalize([ { 'name': nome, **dados } for nome, dados in data.items() ])
    else:
       df = pd.json_normalize(data)
    print("Colunas: ")
    print(list(df.columns))
    print("Primeiras linhas: ", len(df))
    print(df.head())
    return df

def build(df):
   def definir_classe(row):
      if row['arcana'] == 'major':
         return 'major'
      return row['suit']
   df['label'] = df.apply(definir_classe, axis=1)
   print("Classes: ",df['label'].value_counts())
   return df

def features(df):
    categorias_ex = [c for c in categorias if c in df.columns]
    numericas_ex = [c for c in numericas if c in df.columns]
    texto_ex = [c for c in texto_col if c in df.columns]
    text_df = df[texto_ex].copy()
    def normalizar_texto(valor):
        if isinstance(valor, list):
            return ' '.join(str(item) for item in valor)

        if valor is None:
            return ''
        return str(valor)
    
    text_df = text_df.map(normalizar_texto)
    df['text_blob'] = text_df.agg(' '.join, axis=1)
    for col in numericas_ex:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    X = df[ categorias_ex + numericas_ex + ['text_blob']]
    y = df['label']
    return X, y, categorias_ex, numericas_ex

def preprocessamento(categorias,numericas):
   column_transformer = ColumnTransformer(
      transformers = [
         ( 'cat', OneHotEncoder( handle_unknown='ignore' ), categorias ), 
         ( 'num', 'passthrough', numericas ), 
         ( 'text', TfidfVectorizer( max_features=100, stop_words='english' ), 'text_blob' )
      ]
   )
   return column_transformer

def k_fold(X,y,categorias,numericas):
   skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
   modelos={
      'KNN=3': KNeighborsClassifier(n_neighbors=3),
      'KNN=5': KNeighborsClassifier(n_neighbors=5),
      'KNN=7': KNeighborsClassifier(n_neighbors=7),
      'KNN=8': KNeighborsClassifier(n_neighbors=8),
      'KNN=9': KNeighborsClassifier(n_neighbors=9),
      'KNN=10': KNeighborsClassifier(n_neighbors=10),
   }
   print('Resultados da validação cruzada:')
   results = []
   acuracias_media = {}
   for name,clf in modelos.items():
      fold_acuracia = []
      for train_index, test_index in skf.split(X, y):
         X_train = X.iloc[train_index]
         X_test = X.iloc[test_index]
         y_train = y.iloc[train_index]
         y_test = y.iloc[test_index]
         preprocessador = preprocessamento(categorias, numericas)
         X_train_transformed = (preprocessador.fit_transform(X_train))
         X_test_transformed = (preprocessador.transform(X_test))
         clf.fit(X_train_transformed, y_train)
         acuracia = clf.score(X_test_transformed, y_test)
         fold_acuracia.append(acuracia)
      acuracia_media = sum(fold_acuracia) / len(fold_acuracia)
      acuracias_media[name] = acuracia_media
      std_acuracia = (pd.Series(fold_acuracia).std())
      results.append({
         'Modelo': name,
         'Acurácia Média': f"{acuracia_media * 100:.2f}%",
         'Desvio Padrão': f"{std_acuracia * 100:.2f}%"
      })
   melhor = max(acuracias_media, key=acuracias_media.get)
   print(pd.DataFrame(results).to_string(index=False))
   print()
   print(f"Melhor: {melhor}" f" com acurácia média de {acuracias_media[melhor] * 100:.2f}%")

      
def main():
   df = load_dataset(dataset)
   df = build(df)
   X, y, categorias_ex,numericas_ex = features(df)
   k_fold(X, y, categorias_ex, numericas_ex)
if __name__ == "__main__":
   main()

