import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = '../dataset/cards.json'
nome_col = 'name'
categorias = ['planet', 'zodiac', 'element']
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

arcanos_maiores = {
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil", "The Tower",
    "The Star", "The Moon", "The Sun", "Judgement", "The World",
}

def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data,dict):
       df = pd.json_normalize([{nome_col: k, **v} for k, v in data.items()])
    else:
       df = pd.json_normalize(data)
    print("Colunas: ")
    print(list(df.columns))
    print("Primeiras linhas: ", len(df))
    print(df.head())
    return df

def build(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["arcana"]
    print('Classes: ')
    print(df["label"].unique())
    return df

def features(df: pd.DataFrame):
  categorias_ex = [c for c in categorias if c in df.columns]
  texto_ex = [c for c in texto_col if c in df.columns]
  text_df = df[texto_ex].copy()
  for col in texto_ex:
    text_df[col] = text_df[col].apply(
        lambda x: " ".join(x)
        if isinstance(x, list)
        else (str(x) if pd.notna(x) else "")
    )
  df["text_blob"] = text_df.agg(" ".join, axis=1)
  X = df[categorias_ex + ["text_blob"]]
  y = df["label"]
  return X, y, categorias_ex

def preprocessamento(categorias):
   column_transformer = ColumnTransformer(
      transformers = [
         ('cat', OneHotEncoder(handle_unknown='ignore'), categorias),
         ('text', TfidfVectorizer(max_features=100, stop_words='english'), 'text_blob')
      ]
   )
   return column_transformer

def k_fold(X,y,categorias):
   preprocessador = preprocessamento(categorias)
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
      fold_accuracies = []
      for train_index, test_index in skf.split(X, y):
         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
         X_train_transformed = preprocessador.fit_transform(X_train)
         X_test_transformed = preprocessador.transform(X_test)
         clf.fit(X_train_transformed, y_train)
         accuracy = clf.score(X_test_transformed, y_test)
         fold_accuracies.append(accuracy)
      mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
      acuracias_media[name] = mean_accuracy
      results.append({
         'Modelo': name,
         'Acurácia Média': f'{mean_accuracy*100:.2f}%',
         'Desvio Padrão': f'{pd.Series(fold_accuracies).std()*100:.2f}%'
      })
   melhor = max(acuracias_media, key=acuracias_media.get)
   print(pd.DataFrame(results).to_string(index=False))
   print(f"Melhor: {melhor}" f" com acurácia média de {acuracias_media[melhor] * 100:.2f}%")

      
def main():
   df = load_dataset(dataset)
   df = build(df)
   X, y, categorias_ex = features(df)
   k_fold(X, y, categorias_ex)
if __name__ == "__main__":
   main()

