import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


dataset = 'dataset/cards.json'

categorias = [
    'planet',
    'zodiac',
    'element',
    'yes_no',
    'yes_no_reversed'
]

numericas = ['number_numerology']

texto_col = [
    'keywords_upright',
    'keywords_reversed',
    'meaning_upright',
    'meaning_reversed',
    'love',
    'career',
    'mood',
    'spiritual'
]


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        df = pd.json_normalize([
            {'name': nome, **dados}
            for nome, dados in data.items()
        ])
    else:
        df = pd.json_normalize(data)

    print('Quantidade de cartas:', len(df))
    print('Colunas:', list(df.columns))
    return df


def build(df):
    def definir_classe(row):
        if row['arcana'] == 'major':
            return 'major'

        return row['suit']

    df['label'] = df.apply(definir_classe, axis=1)

    print('\nClasses:')
    print(df['label'].value_counts())

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

    for col in categorias_ex:
        df[col] = df[col].fillna('unknown')

    X = df[categorias_ex + numericas_ex + ['text_blob']]
    y = df['label']

    return X, y, categorias_ex, numericas_ex

def preprocessamento(categorias, numericas):
    return ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorias
            ),
            (
                'num',
                MinMaxScaler(),
                numericas
            ),
            (
                'text',
                TfidfVectorizer(
                    max_features=100,
                    stop_words='english'
                ),
                'text_blob'
            )
        ]
    )

def k_fold(X, y, categorias, numericas):
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    modelos = {
        'KNN=3': KNeighborsClassifier(n_neighbors=3),
        'KNN=5': KNeighborsClassifier(n_neighbors=5),
        'KNN=7': KNeighborsClassifier(n_neighbors=7),
        'Naive Bayes': MultinomialNB()
    }

    resultados = []
    medias = {}
    for nome, modelo in modelos.items():
        acuracias = []

        for train_index, test_index in skf.split(X, y):
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            prep = preprocessamento(categorias, numericas)

            X_train = prep.fit_transform(X_train)
            X_test = prep.transform(X_test)

            modelo.fit(X_train, y_train)

            acuracia = modelo.score(X_test, y_test)
            acuracias.append(acuracia)

        media = sum(acuracias) / len(acuracias)
        desvio = pd.Series(acuracias).std()

        medias[nome] = media

        resultados.append({
            'Modelo': nome,
            'Acurácia Média': f'{media * 100:.2f}%',
            'Desvio Padrão': f'{desvio * 100:.2f}%'
        })

    print('\nResultados:')
    print(pd.DataFrame(resultados).to_string(index=False))

    knns = {
        nome: media
        for nome, media in medias.items()
        if nome.startswith('KNN')
    }

    melhor_knn = max(knns, key=knns.get)

    media_knn = knns[melhor_knn]
    media_nb = medias['Naive Bayes']

    print()
    print(f'Melhor KNN: {melhor_knn} - {media_knn * 100:.2f}%')
    print(f'Naive Bayes: {media_nb * 100:.2f}%')

    print('Comparação final:')

    if media_knn > media_nb:
        diferenca = (media_knn - media_nb) * 100

        print(f'{melhor_knn}: {media_knn * 100:.2f}%')
        print(f'Naive Bayes: {media_nb * 100:.2f}%')

        print(
            f'\nO {melhor_knn} foi mais eficiente porque apresentou '
            f'maior acurácia média durante a validação cruzada de 5 folds. '
            f'O KNN obteve {media_knn * 100:.2f}% de acurácia, '
            f'enquanto o Naive Bayes obteve {media_nb * 100:.2f}%. '
            f'A diferença foi de {diferenca:.2f} pontos percentuais.'
        )

    elif media_nb > media_knn:
        diferenca = (media_nb - media_knn) * 100

        print(f'{melhor_knn}: {media_knn * 100:.2f}%')
        print(f'Naive Bayes: {media_nb * 100:.2f}%')

        print(
            f'\nO Naive Bayes foi mais eficiente porque apresentou '
            f'maior acurácia média durante a validação cruzada de 5 folds. '
            f'O Naive Bayes obteve {media_nb * 100:.2f}% de acurácia, '
            f'enquanto o {melhor_knn} obteve {media_knn * 100:.2f}%. '
            f'A diferença foi de {diferenca:.2f} pontos percentuais.'
        )

    else:
        print('\nOs dois algoritmos tiveram a mesma acurácia média.')

def main():
    df = load_dataset(dataset)
    df = build(df)
    X, y, categorias_ex, numericas_ex = features(df)
    k_fold(X, y, categorias_ex, numericas_ex)

if __name__ == '__main__':
    main()