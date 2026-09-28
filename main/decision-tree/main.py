import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

dataset = 'dataset/Dataset-Sheet.xlsx'
df = pd.read_excel(dataset)
categorias = ['planet', 'zodiac', 'yes_no', 'yes_no_reversed']
numericas = ['number_numerology']
print('\nQuantidade de cartas:', len(df))

def classes(row):
    if row['arcana'] == 'major':
        return 'major'
    return row['suit']
df['label'] = df.apply(classes, axis=1)
print('\nClasses:')
print(df['label'].value_counts())

features = df[categorias + numericas].copy()
classe = df['label']

for coluna in categorias:
    features[coluna] = features[coluna].fillna('unknown')
for coluna in numericas:
    features[coluna] = pd.to_numeric(features[coluna], errors='coerce')
    features[coluna] = features[coluna].fillna(features[coluna].median())

preprocessador = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ),
            categorias
        ),
        (
            'num',
            'passthrough',
            numericas
        )
    ]
)

features_transformadas = preprocessador.fit_transform(features)

treinamento, teste, treinamento_classe, teste_classe = (train_test_split(
    features_transformadas,
    classe,
    test_size=0.25,
    random_state=42,
    stratify=classe
))

classificador = DecisionTreeClassifier(
    criterion='gini',
    # criterion'='entropy',
    max_depth=3,
    random_state=42
)

classificador.fit(treinamento, treinamento_classe)

previsao = classificador.predict(teste)
acuracia = accuracy_score(teste_classe, previsao)
print('\nAcurácia do modelo de árvore de decisão:', acuracia)

matrix = confusion_matrix(teste_classe, previsao,labels=classificador.classes_)
print('\nOrdem:')
print(classificador.classes_)
print('\nMatriz de Confusão:')
print(matrix)