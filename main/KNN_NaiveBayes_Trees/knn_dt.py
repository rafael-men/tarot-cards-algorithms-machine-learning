import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline


dataset = '../dataset/Dataset-Sheet.xlsx'

df = pd.read_excel(dataset)

categorias = ['planet','zodiac','yes_no','yes_no_reversed']
numericas = ['number_numerology']

def classes(row):
    if row['arcana'] == 'major':
        return 'major'
    return row['suit']

df['label'] = df.apply(
    classes,
    axis=1
)

# prepara features

features = df[
    categorias + numericas
].copy()
y = df['label']

for coluna in categorias:
    features[coluna] = (
        features[coluna]
        .fillna('unknown')
    )

for coluna in numericas:
    features[coluna] = pd.to_numeric(
        features[coluna],
        errors='coerce'
    )
    features[coluna] = features[coluna].fillna(
        features[coluna].median()
    )

preprocessador = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(
                handle_unknown='ignore'
            ),
            categorias
        ),
        (
            'num',
            StandardScaler(),
            numericas
        )
    ]
)

# treinamento estratificado

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# árvore de decisão

classificador = Pipeline(
    steps=[
        (
            'preprocessador',
            preprocessador
        ),
        (
            'modelo',
            DecisionTreeClassifier(
                criterion='gini',
                max_depth=3,
                random_state=42
            )
        )
    ]
)

resultados_arvore = cross_val_score(
    classificador,
    features,
    y,
    cv=cv,
    scoring='accuracy'
)

print(
    "Acurácia da árvore de decisão: %.4f ± %.4f"
    % (
        resultados_arvore.mean(),
        resultados_arvore.std()
    )
)
# KNN com k = 5

knn = Pipeline(
    steps=[
        (
            'preprocessador',
            preprocessador
        ),

        (
            'modelo',
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]
)

resultados_knn = cross_val_score(
    knn,
    features,
    y,
    cv=cv,
    scoring='accuracy'
)

print(
    "Acurácia do KNN com k=5: %.4f ± %.4f"
    % (
        resultados_knn.mean(),
        resultados_knn.std()
    )
)

classificador.fit(
    X_train,
    y_train
)

y_pred_dt = classificador.predict(
    X_test
)

print(
    "\nResultados da árvore de decisão:"
)

print(
    classification_report(
        y_test,
        y_pred_dt
    )
)


classes_arvore = (
    classificador
    .named_steps['modelo']
    .classes_
)


print(
    "Matriz de confusão:\n",
    confusion_matrix(
        y_test,
        y_pred_dt,
        labels=classes_arvore
    )
)

knn.fit(
    X_train,
    y_train
)

y_pred_knn = knn.predict(
    X_test
)

print("\nResultados do KNN:")
print(classification_report(y_test,y_pred_knn))
classes_knn = (knn.named_steps['modelo'].classes_)
print("Matriz de confusão:\n",confusion_matrix(y_test,y_pred_knn, labels=classes_knn))

acc_dt = (
    y_pred_dt == y_test
).mean()

acc_knn = (
    y_pred_knn == y_test
).mean()

f1_dt = f1_score(y_test,y_pred_dt,average='macro')
f1_knn = f1_score(y_test,y_pred_knn,average='macro')

print("\nAcurácia - Árvore de Decisão: %.4f | KNN k=5: %.4f"% (acc_dt,acc_knn))
print("Teste macro F1 - Árvore de Decisão: %.4f | KNN k=5: %.4f"% (f1_dt,f1_knn))
print("\nValidação cruzada 5-Fold:")
print("Árvore de Decisão: %.2f%% ± %.2f%%"% (resultados_arvore.mean() * 100,resultados_arvore.std() * 100))
print("KNN k=5: %.2f%% ± %.2f%%"% (resultados_knn.mean() * 100,resultados_knn.std() * 100))