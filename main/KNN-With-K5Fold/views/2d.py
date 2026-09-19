import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

algoritmo = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)
if algoritmo not in sys.path:
    sys.path.append(algoritmo)

from main import (
    build,
    features,
    load_dataset,
    preprocessamento,
)

raiz = os.path.dirname(algoritmo)
dataset = os.path.join(
    raiz,
    "dataset",
    "cards.json"
)


def main():
    df = load_dataset(dataset)
    df = build(df)
    X, y, categorias_ex, numericas_ex = features(df)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.canvas.manager.set_window_title("Distribuição das Cartas de Tarot por Classe (Naipes)")

    y_traduzido = y.map({
        "major": "Arcanos Maiores",
        "wands": "Paus",
        "cups": "Copas",
        "swords": "Espadas",
        "pentacles": "Denários"
    }).fillna(y)

    preprocessador = preprocessamento(
        categorias_ex,
        numericas_ex
    )

    X_transformed = preprocessador.fit_transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    pca = PCA(
        n_components=2,
        random_state=42
    )

    X_2d = pca.fit_transform(
        X_transformed
    )
    variancia = pca.explained_variance_ratio_
    sns.set_theme(style="whitegrid")
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=y_traduzido,
        palette={
            "Arcanos Maiores": "#d32f2f",
            "Paus": "#f57c00",
            "Copas": "#1976d2",
            "Espadas": "#7b1fa2",
            "Denários": "#388e3c"
        }, 
        s=100, 
        alpha=0.85,
        ax=ax
    )
    
    plt.title(
        "Distribuição das Cartas de Tarot por Classe (Naipes)",
        fontsize=14,
        fontweight="bold"
    )
    plt.xlabel(
        f"Componente Principal 1 "
        f"({variancia[0] * 100:.2f}% da variância)"
    )

    plt.ylabel(
        f"Componente Principal 2 "
        f"({variancia[1] * 100:.2f}% da variância)"
    )

    plt.legend(
        title="Classe"
    )

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()