import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

algoritmo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if algoritmo not in sys.path:
  sys.path.append(algoritmo)

from main import (
    build,
    features,
    load_dataset,
    preprocessamento,
)
raiz = os.path.dirname(algoritmo)
dataset = os.path.join(raiz, "dataset", "cards.json")
def main():
  df = load_dataset(dataset)
  df = build(df)
  X, y, categorias_ex = features(df)
  y_traduzido = y.map(
      {"Major": "Arcanos Maiores", "Minor": "Arcanos Menores"}
  ).fillna(y)
  preprocessador = preprocessamento(categorias_ex)
  X_transformed = preprocessador.fit_transform(X).toarray()

  pca = PCA(n_components=2, random_state=42)
  X_2d = pca.fit_transform(X_transformed)
  plt.figure(figsize=(9, 6))
  sns.set_theme(style="whitegrid")
  sns.scatterplot(
      x=X_2d[:, 0],
      y=X_2d[:, 1],
      hue=y_traduzido,
      palette={
          "Arcanos Maiores": "#d32f2f", 
          "Arcanos Menores": "#1976d2",  
      },
      s=90,
      alpha=0.85,
  )

  plt.title(
      "Distribuição das Cartas de Tarot por KNN", fontsize=13, fontweight="bold"
  )
  plt.xlabel("Componente Principal 1")
  plt.ylabel("Componente Principal 2")
  plt.legend(title="Tipo de Arcano")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
  main()