from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from main import preprocessador, classificador

feature_names = preprocessador.get_feature_names_out()
plt.figure(figsize=(22, 10))
plot_tree(
    classificador,
    feature_names=feature_names,
    class_names=classificador.classes_,
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title(
    'Árvore de Decisão - Classificação das Cartas de Tarot'
)
plt.tight_layout()
plt.show()