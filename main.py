import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados de treino
jogadores_data = pd.read_csv('jogadores.csv', delimiter=';')
le = LabelEncoder()
jogadores_data['Classe'] = le.fit_transform(jogadores_data['Classe']) # Transformando a classe em número (Defensor: 0 e Atacante: 1)

# Separando os dados gerais (X_train) e o rótulo da classificação (y_train)
X_train = jogadores_data.drop('Classe', axis=1)
y_train = jogadores_data['Classe']

# Treinando o knn com os dados de treino
knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train)

# Carregar os dados para validação
validation_data = pd.read_csv('validacao.csv', delimiter=';')

# Prevendo classes dos dados de validação
predicted_classes = knn.predict(validation_data)
predicted_classes = le.inverse_transform(predicted_classes)

# Adicionar as classes previstas ao DataFrame original
validation_data['Classe'] = predicted_classes

print(validation_data)

# Plotar um gráfico de dispersão
sns.scatterplot(data=jogadores_data, x='Velocidade', y='Drible', hue='Classe', marker='o')
sns.scatterplot(data=validation_data, x='Velocidade', y='Drible', hue='Classe', marker='X')
plt.title('Gráfico de Dispersão - Velocidade vs Drible')
plt.legend()
plt.show()
