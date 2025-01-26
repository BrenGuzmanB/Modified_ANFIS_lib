"""
Created on Sat Nov 30 00:45:19 2024

@author: Bren Guzmán
"""

import anfis
import membership  # import membershipfunction, mfDerivs
import numpy as np
import matplotlib.pyplot as plt
from anfis import predict

# 1. Generar un conjunto de datos de ejemplo
num_samples = 500  # Número de muestras

# Variables de entrada
temperaturas = np.random.rand(num_samples, 1) * 20 + 15  # Temperatura entre 15°C y 35°C
horas_uso = np.random.rand(num_samples, 1) * 11 + 1  # Horas de uso entre 1 y 12
ocupantes = np.random.randint(1, 6, size=(num_samples, 1))  # Número de ocupantes entre 1 y 5

# Variable de salida: Consumo energético estimado (kWh)
# Relación hipotética entre las variables de entrada y la salida
consumo = 0.1 * temperaturas + 0.05 * horas_uso + 0.2 * ocupantes + np.random.randn(num_samples, 1) * 0.5  # Consumo energético con algo de ruido

# Combina las entradas y la salida en un solo conjunto de datos
datos = np.hstack([temperaturas, horas_uso, ocupantes, consumo])

# Dividir los datos en entrenamiento y prueba
train_size = int(0.8 * num_samples)  # 80% para entrenamiento
train_data = datos[:train_size, :]
test_data = datos[train_size:, :]

# 2. Preparar los datos para ANFIS (X = entradas, Y = salida)
X = train_data[:, 0:3]  # Tomamos las 3 primeras columnas como entradas (temperatura, horas de uso, ocupantes)
Y = train_data[:, 3]    # La cuarta columna es la salida (consumo energético)

# 3. Definir las funciones de membresía (MF) para cada variable de entrada
mf = [
    [['gaussmf', {'mean': 20., 'sigma': 5.}], ['gaussmf', {'mean': 25., 'sigma': 5.}], ['gaussmf', {'mean': 30., 'sigma': 5.}]],  # Temperatura
    [['gaussmf', {'mean': 6., 'sigma': 2.}], ['gaussmf', {'mean': 9., 'sigma': 2.}], ['gaussmf', {'mean': 3., 'sigma': 2.}]],   # Horas de uso
    [['gaussmf', {'mean': 2., 'sigma': 1.}], ['gaussmf', {'mean': 3., 'sigma': 1.}], ['gaussmf', {'mean': 4., 'sigma': 1.}]]    # Ocupantes
]

# 4. Inicializar la función de membresía y el modelo ANFIS
mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)

# 5. Entrenar el modelo ANFIS usando el método de aprendizaje híbrido (método de Jang)
anf.trainHybridJangOffLine(epochs=10)

# 6. Imprimir los valores finales de los consecuentes y los valores ajustados
print(f"Último consecuente: {round(anf.consequents[-1][0], 6)}")
print(f"Penúltimo consecuente: {round(anf.consequents[-2][0], 6)}")
print(f"Valor ajustado en el índice 9: {round(anf.fittedValues[9][0], 6)}")

# 7. Evaluar el modelo ANFIS en los datos de prueba
X_test = test_data[:, 0:3]
Y_test = test_data[:, 3]

# Predecir utilizando el modelo entrenado
predicciones =predict(anf, X_test)

# Calcular el RMSE (Root Mean Square Error)
rmse = np.sqrt(np.mean((predicciones - Y_test) ** 2))  # Error cuadrático medio
print(f"RMSE en los datos de prueba: {rmse}")

# 8. Visualizar los errores y los resultados
anf.plotErrors()
anf.plotResults()

# Graficar predicciones vs valores reales
plt.figure()
plt.scatter(Y_test, predicciones, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--')
plt.title('Predicciones vs Real')
plt.xlabel('Consumo real (kWh)')
plt.ylabel('Consumo predicho (kWh)')
plt.show()

# Función para dividir los datos en entrenamiento y prueba
def split_data(datos, porcentaje_entrenamiento):
    num_muestras = datos.shape[0]
    num_entrenamiento = int(porcentaje_entrenamiento * num_muestras)
    indices = np.random.permutation(num_muestras)
    train_data = datos[indices[:num_entrenamiento], :]
    test_data = datos[indices[num_entrenamiento:], :]
    return train_data, test_data
