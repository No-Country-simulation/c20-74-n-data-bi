import pandas as pd
import os

# Cargar el archivo CSV en un DataFrame
archivo = '/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/Base_de_datos_reingreso.csv'  # Cambia esta ruta a la ubicación de tu archivo
df = pd.read_csv(archivo)

# Verificar los valores únicos en Adherencia_Medica y Soporte_Familiar
print("Valores únicos en Adherencia_Medica:", df['Adherencia_Medica'].unique())
print("Valores únicos en Soporte_Familiar:", df['Soporte_Familiar'].unique())

# Limpiar espacios y convertir a minúsculas antes de mapear
df['Condiciones_Cronicas'] = df['Condiciones_Cronicas'].str.strip().str.lower()

# Mapeo para la columna Condiciones_Cronicas
condiciones_cronicas_map = {
    'ninguna': 0,
    'una': 1,
    'dos': 2,
    'tres o más': 3
}
df['Condiciones_Cronicas'] = df['Condiciones_Cronicas'].map(condiciones_cronicas_map)

# Verificar si la columna se ha convertido correctamente
print(df['Condiciones_Cronicas'].head())

# 1. Codificar variables binarias (0 y 1)
df['Genero'] = df['Genero'].apply(lambda x: 1 if x == 'Masculino' else 0)

# 2. Variables ordinales (con jerarquía)
nivel_educacion_map = {
    'Sin educación': 0,
    'Primaria': 1,
    'Secundaria': 2,
    'Preparatoria': 3,
    'Universitaria': 4,
    'Postgrado': 5
}
df['Nivel_de_Educacion'] = df['Nivel_de_Educacion'].map(nivel_educacion_map)

# 3. Identificar columnas categóricas con más de dos valores únicos
columnas_categoricas = df.select_dtypes(include=['object']).columns
columnas_mas_de_dos_valores = [col for col in columnas_categoricas if df[col].nunique() > 2]

# Definir mapeos jerárquicos o personalizados para columnas con más de dos valores únicos
mapeos = {
    'Tipo_Seguimiento_Post_Hospitalizacion': {valor: idx for idx, valor in enumerate(df['Tipo_Seguimiento_Post_Hospitalizacion'].unique())},
    'Adherencia_Medica': {valor: idx for idx, valor in enumerate(df['Adherencia_Medica'].unique())},
    'Soporte_Familiar': {valor: idx for idx, valor in enumerate(df['Soporte_Familiar'].unique())},
    'Tipo_de_Seguro': {valor: idx for idx, valor in enumerate(df['Tipo_de_Seguro'].unique())},
    'Estado_Mental': {valor: idx for idx, valor in enumerate(df['Estado_Mental'].unique())},
    'Frecuencia_Actividad_Fisica': {valor: idx for idx, valor in enumerate(df['Frecuencia_Actividad_Fisica'].unique())},
    'Consumo_de_Tabaco': {valor: idx for idx, valor in enumerate(df['Consumo_de_Tabaco'].unique())},
    'Consumo_de_Alcohol': {valor: idx for idx, valor in enumerate(df['Consumo_de_Alcohol'].unique())},
    'Condicion_Laboral': {valor: idx for idx, valor in enumerate(df['Condicion_Laboral'].unique())},
    'Calidad_de_Vivienda': {valor: idx for idx, valor in enumerate(df['Calidad_de_Vivienda'].unique())},
    'Prescripcion_Medicamentos': {valor: idx for idx, valor in enumerate(df['Prescripcion_Medicamentos'].unique())},
    'Apoyo_Social': {valor: idx for idx, valor in enumerate(df['Apoyo_Social'].unique())},
    'Motivo_Hospitalizacion': {valor: idx for idx, valor in enumerate(df['Motivo_Hospitalizacion'].unique())},
    'Estado_Nutricional': {valor: idx for idx, valor in enumerate(df['Estado_Nutricional'].unique())},
    'Comorbilidades': {valor: idx for idx, valor in enumerate(df['Comorbilidades'].unique())},
    'Frecuencia_de_Visitas_Familiares': {valor: idx for idx, valor in enumerate(df['Frecuencia_de_Visitas_Familiares'].unique())},
    'Historial_de_Alergias': {valor: idx for idx, valor in enumerate(df['Historial_de_Alergias'].unique())}
}

# Aplicar el mapeo
for col, mapeo in mapeos.items():
    df[col] = df[col].map(mapeo)

# Clasificar la distancia al hospital
def clasificar_distancia(distancia):
    if pd.isna(distancia):
        return 'Desconocida'
    elif distancia <= 5:
        return 'Cercana'
    elif distancia <= 15:
        return 'Moderada'
    else:
        return 'Lejana'

df['Clasificacion_Distancia'] = df['Distancia_Al_Hospital'].apply(clasificar_distancia)

# Verificar la nueva columna
print(df[['Distancia_Al_Hospital', 'Clasificacion_Distancia']].head())

# Crear la carpeta si no existe
directorio = '/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry'
if not os.path.exists(directorio):
    os.makedirs(directorio)

# Guardar el DataFrame en la carpeta especificada
ruta_archivo = os.path.join(directorio, 'archivo_transformado.csv')
df.to_csv(ruta_archivo, index=False)

print(f"Transformación completada. Archivo guardado en '{ruta_archivo}'.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Cargar el archivo CSV en un DataFrame
archivo = '/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/archivo_transformado.csv'
df = pd.read_csv(archivo)

# Eliminar la columna 'Distancia_Al_Hospital'
df = df.drop(columns='Distancia_Al_Hospital')

# Identificar la variable objetivo y las características
X = df.drop(columns='Reingreso')
y = df['Reingreso']

# Separar las características en numéricas y categóricas
columnas_numericas = X.select_dtypes(include=[np.number]).columns.tolist()
columnas_categoricas = X.select_dtypes(include=[object]).columns.tolist()

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear el preprocesador y el modelo
preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(), columnas_categoricas)
    ]
)

# Pipeline para regresión logística
pipeline_logistic = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', LogisticRegression(max_iter=1000, random_state=42))
])
pipeline_logistic.fit(X_train, y_train)

# Pipeline para RandomForest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
pipeline_rf = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', rf_model)
])
pipeline_rf.fit(X_train, y_train)

# Pipeline para Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
pipeline_gb = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('modelo', gb_model)
])
pipeline_gb.fit(X_train, y_train)

# Obtener nombres de características
preprocesador_ajustado = pipeline_logistic.named_steps['preprocesador']
onehot_encoder = preprocesador_ajustado.named_transformers_['cat']
onehot_encoder.fit(X_train[columnas_categoricas])  # Ajustar el OneHotEncoder
nombres_columnas = onehot_encoder.get_feature_names_out(columnas_categoricas)

nombres_columnas_numericas = columnas_numericas
nombres_columnas_rf = np.concatenate([nombres_columnas_numericas, nombres_columnas])

# Obtener coeficientes de regresión logística
coeficientes = pipeline_logistic.named_steps['modelo'].coef_[0]
df_coeficientes = pd.DataFrame({
    'Característica': nombres_columnas_rf,
    'Coeficiente_Logistico': coeficientes
})

# Calcular los valores absolutos de los coeficientes
df_coeficientes['Abs_Coeficiente_Logistico'] = df_coeficientes['Coeficiente_Logistico'].abs()

# Obtener importancias de características de RandomForest
importancias_rf = pipeline_rf.named_steps['modelo'].feature_importances_
df_importancias_rf = pd.DataFrame({
    'Característica': nombres_columnas_rf,
    'Importancia_RF': importancias_rf
})

# Obtener importancias de características de Gradient Boosting
importancias_gb = pipeline_gb.named_steps['modelo'].feature_importances_
df_importancias_gb = pd.DataFrame({
    'Característica': nombres_columnas_rf,
    'Importancia_GB': importancias_gb
})

# Normalizar importancias y coeficientes
df_coeficientes['Coeficiente_Logistico_Normalizado'] = df_coeficientes['Abs_Coeficiente_Logistico'] / df_coeficientes['Abs_Coeficiente_Logistico'].max()
df_importancias_rf['Importancia_RF_Normalizada'] = df_importancias_rf['Importancia_RF'] / df_importancias_rf['Importancia_RF'].max()
df_importancias_gb['Importancia_GB_Normalizada'] = df_importancias_gb['Importancia_GB'] / df_importancias_gb['Importancia_GB'].max()

# Combinar las importancias
df_combined = pd.merge(df_coeficientes[['Característica', 'Coeficiente_Logistico_Normalizado']],
                       df_importancias_rf[['Característica', 'Importancia_RF_Normalizada']],
                       on='Característica', how='left')

df_combined = pd.merge(df_combined,
                       df_importancias_gb[['Característica', 'Importancia_GB_Normalizada']],
                       on='Característica', how='left')

df_combined['Importancia_Combinada'] = (df_combined['Coeficiente_Logistico_Normalizado'] + 
                                        df_combined['Importancia_RF_Normalizada'].fillna(0) + 
                                        df_combined['Importancia_GB_Normalizada'].fillna(0)) / 3

df_combined = df_combined.sort_values(by='Importancia_Combinada', ascending=False)

# Mostrar las variables más importantes combinadas
print("Variables más importantes combinadas:")
print(df_combined)

# Crear y guardar la gráfica combinada
plt.figure(figsize=(12, 8))
plt.bar(df_combined['Característica'], df_combined['Importancia_Combinada'], color='lightcoral')
plt.xlabel('Características')
plt.ylabel('Importancia Combinada')
plt.title('Importancia Combinada de las Variables en los Modelos de Regresión Logística, RandomForest y Gradient Boosting')
plt.xticks(rotation=90)  # Rotar etiquetas del eje x para mayor legibilidad
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir anotaciones
for index, value in enumerate(df_combined['Importancia_Combinada']):
    plt.text(index, value, f'{value:.2f}', va='bottom', ha='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/importancia_combinada.png')  # Guardar la gráfica combinada
plt.close()

# Crear y guardar la gráfica de importancias de RandomForest ordenada en orden ascendente
df_importancias_rf_ordenada = df_importancias_rf.sort_values(by='Importancia_RF_Normalizada', ascending=True)

plt.figure(figsize=(14, 10))  # Aumentar el tamaño de la figura
plt.barh(df_importancias_rf_ordenada['Característica'], df_importancias_rf_ordenada['Importancia_RF_Normalizada'], color='lightgreen')
plt.xlabel('Importancia Normalizada')
plt.ylabel('Características')
plt.title('Importancia de las Variables en el Modelo de RandomForest')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadir anotaciones
for index, value in enumerate(df_importancias_rf_ordenada['Importancia_RF_Normalizada']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')

# Ajustar el espaciado manualmente
plt.subplots_adjust(left=0.2, right=0.8, top=0.9, bottom=0.1)

plt.savefig('/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/importancia_rf_ordenada.png')  # Guardar la gráfica de RandomForest ordenada
plt.close()

# Crear y guardar la gráfica de importancias de Gradient Boosting ordenada en orden ascendente
df_importancias_gb_ordenada = df_importancias_gb.sort_values(by='Importancia_GB_Normalizada', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(df_importancias_gb_ordenada['Característica'], df_importancias_gb_ordenada['Importancia_GB_Normalizada'], color='lightblue')
plt.xlabel('Importancia Normalizada')
plt.ylabel('Características')
plt.title('Importancia de las Variables en el Modelo de Gradient Boosting')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadir anotaciones
for index, value in enumerate(df_importancias_gb_ordenada['Importancia_GB_Normalizada']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/importancia_gb_ordenada.png')  # Guardar la gráfica de Gradient Boosting ordenada
plt.close()

# Crear y guardar la gráfica de coeficientes de Regresión Logística ordenada en orden ascendente
df_coeficientes_ordenado = df_coeficientes.sort_values(by='Coeficiente_Logistico_Normalizado', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(df_coeficientes_ordenado['Característica'], df_coeficientes_ordenado['Coeficiente_Logistico_Normalizado'], color='lightcoral')
plt.xlabel('Coeficiente Normalizado')
plt.ylabel('Características')
plt.title('Coeficientes Normalizados en el Modelo de Regresión Logística')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Añadir anotaciones
for index, value in enumerate(df_coeficientes_ordenado['Coeficiente_Logistico_Normalizado']):
    plt.text(value, index, f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/coefs_regresion_logistica_ordenado.png')  # Guardar la gráfica de coeficientes de regresión logística ordenada
plt.close()
