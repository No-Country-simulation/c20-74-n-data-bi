
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder




# Cargar el archivo CSV
archivo = '/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/Base_de_datos_reingreso.csv'  # Cambia esta ruta a la ubicación de tu archivo
df = pd.read_csv(archivo)

# Mostrar las primeras filas del dataset
print(df.head())
print(df.columns)

# Información general del dataset
print(df.info())

# Descripción estadística de las variables numéricas
print(df.describe())

# Verificar valores nulos
print(df.isnull().sum())

le = LabelEncoder()
df['Genero'] = le.fit_transform(df['Genero'])
df['Tipo_de_Seguro'] = le.fit_transform(df['Tipo_de_Seguro'])
df['Estado_Nutricional'] = le.fit_transform(df['Estado_Nutricional'])

# Distribución de la Edad
plt.figure(figsize=(10, 6))
sns.histplot(df['Edad'], bins=20, kde=True)
plt.title('Distribución de la Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()

# Distribución por Género
plt.figure(figsize=(6, 4))
sns.countplot(x='Genero', data=df)
plt.title('Distribución por Género')
plt.xlabel('Género')
plt.ylabel('Frecuencia')
plt.show()

# Relación entre Edad y Duración de Hospitalización
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Edad', y='Duracion_Hospitalizacion_Dias', data=df)
plt.title('Edad vs Duración de Hospitalización')
plt.xlabel('Edad')
plt.ylabel('Días de Hospitalización')
plt.show()

# Boxplot de la Duración de Hospitalización según el Género
plt.figure(figsize=(10, 6))
sns.boxplot(x='Genero', y='Duracion_Hospitalizacion_Dias', data=df)
plt.title('Duración de Hospitalización según Género')
plt.xlabel('Genero')
plt.ylabel('Días de Hospitalización')
plt.show()

# Conteo de las Condiciones Crónicas
plt.figure(figsize=(10, 6))
sns.countplot(x='Condiciones_Cronicas', data=df)
plt.title('Distribución de Condiciones Crónicas')
plt.xlabel('Condiciones Crónicas')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras para Tipo de Seguro
plt.figure(figsize=(10, 6))
sns.countplot(x='Tipo_de_Seguro', data=df)
plt.title('Distribución de Tipo de Seguro')
plt.xlabel('Tipo de Seguro')
plt.ylabel('Frecuencia')
plt.show()

# Relación entre IMC y Estado Nutricional
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Indice_de_Masa_Corporal', y='Estado_Nutricional', data=df)
plt.title('IMC vs Estado Nutricional')
plt.xlabel('Indice de Masa Corporal')
plt.ylabel('Estado Nutricional')
plt.show()

# Correlación entre variables numéricas
numeric_df = df.select_dtypes(include=[float, int])
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# Generar el reporte de datos con ydata-profiling
try:
    # Create a ProfileReport object
    profile = ProfileReport(df, title="Reporte EDA - Reingreso de Pacientes", explorative=True)

    # Save the report as an HTML file
    profile.to_file("/Users/kreynoso/Desktop/Kez/Proyectos/NoCountry/reporte_eda_reingreso_pacientes.html")
    print("Reporte generado exitosamente.")
except Exception as e:
    print(f"Error al generar el reporte: {e}")