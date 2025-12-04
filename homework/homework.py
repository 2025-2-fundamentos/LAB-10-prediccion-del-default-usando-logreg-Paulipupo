# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def cargar_datos_crudos(archivo_comprimido: str) -> pd.DataFrame:
    datos = pd.read_csv(archivo_comprimido, compression="zip").copy()

    datos.rename(columns={"default payment next month": "default"}, inplace=True)
    
    if "ID" in datos.columns:
        datos.drop(columns=["ID"], inplace=True)


    datos = datos[(datos["EDUCATION"] != 0) & (datos["MARRIAGE"] != 0)]
 
    datos["EDUCATION"] = datos["EDUCATION"].apply(lambda valor: 4 if valor > 4 else valor)

    return datos.dropna()


def separar_caracteristicas_objetivo(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    caracteristicas = dataset.drop(columns=["default"])
    objetivo = dataset["default"]
    return caracteristicas, objetivo


def construir_optimizador(numero_caracteristicas: int) -> GridSearchCV:
    
    columnas_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]

    transformador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
            ("num", MinMaxScaler(), [])
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    clasificador = LogisticRegression(max_iter=1000, random_state=42)

   
    flujo_trabajo = Pipeline(
        steps=[
            ("prep", transformador),
            ("kbest", SelectKBest(score_func=f_regression)),
            ("clf", clasificador),
        ]
    )

    espacio_parametros = {
        "kbest__k": list(range(1, numero_caracteristicas + 1)),
        "clf__C": [0.1, 1, 10],
        "clf__solver": ["liblinear", "lbfgs"],
    }

    optimizador = GridSearchCV(
        estimator=flujo_trabajo,
        param_grid=espacio_parametros,
        scoring="balanced_accuracy",
        cv=10,
        refit=True,
        n_jobs=-1,
    )
    
    return optimizador


def calcular_metricas_rendimiento(etiqueta_conjunto: str, valores_reales, valores_predichos) -> dict:
   
    metricas = {
        "type": "metrics",
        "dataset": etiqueta_conjunto,
        "precision": precision_score(valores_reales, valores_predichos, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(valores_reales, valores_predichos),
        "recall": recall_score(valores_reales, valores_predichos, zero_division=0),
        "f1_score": f1_score(valores_reales, valores_predichos, zero_division=0),
    }
    return metricas


def generar_matriz_confusion(etiqueta_conjunto: str, valores_reales, valores_predichos) -> dict:
    matriz = confusion_matrix(valores_reales, valores_predichos)
    
    resultado = {
        "type": "cm_matrix",
        "dataset": etiqueta_conjunto,
        "true_0": {
            "predicted_0": int(matriz[0, 0]), 
            "predicted_1": int(matriz[0, 1])
        },
        "true_1": {
            "predicted_0": int(matriz[1, 0]), 
            "predicted_1": int(matriz[1, 1])
        },
    }
    return resultado


def persistir_modelo(modelo_entrenado, ruta_salida: str) -> None:
    
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    with gzip.open(ruta_salida, "wb") as archivo:
        pickle.dump(modelo_entrenado, archivo)


def guardar_resultados(lista_resultados: list, ruta_archivo: str) -> None:
    os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        for resultado in lista_resultados:
            archivo.write(json.dumps(resultado) + "\n")


def ejecutar_pipeline_completo() -> None:
   
    datos_entrenamiento = cargar_datos_crudos("files/input/train_data.csv.zip")
    datos_prueba = cargar_datos_crudos("files/input/test_data.csv.zip")

    X_entrenamiento, y_entrenamiento = separar_caracteristicas_objetivo(datos_entrenamiento)
    X_prueba, y_prueba = separar_caracteristicas_objetivo(datos_prueba)


    buscador_parametros = construir_optimizador(numero_caracteristicas=X_entrenamiento.shape[1])

    cols_categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    cols_numericas = [col for col in X_entrenamiento.columns if col not in cols_categoricas]

    
    buscador_parametros.estimator.named_steps["prep"] = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cols_categoricas),
            ("num", MinMaxScaler(), cols_numericas),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

   
    buscador_parametros.fit(X_entrenamiento, y_entrenamiento)


    persistir_modelo(buscador_parametros, "files/models/model.pkl.gz")

    predicciones_entrenamiento = buscador_parametros.predict(X_entrenamiento)
    predicciones_prueba = buscador_parametros.predict(X_prueba)

 
    resultados_evaluacion = [
        calcular_metricas_rendimiento("train", y_entrenamiento, predicciones_entrenamiento),
        calcular_metricas_rendimiento("test", y_prueba, predicciones_prueba),
        generar_matriz_confusion("train", y_entrenamiento, predicciones_entrenamiento),
        generar_matriz_confusion("test", y_prueba, predicciones_prueba),
    ]


    guardar_resultados(resultados_evaluacion, "files/output/metrics.json")


if __name__ == "__main__":
    ejecutar_pipeline_completo()