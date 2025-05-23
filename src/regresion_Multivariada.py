import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import time
import gc
import psutil
import os
import sys

if sys.platform.startswith('win'):
    import locale
    locale.setlocale(locale.LC_ALL, 'C')


def obtener_uso_memoria():
    proceso = psutil.Process(os.getpid())
    return proceso.memory_info().rss / 1024 / 1024


def cargar_datos(ruta_entrenamiento, ruta_prueba, max_caracteristicas=None):
    print(f"Memoria inicial: {obtener_uso_memoria():.1f} MB")

    datos_entrenamiento = pd.read_parquet(ruta_entrenamiento)
    datos_prueba = pd.read_parquet(ruta_prueba)

    print(f"Forma del dataset de entrenamiento: {datos_entrenamiento.shape}")
    print(f"Forma del dataset de prueba: {datos_prueba.shape}")
    print(f"Memoria después de cargar: {obtener_uso_memoria():.1f} MB")

    X_entrenamiento = datos_entrenamiento.drop('label', axis=1)
    y_entrenamiento = datos_entrenamiento['label']
    X_prueba = datos_prueba.drop('label', axis=1)
    y_prueba = datos_prueba['label']

    del datos_entrenamiento, datos_prueba
    gc.collect()

    if max_caracteristicas and X_entrenamiento.shape[1] > max_caracteristicas:
        print(f"\nReduciendo características de {X_entrenamiento.shape[1]} a {max_caracteristicas}...")
        selector = SelectKBest(score_func=f_regression, k=max_caracteristicas)
        X_entrenamiento = selector.fit_transform(X_entrenamiento, y_entrenamiento)
        X_prueba = selector.transform(X_prueba)
        print(f"Características seleccionadas: {X_entrenamiento.shape[1]}")

    X_entrenamiento = X_entrenamiento.astype(np.float32)
    X_prueba = X_prueba.astype(np.float32)
    y_entrenamiento = y_entrenamiento.astype(np.float32)
    y_prueba = y_prueba.astype(np.float32)

    print(f"Memoria después de preprocesamiento: {obtener_uso_memoria():.1f} MB")

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba


def configurar_busqueda_parametros(X_entrenamiento, y_entrenamiento):
    modelo = LinearRegression()
    grilla_parametros = {
        'fit_intercept': [True, False],  
        'positive': [False] 
    }
    busqueda_parametros = GridSearchCV(
        estimator=modelo,
        param_grid=grilla_parametros,
        cv=3,
        scoring='r2',
        n_jobs=1,
        verbose=1,
        pre_dispatch=1
    )
    return busqueda_parametros


def entrenar_regresion_multivariada(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    print(" EXPERIMENTO: REGRESIÓN MULTIVARIADA \n")

    tiempo_inicio = time.time()
    print(f"Memoria antes del entrenamiento: {obtener_uso_memoria():.1f} MB")

    print("1. Buscando mejores hiperparámetros...")

    busqueda = configurar_busqueda_parametros(X_entrenamiento, y_entrenamiento)
    busqueda.fit(X_entrenamiento, y_entrenamiento)
    tiempo_optimizacion = time.time()

    mejor_modelo = busqueda.best_estimator_
    mejores_parametros = busqueda.best_params_
    mejor_puntuacion = busqueda.best_score_

    print(f"\n2. Mejores hiperparámetros encontrados:")
    for parametro, valor in mejores_parametros.items():
        print(f"   {parametro}: {valor}")

    print(f"3. Mejor puntuación R² en validación cruzada: {mejor_puntuacion:.4f}")
    print(f"Memoria después de optimización: {obtener_uso_memoria():.1f} MB")

    print("\n4. Evaluando en datos de entrenamiento...")
    predicciones_entrenamiento = mejor_modelo.predict(X_entrenamiento)
    r2_entrenamiento = r2_score(y_entrenamiento, predicciones_entrenamiento)
    mae_entrenamiento = mean_absolute_error(y_entrenamiento, predicciones_entrenamiento)
    mse_entrenamiento = mean_squared_error(y_entrenamiento, predicciones_entrenamiento)

    del predicciones_entrenamiento
    gc.collect()

    print("5. Evaluando en datos de prueba...")
    predicciones_prueba = mejor_modelo.predict(X_prueba)
    r2_prueba = r2_score(y_prueba, predicciones_prueba)
    mae_prueba = mean_absolute_error(y_prueba, predicciones_prueba)
    mse_prueba = mean_squared_error(y_prueba, predicciones_prueba)

    tiempo_final = time.time()

    mostrar_resultados(
        tiempo_optimizacion - tiempo_inicio,
        tiempo_final - tiempo_inicio,
        r2_entrenamiento, mae_entrenamiento, mse_entrenamiento,
        r2_prueba, mae_prueba, mse_prueba,
        X_entrenamiento.shape[1],
        mejor_modelo
    )

    return mejor_modelo


def mostrar_resultados(tiempo_optimizacion, tiempo_total, 
                      r2_entrenamiento, mae_entrenamiento, mse_entrenamiento,
                      r2_prueba, mae_prueba, mse_prueba,
                      num_caracteristicas, modelo):

    print(f"\n RESULTADOS FINALES ")
    print(f"Tiempo de optimización: {tiempo_optimizacion:.2f} segundos")
    print(f"Tiempo total: {tiempo_total:.2f} segundos")
    print(f"Memoria final: {obtener_uso_memoria():.1f} MB")

    print(f"\nRENDIMIENTO EN ENTRENAMIENTO:")
    print(f"! R² Score: {r2_entrenamiento:.4f}")
    print(f"! MAE: {mae_entrenamiento:.4f}")
    print(f"! MSE: {mse_entrenamiento:.4f}")
    print(f"! RMSE: {np.sqrt(mse_entrenamiento):.4f}")

    print(f"\nRENDIMIENTO EN PRUEBA:")
    print(f"! R² Score: {r2_prueba:.4f}")
    print(f"! MAE: {mae_prueba:.4f}")
    print(f"! MSE: {mse_prueba:.4f}")
    print(f"! RMSE: {np.sqrt(mse_prueba):.4f}")

    diferencia_r2 = abs(r2_entrenamiento - r2_prueba)
    print(f"\nANÁLISIS DEL MODELO:")
    print(f"! Diferencia R² (entrenamiento-prueba): {diferencia_r2:.4f}")
    if diferencia_r2 < 0.05:
        print("! [OK] Modelo bien balanceado")
    elif r2_entrenamiento > r2_prueba + 0.05:
        print("! [ADVERTENCIA] Posible sobreajuste")
    else:
        print("! [ADVERTENCIA] Posible subajuste")

    print(f"\nINFORMACIÓN DEL MODELO:")
    print(f"! Número de características: {num_caracteristicas:,}")
    print(f"! Intercepto: {modelo.intercept_:.4f}")
    if hasattr(modelo, 'coef_'):
        print(f"! Rango de coeficientes: [{np.min(modelo.coef_):.4f}, {np.max(modelo.coef_):.4f}]")


def ejecutar_experimento():
    ruta_entrenamiento = "src/data/train_data_clean.parquet"
    ruta_prueba = "src/data/test_data_clean.parquet"
    MAX_CARACTERISTICAS = 2000

    print(" CARGA Y PREPROCESAMIENTO DE DATOS ")
    print(f"RAM disponible: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = cargar_datos(
        ruta_entrenamiento, ruta_prueba, max_caracteristicas=MAX_CARACTERISTICAS
    )

    print(f"\nDimensiones finales:")
    print(f"! Entrenamiento: {X_entrenamiento.shape}")
    print(f"! Prueba: {X_prueba.shape}")

    mejor_modelo = entrenar_regresion_multivariada(
        X_entrenamiento, X_prueba, y_entrenamiento, y_prueba
    )

    return mejor_modelo


if __name__ == "__main__":
    modelo_final = ejecutar_experimento()
