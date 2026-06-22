# Actividad5-Customer-Churn-MLflow_Gustavo-_Marin

## Descripción del proyecto

Este proyecto tiene como objetivo entrenar y comparar modelos de aprendizaje automático para predecir el abandono de clientes, utilizando el dataset Telco Customer Churn.

El caso de estudio fue seleccionado porque representa un problema común en las organizaciones: identificar qué clientes tienen mayor probabilidad de abandonar un servicio. Esto permite tomar decisiones preventivas y diseñar estrategias de retención.

## Modelos utilizados

- Regresión Logística
- Random Forest

## Herramientas

- Python
- Google Colab
- Scikit-learn
- Pandas
- MLflow
- GitHub

## Estructura del repositorio

- datos/datos_ini: dataset original.
- datos/datos_limp: dataset limpio.
- fuentes/entrena.ipynb: notebook principal.
- fuentes/datos_prep.py: funciones de limpieza.
- fuentes/train.py: entrenamiento y registro en MLflow.
- README.md: descripción del proyecto.
- CHANGELOG.md: registro de cambios.

## Métricas evaluadas

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Conclusión

El proyecto permite comparar dos modelos de clasificación y registrar los experimentos mediante MLflow, asegurando trazabilidad, reproducibilidad y una mejor toma de decisiones basada en resultados.
