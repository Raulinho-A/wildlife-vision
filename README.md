# Sistema de Visión Artificial para Detección y Clasificación de Fauna Silvestre

Este proyecto desarrolla un sistema de visión computacional y aprendizaje profundo para automatizar el análisis de datos obtenidos mediante cámaras trampa usadas en el monitoreo de fauna silvestre.

## Objetivo

Reducir el tiempo y esfuerzo del procesamiento manual de imágenes mediante:

- Detección automática de animales en imágenes.
- Clasificación de especies presentes.
- Generación de métricas útiles para tareas de conservación.

El sistema aborda desafíos como:

- Imágenes de baja calidad y ruido ambiental.
- Desbalance significativo entre clases de especies.
- Datos incompletos o parcialmente anotados.

## Estado actual

Actualmente nos encontramos en fase experimental, trabajando en:

- Preparación del dataset (recortes, balanceo, augmentations).
- Entrenamiento de modelos CNN base para establecer líneas comparativas.
- Exploración de técnicas de transfer learning para mejorar la robustez.
- Evaluación y comparación de métricas avanzadas más allá del accuracy.

## Estructura del proyecto

- Código fuente: `vision/`
- Datos: `data/`
- Modelos guardados: `models/`
- Notebooks de experimentación: `notebooks/`
- Reportes y visualizaciones: `reports/`

## Instalación del entorno

1. Clonar este repositorio.
2. Crear y activar un entorno virtual.
3. Instalar las dependencias principales:
   ```bash
   pip install .
   ```
4. (Opcional) Instalar dependencias de desarrollo:
   ```bash
   pip install .[dev]
   ```
