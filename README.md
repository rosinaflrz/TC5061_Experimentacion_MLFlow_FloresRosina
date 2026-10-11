# Experimentación con MLflow — TC5061

## 1. Descripción del proyecto
En esta actividad se utiliza el modelo de Ridge Regression que desarrollé en la práctica anterior para predecir tiempos de espera en Walt Disney World usando el dataset `calendar_enriched.csv` que obtuve de Kaggle

El objetivo es realizar diferentes entrenamientos cambiando algunos parámetros del modelo y registrar los resultados en MLflow para poder compararlos y encontrar la configuración que tenga un mejor desempeño

Para esto el entrenamiento se pasó del notebook original a un archivo de Python llamado `train.py`, que permite ejecutar el modelo sin tener que modificar el código cada vez

## 2. Archivos del proyecto
- `train.py`: contiene el código para preparar los datos, entrenar el modelo, evaluarlo y guardar los resultados en MLflow
- `requirements.txt`: contiene las librerías y versiones necesarias para ejecutar el proyecto
- `README.md`: explica cómo ejecutar el programa y consultar los resultados.
- `.gitignore`: evita que se suban archivos temporales o innecesarios al repositorio

## 3. Cómo ejecutar el proyecto
Primero, hay que abrir una terminal dentro de la carpeta del proyecto.
**Paso 1. Crear un entorno virtual**
Esto permite instalar las librerías del proyecto sin afectar otras instalaciones de Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Paso 2. Instalar las librerías necesarias**
```bash
python -m pip install -r requirements.txt
```

**Paso 3. Ejecutar el modelo**
```bash
python train.py
```

La primera vez el programa descargará automáticamente el dataset desde Kaggle
También se puede usar un archivo CSV descargado previamente, indicando su ubicación mediante `--data-path`

## 4. Experimentos realizados
Se plantearon cinco configuraciones diferentes para analizar cómo afectan los parámetros al rendimiento del modelo

Para ejecutar cada experimento, se utilizan los siguientes comandos:
**Experimento 1: Regularización baja**
```bash
python train.py --alpha 0.1 --tol 0.001 --fit-intercept --seed 42
```

**Experimento 2: Configuración base**
```bash
python train.py --alpha 1.0 --tol 0.001 --fit-intercept --seed 42
```

**Experimento 3: Regularización alta**
```bash
python train.py --alpha 10.0 --tol 0.001 --fit-intercept --seed 42
```

**Experimento 4: Menor tolerancia**
```bash
python train.py --alpha 1.0 --tol 0.00001 --fit-intercept --seed 42
```

**Experimento 5: Modelo sin intercepto**
```bash
python train.py --alpha 1.0 --tol 0.001 --no-fit-intercept --seed 42
```

En todos los experimentos se usa la semilla 42 y la misma división de datos: 80% para entrenamiento y 20% para evaluación. Esto permite comparar los resultados bajo las mismas condiciones

Los parámetros que se modifican son:
- **Alpha:** controla qué tanta regularización se aplica al modelo.
- **Tol:** indica qué tan preciso debe ser el proceso de entrenamiento antes de detenerse.
- **Fit intercept:** determina si el modelo utiliza o no un término independiente.
- **Seed:** permite obtener resultados reproducibles al mantener la misma división de datos.

## 5. Cómo consultar los resultados en MLflow
Una vez ejecutados los experimentos, se abre otra terminal dentro de la carpeta del proyecto y se escribe:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
```

Después, se abre en el navegador:
            http://127.0.0.1:5000

Dentro de MLflow se selecciona el experimento `WDW_Ridge_Hyperparameter_Experiments`.

En esta sección se pueden consultar las diferentes ejecuciones, comparar sus parámetros y revisar las métricas obtenidas.

También se encuentran los archivos de evaluación y el modelo entrenado que se guardan durante cada ejecución.

## 6. Evaluación del modelo
Para evaluar los resultados se utilizan tres métricas:
- **MAE:** indica cuánto se equivocan las predicciones en promedio. Entre menor sea el valor, mejor.
- **RMSE:** también mide el error de las predicciones, pero penaliza más los errores grandes.
- **R²:** muestra qué tanto de la variación de los datos logra explicar el modelo. En general, un valor más alto indica un mejor ajuste.

Estas métricas permiten identificar qué configuración obtuvo mejores resultados.

## 7. Reproducibilidad
Para comprobar que los experimentos son reproducibles se puede ejecutar dos veces el mismo comando con la misma semilla

Los resultados deberían coincidir siempre que se utilicen el mismo dataset, las mismas versiones de las librerías y la misma configuración.

El programa también registra una huella digital del archivo de datos (`data_sha256`) para comprobar que se está utilizando la misma versión del dataset

## 8. Limitaciones
El modelo usa una división aleatoria de los datos en entrenamiento y prueba, al igual que en la actividad anterior. Sin embargo, como los datos están relacionados con fechas, una posible mejora sería utilizar una división temporal para evaluar qué tan bien predice el modelo sobre periodos posteriores

También es importante considerar que modificar los hiperparámetros no garantiza que el rendimiento mejore. Por eso se realizan varias pruebas y se comparan los resultados obtenidos en MLflow