# Desafío Académico: Asignación de Recursos para Entrenamiento de IA 🚀

**Proyecto — Métodos Numéricos**
Este repositorio contiene una aplicación web interactiva (Dashboard) desarrollada en Python con **Streamlit**. El objetivo del proyecto es simular y resolver un modelo matemático de asignación de recursos de hardware (Memoria RAM, Núcleos CPU y GPUs) utilizando métodos numéricos iterativos para sistemas de ecuaciones lineales ($3 \times 3$).

El proyecto incluye la implementación desde cero de los métodos de **Sobrerelajación Sucesiva (SOR)** y **Gradiente Conjugado Precondicionado (GCP)**, basados en la literatura académica de la asignatura (Suñagua, 2020).

---

## 🛠️ Tecnologías Utilizadas

* **Python 3.8+**
* **Streamlit:** Para la interfaz gráfica y el servidor web local.
* **NumPy:** Para el cálculo numérico y manejo de matrices algebraicas.
* **Plotly:** Para la visualización interactiva del error y la gráfica 3D de intersección de hiperplanos.
* **Pandas:** Para la estructuración de la tabla de resultados.

---

## ⚙️ Instrucciones de Instalación y Ejecución

Siga estos pasos para ejecutar el proyecto en su máquina local:

### Paso 1: Clonar o descargar el proyecto

Si descargó el proyecto en formato `.zip`, extráigalo en una carpeta de su preferencia. Si usa Git, puede clonarlo:

```bash
git clone <URL_DE_TU_REPOSITORIO_O_DEJAR_SOLO_LA_DESCARGA>
cd <NOMBRE_DE_LA_CARPETA>
```

### Paso 2: Instalar las dependencias

Asegúrese de tener Python instalado. Se recomienda abrir una terminal (Símbolo del sistema o PowerShell) en la ruta de la carpeta del proyecto y ejecutar el siguiente comando para instalar todas las librerías necesarias mediante el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar la aplicación

Una vez instaladas las dependencias, inicie el servidor de Streamlit ejecutando el siguiente comando en la misma terminal:

```bash
python -m streamlit run app.py
```

*(Nota: Se utiliza `python -m streamlit` para garantizar compatibilidad con las variables de entorno en Windows).*

El comando abrirá automáticamente una nueva pestaña en su navegador web predeterminado (típicamente en `http://localhost:8501`) con el Dashboard funcionando.

---

## 📖 Guía de Uso para la Evaluación

Para probar los distintos escenarios solicitados en la rúbrica del desafío, utilice la barra lateral izquierda del Dashboard:

1. **Escenarios Predefinidos:** Haga clic en los botones **"Ideal"**, **"Estrés"** o **"Mal C." (Mal Condicionado)**. Esto cargará automáticamente las matrices $A$ y los vectores $\mathbf{b}$ analizados en el informe escrito.
2. **Parámetros en Tiempo Real:** Puede modificar manualmente el parámetro $\omega$ (Omega para SOR), la tolerancia (por defecto $10^{-6}$) y el límite máximo de iteraciones.
3. **Visualización 2D:** Observe el gráfico central de líneas (escala logarítmica) que compara la velocidad de convergencia (caída del error) entre el método SOR y el Gradiente Conjugado.
4. **Visualización 3D Interactiva:** Desplácese hacia la parte inferior para interactuar con el gráfico tridimensional. Puede usar el clic izquierdo para **rotar**, la rueda del ratón para hacer **zoom** y visualizar exactamente el punto donde convergen los 3 hiperplanos de recursos.

---

## 📄 Archivos del Proyecto

* `app.py`: Código fuente principal de la aplicación web y los algoritmos numéricos.
* `requirements.txt`: Lista de dependencias y librerías de Python.
* `Informe_Desafio.pdf`: (Asegúrate de cambiar este nombre por el real) Informe matemático completo con el modelado, los cálculos de condicionamiento $\kappa$ y el análisis teórico.

---
**Desarrollado por:** CRISTHIAN PABLO ALVAREZ GUARACHI / CI. 13695631

**Asignatura:** Métodos Numéricos
