import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. LÓGICA MATEMÁTICA (MÉTODOS NUMÉRICOS)
# ==========================================

def solve_sor(A, b, omega, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    errors = []
    
    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (1 - omega) * x[i] + (omega / A[i, i]) * (b[i] - sigma)
        
        error = np.linalg.norm(x - x_old, ord=np.inf)
        errors.append(error)
        if error < tol:
            return x, k + 1, errors
    return x, max_iter, errors

def solve_cg(A, b, tol, max_iter):
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x)
    p = r.copy()
    errors = []
    
    for k in range(max_iter):
        Ap = np.dot(A, p)
        # Evitar división por cero
        denom = np.dot(p, Ap)
        if denom == 0: break
            
        alpha = np.dot(r, r) / denom
        x = x + alpha * p
        r_new = r - alpha * Ap
        
        error = np.linalg.norm(r_new, ord=np.inf)
        errors.append(error)
        
        if error < tol:
            return x, k + 1, errors
        
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        
    return x, max_iter, errors

# ==========================================
# 2. CONFIGURACIÓN DE LA INTERFAZ Y ESTADO
# ==========================================

st.set_page_config(page_title="IA Resource Allocator - Dashboard", layout="wide")

# Función para actualizar las variables directamente en la memoria de los inputs
def set_scenario(escenario):
    if escenario == 'ideal':
        A_new = [[10., 2., 1.], [2., 8., 1.], [1., 1., 6.]]
        b_new = [90., 50., 24.]
    elif escenario == 'estres':
        A_new = [[500., 200., 150.], [200., 800., 300.], [150., 300., 1000.]]
        b_new = [7500., 10200., 9900.]
    elif escenario == 'mal_c':
        A_new = [[1., 0.95, 0.], [0.95, 1., 0.], [0., 0., 4.]]
        b_new = [1.95, 1.95, 4.]
        
    for i in range(3):
        for j in range(3):
            st.session_state[f"a{i}{j}"] = float(A_new[i][j])
        st.session_state[f"b{i}"] = float(b_new[i])

# Inicializar los valores la primera vez que se abre la app
if "a00" not in st.session_state:
    set_scenario('ideal')

st.title("🚀 Dashboard: Asignación de Recursos para Entrenamiento de IA")
st.markdown("Simulación de asignación de **RAM ($x_1$)**, **CPU ($x_2$)** y **GPUs ($x_3$)** mediante Métodos Numéricos.")

# --- BARRA LATERAL (CONFIGURACIÓN) ---
st.sidebar.header("⚙️ Configuración del Sistema")

# Botones de Carga Rápida usando callbacks
st.sidebar.subheader("Escenarios Predefinidos")
col_esc1, col_esc2, col_esc3 = st.sidebar.columns(3)

col_esc1.button("Ideal", on_click=set_scenario, args=('ideal',))
col_esc2.button("Estrés", on_click=set_scenario, args=('estres',))
col_esc3.button("Mal C.", on_click=set_scenario, args=('mal_c',))

# Inputs de Matriz A y Vector B 
st.sidebar.subheader("Matriz de Coeficientes (A)")
a_cols = st.sidebar.columns(3)
new_A = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        new_A[i,j] = a_cols[j].number_input(f"A[{i+1},{j+1}]", key=f"a{i}{j}", step=1.0)

st.sidebar.subheader("Vector de Requerimientos (b)")
b_cols = st.sidebar.columns(3)
new_b = np.zeros(3)
for i in range(3):
    new_b[i] = b_cols[i].number_input(f"b[{i+1}]", key=f"b{i}", step=1.0)

# Parámetros del método
st.sidebar.subheader("Parámetros de Convergencia")
omega = st.sidebar.slider("Parámetro Omega (SOR)", 0.1, 1.9, 1.05, 0.01)
tol = st.sidebar.number_input("Tolerancia", value=1e-6, format="%.1e")
max_iters = st.sidebar.number_input("Iteraciones Máximas", value=500)

# ==========================================
# 3. CÁLCULOS Y VISUALIZACIÓN SUPERIOR
# ==========================================

col_main, col_res = st.columns([2, 1])

# Cálculos Numéricos
sol_sor, iter_sor, errs_sor = solve_sor(new_A, new_b, omega, tol, max_iters)
sol_cg, iter_cg, errs_cg = solve_cg(new_A, new_b, tol, max_iters)

with col_main:
    st.subheader("📊 Análisis de Convergencia en Tiempo Real")
    
    # Gráfico de convergencia 2D con Plotly
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(y=errs_sor, mode='lines+markers', name='SOR', line=dict(color='firebrick')))
    fig_conv.add_trace(go.Scatter(y=errs_cg, mode='lines+markers', name='Gradiente Conjugado', line=dict(color='royalblue')))
    
    fig_conv.update_layout(
        xaxis_title="Iteraciones",
        yaxis_title="Error (Norma Infinito)",
        yaxis_type="log",
        template="plotly_white",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_conv, use_container_width=True)

with col_res:
    st.subheader("🎯 Resultados Finales")
    
    # Tabla comparativa rápida
    res_df = pd.DataFrame({
        "Métrica": ["Iteraciones", "RAM (x1)", "CPU (x2)", "GPU (x3)"],
        "SOR": [iter_sor, f"{sol_sor[0]:.4f}", f"{sol_sor[1]:.4f}", f"{sol_sor[2]:.4f}"],
        "G. Conjugado": [iter_cg, f"{sol_cg[0]:.4f}", f"{sol_cg[1]:.4f}", f"{sol_cg[2]:.4f}"]
    })
    st.table(res_df)
    
    # Estado del sistema (Determinante y Condición)
    try:
        kappa = np.linalg.cond(new_A, p=np.inf)
        st.info(f"**Número de Condición ($\kappa_\infty$):** {kappa:.2f}")
        
        if kappa > 50:
            st.warning("⚠️ Sistema Mal Condicionado: La convergencia de métodos iterativos clásicos puede ser lenta.")
        else:
            st.success("✅ Sistema Bien Condicionado.")
    except np.linalg.LinAlgError:
        st.error("❌ Matriz Singular: El sistema no tiene solución única.")

# ==========================================
# 4. VISUALIZACIÓN 3D INTERACTIVA
# ==========================================
st.markdown("---")
st.subheader("🌐 Visualización 3D Interactiva del Sistema (Intersección de Planos)")
st.write("Puedes rotar el gráfico con el clic izquierdo, hacer zoom con la rueda del ratón y ver los valores exactos pasando el cursor por encima.")

# Extraer el punto exacto de la solución (usando GC como referencia)
x_val = sol_cg[0] if not np.isnan(sol_cg[0]) and not np.isinf(sol_cg[0]) else 8
y_val = sol_cg[1] if not np.isnan(sol_cg[1]) and not np.isinf(sol_cg[1]) else 4
z_val = sol_cg[2] if not np.isnan(sol_cg[2]) and not np.isinf(sol_cg[2]) else 2

# Definir la malla (grid) centrada alrededor de la solución encontrada
x_range = np.linspace(x_val - 5, x_val + 5, 20)
y_range = np.linspace(y_val - 5, y_val + 5, 20)
X1, X2 = np.meshgrid(x_range, y_range)

fig3d = go.Figure()

colores_planos = ['Blues', 'Reds', 'Greens']
nombres_planos = ['Balance Memoria (RAM)', 'Cómputo FLOPS (CPU)', 'Transferencia (PCIe)']

# Generar los 3 planos despejando Z (x3) de cada ecuación
for i in range(3):
    # Asegurarnos de no dividir por cero si el coeficiente z es 0
    if new_A[i, 2] != 0:
        Z = (new_b[i] - new_A[i, 0]*X1 - new_A[i, 1]*X2) / new_A[i, 2]
        fig3d.add_trace(go.Surface(x=X1, y=X2, z=Z, 
                                   name=nombres_planos[i], 
                                   colorscale=colores_planos[i], 
                                   opacity=0.6, 
                                   showscale=False))

# Dibujar la solución exacta como un punto negro flotando
fig3d.add_trace(go.Scatter3d(
    x=[x_val], y=[y_val], z=[z_val],
    mode='markers',
    marker=dict(size=8, color='black', symbol='circle', line=dict(color='white', width=2)),
    name='Solución Encontrada'
))

fig3d.update_layout(
    scene=dict(
        xaxis_title='RAM (x1)',
        yaxis_title='CPU (x2)',
        zaxis_title='GPUs (x3)',
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.5)) # Ángulo inicial de la cámara
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=600
)

st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")
st.caption("Desarrollado para el Desafío Académico de Métodos Numéricos.")