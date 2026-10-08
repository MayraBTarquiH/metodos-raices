"""
Método de Bisección y Newton-Raphson
Problema: f(x) = x^2 - 5x*sen(3x) + 3 = 0
Criterio de parada: error relativo < 0.0005 %

Ejecutar con:  streamlit run raices_streamlit.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bisección y Newton-Raphson", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------
# Función y derivada
# ----------------------------------------------------------------------
def f(x):
    return x**2 - 5 * x * np.sin(3 * x) + 3


def df(x):
    return 2 * x - 5 * np.sin(3 * x) - 15 * x * np.cos(3 * x)


def error_relativo(nuevo, anterior):
    """Error relativo porcentual entre dos aproximaciones sucesivas."""
    if nuevo == 0:
        return abs(nuevo - anterior) * 100
    return abs((nuevo - anterior) / nuevo) * 100


# ----------------------------------------------------------------------
# Métodos numéricos
# ----------------------------------------------------------------------
def biseccion(a, b, tol_pct, max_iter):
    filas = []
    if f(a) * f(b) > 0:
        return filas, False, "f(a) y f(b) tienen el mismo signo: no hay cambio de signo en [a, b]."

    c_ant = None
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        err = None if c_ant is None else error_relativo(c, c_ant)
        filas.append({"Iter": i, "a": a, "b": b, "c (raíz)": c, "f(c)": fc, "Error rel. (%)": err})

        if fc == 0 or (err is not None and err < tol_pct):
            return filas, True, "Convergió."
        if f(a) * fc < 0:
            b = c
        else:
            a = c
        c_ant = c
    return filas, False, "Se alcanzó el máximo de iteraciones."


def newton_raphson(x0, tol_pct, max_iter):
    filas = []
    x = x0
    for i in range(1, max_iter + 1):
        d = df(x)
        if abs(d) < 1e-12:
            return filas, False, "La derivada es casi cero: elige otro punto inicial."
        x_nuevo = x - f(x) / d
        err = error_relativo(x_nuevo, x)
        filas.append({"Iter": i, "x_n": x, "f(x_n)": f(x), "f'(x_n)": d,
                      "x_n+1": x_nuevo, "Error rel. (%)": err})
        x = x_nuevo
        if err < tol_pct:
            return filas, True, "Convergió."
    return filas, False, "Se alcanzó el máximo de iteraciones."


def mostrar_tabla(filas):
    df_tabla = pd.DataFrame(filas)
    config = {
        col: st.column_config.NumberColumn(col, format="%.8f")
        for col in df_tabla.columns if col != "Iter"
    }
    st.dataframe(df_tabla, column_config=config, hide_index=True)


# ----------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------
st.title("📈 Búsqueda de raíces: Bisección y Newton-Raphson")
st.latex(r"f(x) = x^2 - 5x\,\sin(3x) + 3 = 0 \qquad f'(x) = 2x - 5\sin(3x) - 15x\cos(3x)")

with st.sidebar:
    st.header("Parámetros")
    x0 = st.number_input("Punto inicial x₀ (Newton-Raphson)", value=2.4, step=0.05, format="%.4f",
                         help="Elígelo viendo la gráfica, cerca de la raíz que buscas.")
    st.subheader("Intervalo para Bisección")
    a = st.number_input("a", value=2.0, step=0.1, format="%.4f")
    b = st.number_input("b", value=2.5, step=0.1, format="%.4f")
    st.subheader("Criterio de parada")
    tol_pct = st.number_input("Error relativo (%)", value=0.0005, min_value=0.0, step=0.0005, format="%.6f")
    max_iter = st.number_input("Máximo de iteraciones", value=100, min_value=1, step=10)
    st.subheader("Rango de la gráfica")
    xmin, xmax = st.slider("x", -6.0, 6.0, (0.0, 5.0), step=0.1)

# ---------------------------- Gráfica ---------------------------------
st.subheader("1. Gráfica de f(x) para elegir el punto inicial")

xs = np.linspace(xmin, xmax, 2000)
ys = f(xs)

# Aproximación de raíces por cambio de signo (ayuda para elegir x0)
cambios = np.where(np.sign(ys[:-1]) * np.sign(ys[1:]) < 0)[0]
aprox = [(xs[i] + xs[i + 1]) / 2 for i in cambios]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xs, ys, label="f(x)")
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(x0, color="red", linestyle="--", label=f"x₀ = {x0:.4f}")
ax.scatter([x0], [f(x0)], color="red", zorder=5)
if a < b:
    ax.axvspan(a, b, color="orange", alpha=0.15, label=f"Intervalo [{a:.2f}, {b:.2f}]")
if aprox:
    ax.scatter(aprox, [0] * len(aprox), color="green", zorder=5, label="Raíces aprox.")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(alpha=0.3)
ax.legend()
st.pyplot(fig)

if aprox:
    st.caption("Raíces aproximadas en el rango graficado: " + ", ".join(f"{r:.3f}" for r in aprox))
else:
    st.caption("No se detectan cambios de signo en el rango graficado.")

# ---------------------------- Cálculo ---------------------------------
filas_b, ok_b, msg_b = biseccion(a, b, tol_pct, int(max_iter))
filas_n, ok_n, msg_n = newton_raphson(x0, tol_pct, int(max_iter))

st.subheader("2. Resultados")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Bisección")
    if filas_b:
        raiz_b = filas_b[-1]["c (raíz)"]
        st.metric("Raíz aproximada", f"{raiz_b:.8f}")
        st.write(f"**Iteraciones:** {len(filas_b)}  |  **f(raíz):** {f(raiz_b):.2e}")
        mostrar_tabla(filas_b)
    (st.success if ok_b else st.error)(msg_b)

with col2:
    st.markdown("### Newton-Raphson")
    if filas_n:
        raiz_n = filas_n[-1]["x_n+1"]
        st.metric("Raíz aproximada", f"{raiz_n:.8f}")
        st.write(f"**Iteraciones:** {len(filas_n)}  |  **f(raíz):** {f(raiz_n):.2e}")
        mostrar_tabla(filas_n)
    (st.success if ok_n else st.error)(msg_n)

# ---------------------------- Comparación -----------------------------
st.subheader("3. Comparación de eficiencia")

if filas_b and filas_n and ok_b and ok_n:
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    err_b = [r["Error rel. (%)"] for r in filas_b if r["Error rel. (%)"] is not None]
    err_n = [r["Error rel. (%)"] for r in filas_n]
    ax2.semilogy(range(2, len(err_b) + 2), err_b, marker="o", label="Bisección")
    ax2.semilogy(range(1, len(err_n) + 1), err_n, marker="s", label="Newton-Raphson")
    ax2.axhline(tol_pct, color="red", linestyle="--", label=f"Tolerancia {tol_pct}%")
    ax2.set_xlabel("Iteración")
    ax2.set_ylabel("Error relativo (%)")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    st.pyplot(fig2)

    resumen = pd.DataFrame({
        "Método": ["Bisección", "Newton-Raphson"],
        "Iteraciones": [len(filas_b), len(filas_n)],
        "Raíz": [filas_b[-1]["c (raíz)"], filas_n[-1]["x_n+1"]],
    })
    st.dataframe(resumen, column_config={"Raíz": st.column_config.NumberColumn(format="%.8f")},
                 hide_index=True)

    if abs(filas_b[-1]["c (raíz)"] - filas_n[-1]["x_n+1"]) > 1e-3:
        st.warning("Los métodos convergieron a raíces distintas: la función tiene varias raíces. "
                   "Ajusta x₀ o el intervalo [a, b] para que apunten a la misma.")

    mas_eficiente = "Newton-Raphson" if len(filas_n) < len(filas_b) else "Bisección"
    st.info(
        f"**{mas_eficiente}** fue más eficiente: Newton-Raphson necesitó {len(filas_n)} iteraciones "
        f"y Bisección {len(filas_b)}. Newton-Raphson converge de forma cuadrática, pero necesita f'(x) "
        "y un buen x₀; Bisección siempre converge si hay cambio de signo, aunque es más lenta."
    )
else:
    st.info("Corrige los parámetros para que ambos métodos converjan y poder compararlos.")
