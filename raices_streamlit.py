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
# Barra lateral: menú y parámetros
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("☰ Menú")
    pagina = st.radio("Ir a:", ["📊 Métodos", "⚖️ Comparación"], label_visibility="collapsed")
 
    st.divider()
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
 
# ----------------------------------------------------------------------
# Cálculo (se usa en ambas páginas)
# ----------------------------------------------------------------------
filas_b, ok_b, msg_b = biseccion(a, b, tol_pct, int(max_iter))
filas_n, ok_n, msg_n = newton_raphson(x0, tol_pct, int(max_iter))
 
 
# ----------------------------------------------------------------------
# Página 1: Métodos
# ----------------------------------------------------------------------
def pagina_metodos():
    st.title("📈 Búsqueda de raíces: Bisección y Newton-Raphson")
    st.latex(r"f(x) = x^2 - 5x\,\sin(3x) + 3 = 0 \qquad f'(x) = 2x - 5\sin(3x) - 15x\cos(3x)")
 
    st.subheader("1. Gráfica de f(x) para elegir el punto inicial")
    xs = np.linspace(xmin, xmax, 2000)
    ys = f(xs)
 
    # Raíces aproximadas por cambio de signo (ayuda para elegir x0)
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
 
    st.info("Para ver la comparación detallada entre ambos métodos, ve a **⚖️ Comparación** en el menú lateral.")
 
 
# ----------------------------------------------------------------------
# Página 2: Comparación
# ----------------------------------------------------------------------
def pagina_comparacion():
    st.title("⚖️ Comparación: Bisección vs Newton-Raphson")
 
    if not (filas_b and filas_n and ok_b and ok_n):
        st.warning("Para comparar, ambos métodos deben converger. Revisa x₀, el intervalo [a, b] "
                   "y el máximo de iteraciones en la barra lateral.")
        if not ok_b:
            st.error(f"Bisección: {msg_b}")
        if not ok_n:
            st.error(f"Newton-Raphson: {msg_n}")
        return
 
    # --- Datos de cada método ---
    raiz_b = filas_b[-1]["c (raíz)"]
    raiz_n = filas_n[-1]["x_n+1"]
    it_b, it_n = len(filas_b), len(filas_n)
    ev_b = it_b + 2        # 1 evaluación de f por iteración + f(a) y f(b) iniciales
    ev_n = it_n * 2        # f(x) y f'(x) en cada iteración
    err_b = [r["Error rel. (%)"] for r in filas_b if r["Error rel. (%)"] is not None]
    err_n = [r["Error rel. (%)"] for r in filas_n]
    raiz_ref = raiz_n      # referencia para las gráficas
 
    # --- 1. Resultados lado a lado ---
    st.subheader("1. Resultados de ambos métodos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raíz por Bisección", f"{raiz_b:.8f}")
    c2.metric("Raíz por Newton-Raphson", f"{raiz_n:.8f}")
    c3.metric("Diferencia entre raíces", f"{abs(raiz_b - raiz_n):.2e}")
 
    resumen = pd.DataFrame({
        "Método": ["Bisección", "Newton-Raphson"],
        "Valor inicial": [f"[{a}, {b}]", f"x₀ = {x0}"],
        "Raíz": [raiz_b, raiz_n],
        "f(raíz)": [f(raiz_b), f(raiz_n)],
        "Iteraciones": [it_b, it_n],
        "Evaluaciones de función": [ev_b, ev_n],
        "Error final (%)": [err_b[-1] if err_b else 0.0, err_n[-1]],
    })
    st.dataframe(
        resumen,
        column_config={
            "Raíz": st.column_config.NumberColumn(format="%.8f"),
            "f(raíz)": st.column_config.NumberColumn(format="%.2e"),
            "Error final (%)": st.column_config.NumberColumn(format="%.2e"),
        },
        hide_index=True,
    )
    st.caption("Evaluaciones de función: Bisección = 1 por iteración + f(a) y f(b) iniciales; "
               "Newton-Raphson = 2 por iteración (f y f').")
 
    if abs(raiz_b - raiz_n) > 1e-3:
        st.warning("Los métodos convergieron a raíces distintas: la función tiene varias raíces. "
                   "Ajusta x₀ o el intervalo [a, b] para que apunten a la misma y la comparación sea justa.")
 
    # --- 2. Gráficas ---
    st.subheader("2. Gráficas de comparación")
    g1, g2 = st.columns(2)
 
    with g1:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.semilogy(range(2, len(err_b) + 2), err_b, marker="o", label="Bisección")
        ax1.semilogy(range(1, len(err_n) + 1), err_n, marker="s", label="Newton-Raphson")
        ax1.axhline(tol_pct, color="red", linestyle="--", label=f"Tolerancia {tol_pct}%")
        ax1.set_title("Error relativo por iteración")
        ax1.set_xlabel("Iteración")
        ax1.set_ylabel("Error relativo (%) – escala log")
        ax1.grid(alpha=0.3, which="both")
        ax1.legend()
        st.pyplot(fig1)
 
    with g2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        x_b = [r["c (raíz)"] for r in filas_b]
        x_n = [x0] + [r["x_n+1"] for r in filas_n]
        ax2.plot(range(1, len(x_b) + 1), x_b, marker="o", label="Bisección")
        ax2.plot(range(0, len(x_n)), x_n, marker="s", label="Newton-Raphson")
        ax2.axhline(raiz_ref, color="green", linestyle="--", label=f"Raíz ≈ {raiz_ref:.5f}")
        ax2.set_title("Aproximación a la raíz por iteración")
        ax2.set_xlabel("Iteración (0 = punto inicial de Newton)")
        ax2.set_ylabel("Valor de x")
        ax2.grid(alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
 
    fig3, ax3 = plt.subplots(figsize=(10, 3))
    posiciones = np.arange(2)
    ancho = 0.35
    ax3.bar(posiciones - ancho / 2, [it_b, it_n], ancho, label="Iteraciones")
    ax3.bar(posiciones + ancho / 2, [ev_b, ev_n], ancho, label="Evaluaciones de función")
    ax3.set_xticks(posiciones)
    ax3.set_xticklabels(["Bisección", "Newton-Raphson"])
    ax3.set_title("Costo computacional")
    ax3.grid(alpha=0.3, axis="y")
    ax3.legend()
    for barra in ax3.patches:
        ax3.annotate(f"{int(barra.get_height())}", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                     ha="center", va="bottom")
    st.pyplot(fig3)
 
    # --- 3. Explicación ---
    st.subheader("3. Explicación de la comparación")
    mas_eficiente = "Newton-Raphson" if it_n < it_b else "Bisección"
    ancho_final = (b - a) / 2**it_b
    errores_n_txt = " → ".join(f"{e:.2e}%" for e in err_n)
 
    st.markdown(f"""
**¿Qué hizo cada método?**
 
- **Bisección** encerró la raíz en el intervalo [{a}, {b}] y lo dividió a la mitad en cada iteración.
  Después de {it_b} iteraciones el intervalo quedó de ancho ≈ {ancho_final:.2e}. Su convergencia es
  **lineal**: en cada paso el error se reduce más o menos a la mitad, por eso necesita varias iteraciones.
- **Newton-Raphson** partió de x₀ = {x0} y en cada paso usó la recta tangente a f(x) para acercarse a la raíz.
  Sus errores relativos fueron: {errores_n_txt}. Su convergencia es **cuadrática**: cuando ya está cerca
  de la raíz, la cantidad de cifras correctas aproximadamente se duplica en cada iteración.
 
**¿Cuál fue más eficiente?**
 
**{mas_eficiente}** fue el más eficiente en este ejercicio: Newton-Raphson necesitó **{it_n} iteraciones**
({ev_n} evaluaciones de función) y Bisección **{it_b} iteraciones** ({ev_b} evaluaciones).
Ambos llegaron a la misma raíz, con una diferencia de {abs(raiz_b - raiz_n):.2e}, y cumplieron el
error relativo de {tol_pct}%.
 
**Ventajas y desventajas**
""")
 
    tabla_comp = pd.DataFrame({
        "Aspecto": ["Velocidad", "Necesita f'(x)", "Punto de partida", "Garantía de convergencia",
                    "Riesgo principal"],
        "Bisección": ["Lenta (lineal)", "No", "Intervalo [a, b] con cambio de signo",
                      "Sí, si f(a)·f(b) < 0", "Muchas iteraciones"],
        "Newton-Raphson": ["Rápida (cuadrática)", "Sí", "Un x₀ cercano a la raíz",
                           "No siempre", "Divergir o saltar a otra raíz; falla si f'(x) ≈ 0"],
    })
    st.dataframe(tabla_comp, hide_index=True)
 
    st.markdown("""
**Conclusión:** como f(x) = x² − 5x·sen(3x) + 3 oscila por el seno y tiene varias raíces, Newton-Raphson
es mucho más rápido **siempre que x₀ se elija bien con la gráfica**. Bisección es más lenta, pero es la
opción segura cuando se conoce un intervalo con cambio de signo. En la práctica se suelen combinar:
bisección para ubicar la raíz y Newton-Raphson para afinarla.
""")
 
 
# ----------------------------------------------------------------------
# Navegación
# ----------------------------------------------------------------------
if pagina == "📊 Métodos":
    pagina_metodos()
else:
    pagina_comparacion()
 
