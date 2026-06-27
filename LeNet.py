import streamlit as st

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="LeNet-5 (1998) - Análisis Estructural",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --- FUNCIÓN VISUALIZADORA DE TENSORES EN 3D ---
# Esta función dibuja cajas superpuestas para simular las dimensiones de la matriz de datos
def dibujar_tensor(alto, ancho, profundidad, tipo="3d", color="#1f77b4"):
    html = ""
    escala = 5  # Factor de multiplicación para que se vea bien en pantalla

    if tipo == "3d":
        h, w = alto * escala, ancho * escala
        desplazamiento_max = profundidad * 3
        html += f'<div style="position:relative; width:{w + desplazamiento_max}px; height:{h + desplazamiento_max}px; margin: 30px auto;">'

        # Dibujar cada "capa" o mapa de características
        for i in range(profundidad):
            offset = i * 3
            opacidad = 0.1 if i < profundidad - 1 else 0.4
            html += f'''
            <div style="position:absolute; top:{offset}px; left:{offset}px; 
                        width:{w}px; height:{h}px; 
                        border: 1.5px solid {color}; 
                        background-color: {color}{int(opacidad * 255):02x}; 
                        box-shadow: -1px -1px 3px rgba(0,0,0,0.1);">
            </div>'''
        html += '</div>'

    elif tipo == "1d":
        # Dibuja un vector (fila de datos aplastada)
        largo = min(alto, 100) * 4  # Límite visual para no salir de pantalla
        html += f'<div style="margin: 40px auto; display: flex; align-items: center; justify-content: center;">'
        html += f'<div style="width:{largo}px; height:25px; border: 2px solid {color}; background-color: {color}40;"></div>'
        html += f'<span style="margin-left: 15px; font-weight: bold; color: #555;">Vector de {alto} neuronas</span>'
        html += '</div>'

    return html


# 2. ENCABEZADO ACADÉMICO
st.title("Análisis Estructural de LeNet-5")
st.markdown("**Basado en: *Gradient-Based Learning Applied to Document Recognition* (LeCun et al., 1998)**")
st.markdown(
    "Simulador visual: Transición de la extracción heurística a la representación matemática automática (End-to-End).")
st.divider()

# 3. ORGANIZACIÓN EN PESTAÑAS (Solo las teóricas y visuales)
tab_arquitectura, tab_eficiencia = st.tabs([
    "Simulador: Flujo de Tensores 3D",
    "Justificación de Eficiencia"
])

# --- PESTAÑA 1: SIMULADOR VISUAL PASO A PASO ---
with tab_arquitectura:
    st.header("Transformación Espacial de la Información")
    st.markdown(
        "Seleccione una fase matemática para observar cómo la red altera la topología de los datos (Resolución vs. Profundidad).")

    # Menú interactivo paso a paso
    capas_lenet = [
        "0. Entrada (Input)",
        "1. C1 - Capa Convolucional",
        "2. S2 - Submuestreo (Pooling)",
        "3. C3 - Convolución Profunda",
        "4. S4 - Submuestreo Final",
        "5. Aplanamiento (Flatten)",
        "6. C5 / F6 - Red Neuronal Densa",
        "7. Salida (Clasificador RBF)"
    ]

    seleccion = st.selectbox("Seleccione la fase de procesamiento:", capas_lenet)

    # Columnas para Teoría y Visualización 3D
    col_teoria, col_visual = st.columns([1, 1])

    if "0. Entrada" in seleccion:
        with col_teoria:
            st.subheader("Ingesta de Datos Normalizados")
            st.write(
                "La red recibe una imagen en escala de grises. Se observa un único mapa de características (profundidad = 1).")
            st.metric("Dimensión Matemática", "32 x 32 x 1")
            st.metric("Parámetros Entrenables", "0")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Representación del Tensor</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(32, 32, 1, color="#333333"), unsafe_allow_html=True)

    elif "1. C1" in seleccion:
        with col_teoria:
            st.subheader("C1: Extracción de Características Locales")
            st.write(
                "Aplicación de **Campos Receptivos Locales**. La dimensión espacial se reduce (de 32 a 28), pero la profundidad aumenta al aplicar 6 filtros distintos.")
            st.metric("Dimensión Matemática", "28 x 28 x 6")
            st.metric("Parámetros Entrenables", "156")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Resolución decrece, Profundidad crece</h4>",
                        unsafe_allow_html=True)
            st.markdown(dibujar_tensor(28, 28, 6), unsafe_allow_html=True)

    elif "2. S2" in seleccion:
        with col_teoria:
            st.subheader("S2: Invariancia Espacial")
            st.write(
                "Aplicación de **Submuestreo (Average Pooling)**. La resolución se reduce exactamente a la mitad (promediando bloques de 2x2). Los mapas mantienen su profundidad.")
            st.metric("Dimensión Matemática", "14 x 14 x 6")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>El tensor se comprime a la mitad</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(14, 14, 6, color="#2ca02c"), unsafe_allow_html=True)

    elif "3. C3" in seleccion:
        with col_teoria:
            st.subheader("C3: Combinación de Rasgos")
            st.write(
                "Se aplican 16 nuevos filtros combinando los mapas anteriores. Los bordes simples se cruzan para formar figuras cerradas.")
            st.metric("Dimensión Matemática", "10 x 10 x 16")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Mayor abstracción, más profundidad</h4>",
                        unsafe_allow_html=True)
            st.markdown(dibujar_tensor(10, 10, 16), unsafe_allow_html=True)

    elif "4. S4" in seleccion:
        with col_teoria:
            st.subheader("S4: Abstracción Visual Final")
            st.write(
                "Última capa de topología espacial. Se vuelve a encoger la imagen a la mitad. Quedan matrices minúsculas pero con un altísimo nivel de información abstracta.")
            st.metric("Dimensión Matemática", "5 x 5 x 16")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Esencia visual pura</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(5, 5, 16, color="#2ca02c"), unsafe_allow_html=True)

    elif "5. Aplanamiento" in seleccion:
        with col_teoria:
            st.subheader("Transición Topológica (Flatten)")
            st.write(
                "Se abandona el espacio 3D. Las matrices bidimensionales (16 mapas de 5x5) se 'aplastan' matemáticamente en un vector lineal.")
            st.metric("Dimensión Matemática", "Vector 1D (400 neuronas)")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Colapso de las dimensiones</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(400, 0, 0, tipo="1d", color="#ff7f0e"), unsafe_allow_html=True)

    elif "6. C5 / F6" in seleccion:
        with col_teoria:
            st.subheader("Red Neuronal de Clasificación")
            st.write(
                "El vector de 400 neuronas pasa por las capas de razonamiento lógico. Primero se reduce a 120, y luego a 84 (representando un mapa de bits lógico).")
            st.metric("Dimensión Matemática", "Vector 1D (84 neuronas)")
            st.metric("Parámetros Combinados", "58,244")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Razonamiento Lógico</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(84, 0, 0, tipo="1d", color="#ff7f0e"), unsafe_allow_html=True)

    elif "7. Salida" in seleccion:
        with col_teoria:
            st.subheader("Decisión Euclidiana")
            st.write(
                "La red emite 10 valores, calculando la distancia euclidiana frente a plantillas ideales (RBF). El valor más bajo dicta qué dígito se identificó.")
            st.metric("Dimensión Matemática", "Vector 1D (10 neuronas)")
            st.metric("Predicción", "Dígitos del 0 al 9")
        with col_visual:
            st.markdown("<h4 style='text-align: center;'>Salida Final</h4>", unsafe_allow_html=True)
            st.markdown(dibujar_tensor(10, 0, 0, tipo="1d", color="#d62728"), unsafe_allow_html=True)

# --- PESTAÑA 2: EFICIENCIA PARAMÉTRICA ---
with tab_eficiencia:
    st.header("Justificación Estructural de las CNN")
    st.markdown(
        "¿Por qué el paper de LeCun descartó el uso de redes tradicionales (Fully Connected) para analizar imágenes?")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("LeNet-5 (Pesos Compartidos)")
        st.write(
            "Al obligar a la red a utilizar el mismo filtro (kernel) para recorrer toda la imagen, se evita la necesidad de asignar un peso individual a cada píxel, logrando invariancia espacial.")
        st.metric("Total de Parámetros", "~61,706")
        st.success(
            "✅ Eficiente computacionalmente, previene el sobreajuste y generaliza formas geométricas independientemente de su posición.")

    with col_b:
        st.subheader("Red Tradicional (Fully Connected)")
        st.write(
            "Si se conectara la imagen de 32x32 píxeles directamente a una red densa con capacidades de aprendizaje equivalentes, la explosión combinatoria arruinaría el proceso.")
        st.metric("Total de Parámetros Estimados", "> 1,200,000")
        st.error(
            "❌ Memoriza los datos ciegamente (Overfitting), requiere demasiada memoria y falla ante la mínima rotación o traslado del dígito.")