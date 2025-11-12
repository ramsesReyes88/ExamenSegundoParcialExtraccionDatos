"""
============================================================================
APLICACIÓN WEB PARA PREDICCIÓN DE CALIDAD DEL AGUA
Tecnología: Streamlit
Autor: [Tu Nombre]
Fecha: 2024
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


st.set_page_config(
    page_title="Predicción de Calidad del Agua",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)



st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px 24px;
        border-radius: 8px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .potable {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .no-potable {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
    }
    .info-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

@st.cache_resource
def load_model_and_preprocessors():
    """Carga el modelo y los preprocesadores"""
    try:
        model = joblib.load('water_quality_model.joblib')
        scaler = joblib.load('scaler.joblib')
        imputer = joblib.load('imputer.joblib')
        feature_names = joblib.load('feature_names.joblib')
        
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        
        return model, scaler, imputer, feature_names, config
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.stop()

# Cargar modelo y configuración
model, scaler, imputer, feature_names, config = load_model_and_preprocessors()

# ============================================================================
# FUNCIONES DE PREDICCIÓN
# ============================================================================

def make_prediction(input_data):
    """Realiza la predicción con el modelo cargado"""
    try:
        # Convertir a array numpy
        input_array = np.array(input_data).reshape(1, -1)
        
        # Aplicar preprocesamiento
        input_imputed = imputer.transform(input_array)
        input_scaled = scaler.transform(input_imputed)
        
        # Hacer predicción
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        return prediction, probability
    except Exception as e:
        st.error(f"❌ Error en la predicción: {str(e)}")
        return None, None

def get_feature_info(feature_name):
    """Obtiene información sobre los rangos de una característica"""
    if feature_name in config['feature_ranges']:
        info = config['feature_ranges'][feature_name]
        return info
    return None

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

# Título principal
st.title("💧 Sistema de Predicción de Calidad del Agua")
st.markdown("---")

# Información del modelo en la barra lateral
with st.sidebar:
    st.header("📊 Información del Modelo")
    
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.write(f"**Modelo:** {config['model']['name']}")
    st.write(f"**Tipo:** {config['model']['type']}")
    st.write(f"**Versión:** {config['model']['version']}")
    st.write(f"**Fecha de Entrenamiento:** {config['model']['training_date']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🎯 Métricas del Modelo")
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{config['metrics']['accuracy']:.4f}")
        st.metric("Precision", f"{config['metrics']['precision']:.4f}")
    with col2:
        st.metric("Recall", f"{config['metrics']['recall']:.4f}")
        st.metric("F1-Score", f"{config['metrics']['f1_score']:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📖 Guía de Uso")
    st.info("""
    1. Ingrese los valores de las características del agua
    2. Los valores se ajustan automáticamente según rangos válidos
    3. Presione el botón "🔍 Realizar Predicción"
    4. Obtenga el resultado y la probabilidad
    """)

# Tabs principales
tab1, tab2, tab3 = st.tabs(["🔍 Predicción Individual", "📊 Predicción por Lote", "📈 Análisis"])

# ============================================================================
# TAB 1: PREDICCIÓN INDIVIDUAL
# ============================================================================

with tab1:
    st.header("Ingrese las Características del Agua")
    
    # Crear columnas para los inputs
    col1, col2, col3 = st.columns(3)
    
    input_values = []
    
    # Distribuir los inputs en 3 columnas
    for idx, feature in enumerate(feature_names):
        feature_info = get_feature_info(feature)
        
        if idx % 3 == 0:
            with col1:
                value = st.number_input(
                    f"**{feature}**",
                    min_value=float(feature_info['min']),
                    max_value=float(feature_info['max']),
                    value=float(feature_info['mean']),
                    step=float(feature_info['std']) / 10,
                    help=f"Rango: {feature_info['min']:.2f} - {feature_info['max']:.2f}\n"
                         f"Media: {feature_info['mean']:.2f} ± {feature_info['std']:.2f}",
                    key=f"input_{feature}"
                )
                input_values.append(value)
        elif idx % 3 == 1:
            with col2:
                value = st.number_input(
                    f"**{feature}**",
                    min_value=float(feature_info['min']),
                    max_value=float(feature_info['max']),
                    value=float(feature_info['mean']),
                    step=float(feature_info['std']) / 10,
                    help=f"Rango: {feature_info['min']:.2f} - {feature_info['max']:.2f}\n"
                         f"Media: {feature_info['mean']:.2f} ± {feature_info['std']:.2f}",
                    key=f"input_{feature}"
                )
                input_values.append(value)
        else:
            with col3:
                value = st.number_input(
                    f"**{feature}**",
                    min_value=float(feature_info['min']),
                    max_value=float(feature_info['max']),
                    value=float(feature_info['mean']),
                    step=float(feature_info['std']) / 10,
                    help=f"Rango: {feature_info['min']:.2f} - {feature_info['max']:.2f}\n"
                         f"Media: {feature_info['mean']:.2f} ± {feature_info['std']:.2f}",
                    key=f"input_{feature}"
                )
                input_values.append(value)
    
    st.markdown("---")
    
    # Botón de predicción
    if st.button("🔍 Realizar Predicción", key="predict_button"):
        with st.spinner("🔄 Analizando calidad del agua..."):
            prediction, probability = make_prediction(input_values)
            
            if prediction is not None:
                st.markdown("### 📊 Resultado de la Predicción")
                
                # Mostrar resultado principal
                if prediction == 1:
                    st.markdown(
                        '<div class="prediction-box potable">✅ AGUA POTABLE</div>',
                        unsafe_allow_html=True
                    )
                    st.success("El agua cumple con los estándares de potabilidad")
                else:
                    st.markdown(
                        '<div class="prediction-box no-potable">⚠️ AGUA NO POTABLE</div>',
                        unsafe_allow_html=True
                    )
                    st.warning("El agua NO cumple con los estándares de potabilidad")
                
                # Mostrar probabilidades
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Probabilidad de NO Potable",
                        f"{probability[0]*100:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Probabilidad de Potable",
                        f"{probability[1]*100:.2f}%",
                        delta=None
                    )
                
                # Gráfico de probabilidades
                fig = go.Figure(data=[
                    go.Bar(
                        x=['No Potable', 'Potable'],
                        y=[probability[0]*100, probability[1]*100],
                        marker_color=['#ff6b6b', '#51cf66'],
                        text=[f'{probability[0]*100:.2f}%', f'{probability[1]*100:.2f}%'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Distribución de Probabilidades",
                    yaxis_title="Probabilidad (%)",
                    xaxis_title="Clase",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar valores ingresados
                with st.expander("📋 Ver Valores Ingresados"):
                    input_df = pd.DataFrame([input_values], columns=feature_names)
                    st.dataframe(input_df.T.rename(columns={0: 'Valor'}), use_container_width=True)
                
                # Recomendaciones
                st.markdown("### 💡 Recomendaciones")
                if prediction == 0:
                    st.info("""
                    **Se recomienda:**
                    - No consumir esta agua sin tratamiento previo
                    - Realizar análisis detallado de contaminantes
                    - Consultar con especialistas en tratamiento de agua
                    - Verificar los parámetros fuera de rango normal
                    """)
                else:
                    st.success("""
                    **El agua es apta para consumo humano según el análisis:**
                    - Los parámetros están dentro de rangos aceptables
                    - Se recomienda realizar controles periódicos
                    - Mantener las condiciones de almacenamiento adecuadas
                    """)

# ============================================================================
# TAB 2: PREDICCIÓN POR LOTE
# ============================================================================

with tab2:
    st.header("📊 Predicción por Lote (CSV)")
    
    st.info("""
    **Formato del archivo CSV:**
    - El archivo debe contener las mismas columnas que el modelo fue entrenado
    - Las columnas deben estar en el siguiente orden: """ + ", ".join(feature_names) + """
    - Los valores faltantes serán imputados automáticamente
    """)
    
    uploaded_file = st.file_uploader("Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Leer CSV
            df_batch = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Archivo cargado: {len(df_batch)} muestras")
            
            # Mostrar preview
            with st.expander("👁️ Vista Previa del Dataset"):
                st.dataframe(df_batch.head(10), use_container_width=True)
            
            # Botón de predicción
            if st.button("🔍 Realizar Predicciones por Lote", key="batch_predict"):
                with st.spinner("🔄 Procesando predicciones..."):
                    # Verificar columnas
                    if list(df_batch.columns) == feature_names:
                        # Hacer predicciones
                        batch_data = df_batch.values
                        batch_imputed = imputer.transform(batch_data)
                        batch_scaled = scaler.transform(batch_imputed)
                        
                        predictions = model.predict(batch_scaled)
                        probabilities = model.predict_proba(batch_scaled)
                        
                        # Agregar resultados al dataframe
                        df_results = df_batch.copy()
                        df_results['Predicción'] = ['Potable' if p == 1 else 'No Potable' for p in predictions]
                        df_results['Prob_No_Potable'] = probabilities[:, 0]
                        df_results['Prob_Potable'] = probabilities[:, 1]
                        
                        # Mostrar resumen
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total de Muestras", len(df_results))
                        with col2:
                            potable_count = sum(predictions == 1)
                            st.metric("Muestras Potables", potable_count)
                        with col3:
                            no_potable_count = sum(predictions == 0)
                            st.metric("Muestras No Potables", no_potable_count)
                        
                        # Gráfico de distribución
                        fig = px.pie(
                            names=['Potable', 'No Potable'],
                            values=[potable_count, no_potable_count],
                            title="Distribución de Predicciones",
                            color_discrete_map={'Potable': '#51cf66', 'No Potable': '#ff6b6b'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar resultados completos
                        st.markdown("### 📋 Resultados Detallados")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Botón de descarga
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados (CSV)",
                            data=csv,
                            file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("❌ Las columnas del CSV no coinciden con las características del modelo")
                        st.write("**Columnas esperadas:**", feature_names)
                        st.write("**Columnas encontradas:**", list(df_batch.columns))
                        
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

# ============================================================================
# TAB 3: ANÁLISIS
# ============================================================================

with tab3:
    st.header("📈 Análisis y Estadísticas del Modelo")
    
    # Información del dataset de entrenamiento
    st.subheader("📊 Estadísticas del Dataset de Entrenamiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Características")
        st.write(f"**Número de características:** {len(feature_names)}")
        st.write("**Lista de características:**")
        for feature in feature_names:
            st.write(f"- {feature}")
    
    with col2:
        st.markdown("### Preprocesamiento")
        st.write(f"**Método de imputación:** {config['preprocessing']['imputer']}")
        st.write(f"**Método de escalado:** {config['preprocessing']['scaler']}")
        st.write("**Pipeline:**")
        st.write("1. Imputación de valores faltantes")
        st.write("2. Estandarización de características")
        st.write("3. Predicción con el modelo")
    
    st.markdown("---")
    
    # Rangos de características
    st.subheader("📏 Rangos de Características")
    
    ranges_data = []
    for feature in feature_names:
        info = config['feature_ranges'][feature]
        ranges_data.append({
            'Característica': feature,
            'Mínimo': f"{info['min']:.2f}",
            'Máximo': f"{info['max']:.2f}",
            'Media': f"{info['mean']:.2f}",
            'Desv. Est.': f"{info['std']:.2f}"
        })
    
    df_ranges = pd.DataFrame(ranges_data)
    st.dataframe(df_ranges, use_container_width=True)
    
    st.markdown("---")
    
    # Métricas del modelo
    st.subheader("🎯 Métricas Detalladas del Modelo")
    
    metrics_data = config['metrics']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics_data['accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{metrics_data['precision']:.4f}")
    with col3:
        st.metric("Recall", f"{metrics_data['recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics_data['f1_score']:.4f}")
    
    if metrics_data['roc_auc']:
        st.metric("ROC-AUC", f"{metrics_data['roc_auc']:.4f}")
    
    # Gráfico de métricas
    fig = go.Figure()
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    metrics_values = [metrics_data[m] for m in metrics_to_plot]
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig.add_trace(go.Bar(
        x=metrics_labels,
        y=metrics_values,
        text=[f'{v:.4f}' for v in metrics_values],
        textposition='auto',
        marker_color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    ))
    
    fig.update_layout(
        title="Métricas de Evaluación del Modelo",
        yaxis_title="Valor",
        yaxis_range=[0, 1],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Sistema de Predicción de Calidad del Agua v1.0</strong></p>
        <p>Desarrollado con ❤️ usando Streamlit | Modelo: {model_name}</p>
        <p>© 2024 - Todos los derechos reservados</p>
    </div>
""".format(model_name=config['model']['name']), unsafe_allow_html=True)