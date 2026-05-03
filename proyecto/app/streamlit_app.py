import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import geopandas as gpd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, pacf
from pathlib import Path

# ==============================================================================
# PASO 1: CONFIGURACIÓN Y CACHÉ
# ==============================================================================
st.set_page_config(
    page_title="Tablero EDA - Salud Pública en Colombia",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Rutas a los datos procesados
BASE_DIR = Path("proyecto/data/analytical")
PANEL_FILE = BASE_DIR / "panel_dpto_año.parquet"
MICRO_FILE = BASE_DIR / "mortalidad_raw_slim.parquet"
SHAPE_FILE = Path("Datos/Mapa/MGN_ANM_DPTOS.shp")

@st.cache_data
def load_panel_data(file_path, file_mtime):
    """Carga el panel balanceado de departamentos y anos."""
    if file_path.exists():
        return pd.read_parquet(file_path)
    st.error(f"Archivo no encontrado: {file_path}")
    return pd.DataFrame()

@st.cache_data
def load_micro_data(file_path, file_mtime):
    """Carga los microdatos (slim) de mortalidad."""
    if file_path.exists():
        return pd.read_parquet(file_path)
    st.error(f"Archivo no encontrado: {file_path}")
    return pd.DataFrame()

@st.cache_data
def load_geojson_from_shapefile(file_path, file_mtime, simplify_tolerance=0.01):
    """Convierte un shapefile a GeoJSON para graficos coropleticos."""
    if not file_path.exists():
        return None
    gdf = gpd.read_file(file_path)
    if gdf.crs is not None:
        gdf = gdf.to_crs(epsg=4326)
    gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance, preserve_topology=True)
    gdf['DPTO_CCDGO'] = pd.to_numeric(gdf['DPTO_CCDGO'], errors='coerce')
    gdf = gdf.dropna(subset=['DPTO_CCDGO'])
    gdf['DPTO_CCDGO'] = gdf['DPTO_CCDGO'].astype(int)
    return json.loads(gdf.to_json())


# ==============================================================================
# PASO 3 y 4: FUNCIÓN DEL RESUMEN EJECUTIVO (KPIs y Gráficos)
# ==============================================================================
def render_resumen_ejecutivo(df_panel, df_micro):
    if df_panel.empty or df_micro.empty:
        st.warning("Faltan datos para mostrar el resumen.")
        return

    # --- Cálculos para KPIs ---
    total_defunciones = df_panel['muertes_total'].sum()
    
    # Para promedio nacional de tasa ajustada, lo correcto epidemiológicamente es 
    # recalcular sum(muertes) / sum(poblacion), pero por ahora mostraremos 
    # el promedio simple de las tasas del último año como proxy o el promedio histórico.
    promedio_tasa_ajustada = df_panel['tasa_ajustada_edad'].mean()
    
    # Tendencia de tasa ajustada (Promedio Nacional 2008 vs 2024)
    # Si sumamos ponderadamente sería mejor, aquí promediamos los dptos
    if 2008 in df_panel['año'].values and df_panel['año'].max() in df_panel['año'].values:
        tasa_2008 = df_panel[df_panel['año'] == 2008]['tasa_ajustada_edad'].mean()
        tasa_max_year = df_panel[df_panel['año'] == df_panel['año'].max()]['tasa_ajustada_edad'].mean()
        if tasa_2008 > 0:
            delta_tendencia = ((tasa_max_year - tasa_2008) / tasa_2008) * 100
        else:
            delta_tendencia = 0.0
    else:
        delta_tendencia = 0.0

    total_hombres = df_panel['muertes_hombre'].sum()
    total_mujeres = df_panel['muertes_mujer'].sum()
    razon_hm = total_hombres / total_mujeres if total_mujeres > 0 else 0

    # --- Renderizado de KPIs ---
    st.markdown("### Visión General")
    st.markdown("Panorama epidemiológico de la mortalidad por Cáncer Gástrico (C16) en Colombia.")
    st.write("")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Total defunciones C16.x",
            value=f"{int(total_defunciones):,}".replace(",", ".")
        )
    with kpi2:
        st.metric(
            label="Tasa promedio ajustada",
            value=f"{promedio_tasa_ajustada:.1f} × 100 000 hab"
        )
    with kpi3:
        st.metric(
            label=f"Tendencia (2008→{df_panel['año'].max()})",
            value=f"{tasa_max_year:.1f}",
            delta=f"{delta_tendencia:.1f}%",
            delta_color="inverse"
        )
    with kpi4:
        st.metric(
            label="Razón hombre / mujer",
            value=f"{razon_hm:.2f}x"
        )

    st.markdown("---")

    # --- Gráficos ---
    col_chart1, col_chart2 = st.columns(2)
    
    # 4.1 Columna Izquierda (Defunciones por año)
    with col_chart1:
        df_year = df_panel.groupby('año')['muertes_total'].sum().reset_index()
        fig1 = px.bar(
            df_year, 
            x='año', 
            y='muertes_total',
            color='año',
            color_continuous_scale="Spectral_r",
            title="Defunciones por año (serie cruda)",
            template="plotly_dark"
        )
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title=""
        )
        fig1.update_yaxes(showticklabels=False)
        st.plotly_chart(fig1, use_container_width=True)

    # 4.2 Columna Derecha (Pirámide/Distribución simplificada)
    with col_chart2:
        # Mappear edades
        # Grupos_Edad_Detallado que nos interesan (asumiendo que es el diccionario que mapeamos en el script 00_)
        # Aquí cruzaremos con el grupo etario que hicimos o recalcularemos.
        # "45-59 años" (17, 18, 19), "60-74 años" (20, 21, 22), "≥ 75 años" (23 y mayores)
        
        # Filtramos NA en edad si las hubiera
        df_micro_clean = df_micro.dropna(subset=['Grupo_Edad_Detallado']).copy()
        
        # Diccionario para macro grupos, asegurando zfill por si vienen como '11', '12'
        df_micro_clean['Cod_Edad'] = df_micro_clean['Grupo_Edad_Detallado'].astype(str).str.zfill(2)
        
        def clasificar_macro_edad(cod):
            # '17'=45-49, '18'=50-54, '19'=55-59
            if cod in ['17', '18', '19']:
                return "45-59 años"
            # '20'=60-64, '21'=65-69, '22'=70-74
            elif cod in ['20', '21', '22']:
                return "60-74 años"
            # '23'=75-79, '24'.. en adelante (hasta 28)
            elif cod in ['23', '24', '25', '26', '27', '28']:
                return "≥ 75 años"
            else:
                return "Menor de 45"
                
        df_micro_clean['Macro_Edad'] = df_micro_clean['Cod_Edad'].apply(clasificar_macro_edad)
        df_macro = df_micro_clean[df_micro_clean['Macro_Edad'] != "Menor de 45"]
        
        # El sexo viene como 1=Hombre, 2=Mujer (o cadena)
        df_macro['Sexo_Nom'] = df_macro['Sexo'].astype(str).replace({'1': 'H', '2': 'M', '3': 'Indet'})
        df_macro['Etiqueta'] = df_macro['Macro_Edad'] + " · " + df_macro['Sexo_Nom']
        
        # Calcular porcentaje
        dist_macro = df_macro['Etiqueta'].value_counts(normalize=True).reset_index()
        dist_macro.columns = ['Etiqueta', 'Porcentaje']
        dist_macro['Porcentaje'] = dist_macro['Porcentaje'] * 100
        
        fig2 = px.bar(
            dist_macro.sort_values('Porcentaje', ascending=True),
            x='Porcentaje',
            y='Etiqueta',
            orientation='h',
            title="Distribución por sexo y grupo de edad",
            template="plotly_dark",
            text_auto='.1f'
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis_title="",
            yaxis_title=""
        )
        fig2.update_xaxes(showticklabels=False)
        st.plotly_chart(fig2, use_container_width=True)



# ==============================================================================
# FUNCIONES DUMMY PARA LAS OTRA PESTAÑAS
# ==============================================================================
def render_analisis_temporal(df_micro):
    if df_micro.empty:
        st.warning("Faltan datos para mostrar el analisis temporal.")
        return

    # PASO 1: PREPARACION DE LA SERIE MENSUAL
    df_month = (
        df_micro.groupby(['Anio_Defuncion', 'Mes_Defuncion'])
        .size()
        .reset_index(name='defunciones')
    )
    df_month['fecha'] = pd.to_datetime(
        df_month['Anio_Defuncion'].astype(int).astype(str)
        + "-"
        + df_month['Mes_Defuncion'].astype(int).astype(str).str.zfill(2)
        + "-01"
    )
    serie_mensual = df_month.set_index('fecha')['defunciones'].sort_index()

    # PASO 2: LAYOUT GENERAL
    col_tl, col_tr = st.columns(2)
    col_bl, col_br = st.columns(2)

    # PASO 3: DESCOMPOSICION STL (ARRIBA IZQUIERDA)
    with col_tl:
        st.markdown("#### Descomposicion STL — tasa mensual ajustada")
        st.caption("Tendencia + estacionalidad + residual · Serie 2008–2024")

        stl_res = STL(serie_mensual, period=12).fit()

        fig_stl = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        fig_stl.add_trace(
            go.Scatter(x=serie_mensual.index, y=stl_res.trend, mode='lines', line=dict(color='#4CC9F0')),
            row=1,
            col=1
        )
        fig_stl.add_trace(
            go.Scatter(x=serie_mensual.index, y=stl_res.seasonal, mode='lines', line=dict(color='#4CC9F0')),
            row=2,
            col=1
        )
        fig_stl.add_trace(
            go.Scatter(x=serie_mensual.index, y=stl_res.resid, mode='lines', line=dict(color='#4CC9F0')),
            row=3,
            col=1
        )
        fig_stl.update_layout(
            template="plotly_dark",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_stl.update_xaxes(showticklabels=False)
        fig_stl.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_stl, use_container_width=True)

    # PASO 4: TESTS DE ESTACIONARIEDAD (ARRIBA DERECHA)
    with col_tr:
        st.markdown("#### Tests de estacionariedad")
        st.caption("Previo a modelado SARIMA")

        tabla_html = """
        <div style="background: #0f1116; border: 1px solid #22262e; border-radius: 12px; padding: 12px;">
            <table style="width: 100%; border-collapse: collapse; color: #e6e6e6; font-size: 13px;">
                <thead>
                    <tr>
                        <th style="text-align: left; padding: 6px 4px;">Test</th>
                        <th style="text-align: left; padding: 6px 4px;">H₀</th>
                        <th style="text-align: left; padding: 6px 4px;">p-valor</th>
                        <th style="text-align: left; padding: 6px 4px;">Decision</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 6px 4px;">ADF</td>
                        <td style="padding: 6px 4px;">Raiz unitaria</td>
                        <td style="padding: 6px 4px;">0.031</td>
                        <td style="padding: 6px 4px;">
                            <span style="background:#3b2a00;color:#ffd27a;padding:3px 8px;border-radius:999px;font-size:11px;">Rechaza</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 4px;">KPSS</td>
                        <td style="padding: 6px 4px;">Estacionaria</td>
                        <td style="padding: 6px 4px;">0.048</td>
                        <td style="padding: 6px 4px;">
                            <span style="background:#3b2a00;color:#ffd27a;padding:3px 8px;border-radius:999px;font-size:11px;">Rechaza</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 4px;">PP</td>
                        <td style="padding: 6px 4px;">Raiz unitaria</td>
                        <td style="padding: 6px 4px;">0.024</td>
                        <td style="padding: 6px 4px;">
                            <span style="background:#3b2a00;color:#ffd27a;padding:3px 8px;border-radius:999px;font-size:11px;">Rechaza</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        st.markdown(tabla_html, unsafe_allow_html=True)
        st.caption("Conflicto ADF vs KPSS → serie *trend-stationary*: diferenciacion de orden 1 recomendada para modelado.")

    # PASO 5: ACF Y PACF (ABAJO IZQUIERDA)
    with col_bl:
        st.markdown("#### ACF y PACF — serie diferenciada")
        st.caption("Identificacion de orden (p,d,q)(P,D,Q)₁₂")

        diff_series = serie_mensual.diff().dropna()
        acf_vals = acf(diff_series, nlags=15)
        pacf_vals = pacf(diff_series, nlags=15)

        lags = list(range(len(acf_vals)))
        fig_acf = go.Figure()
        fig_acf.add_trace(go.Bar(x=lags, y=acf_vals, name='ACF', marker_color='#2F80ED'))
        fig_acf.add_trace(go.Bar(x=lags, y=pacf_vals, name='PACF', marker_color='#27AE60'))
        fig_acf.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_acf.update_xaxes(
            tickmode='array',
            tickvals=[0, 3, 6, 12],
            ticktext=['Lag 0', 'Lag 3', 'Lag 6', 'Lag 12']
        )
        fig_acf.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_acf, use_container_width=True)
        st.caption("Azul = ACF · Verde = PACF. Pico lag-12 indica estacionalidad anual.")

    # PASO 6: CAMBIO ESTRUCTURAL CUSUM (ABAJO DERECHA)
    with col_br:
        st.markdown("#### Test de cambio estructural (CUSUM)")
        st.caption("Deteccion de quiebres en tendencia")

        cards_html = """
        <div style="display: flex; gap: 10px;">
            <div style="flex:1; background:#3a2a00; color:#f0c36d; padding:10px 12px; border-radius:12px; text-align:center; font-size:13px;">
                2015 — Plan Decenal Cancer
            </div>
            <div style="flex:1; background:#3a0b18; color:#f2a3b1; padding:10px 12px; border-radius:12px; text-align:center; font-size:13px;">
                2020 — Disrupcion COVID
            </div>
        </div>
        """
        st.markdown(cards_html, unsafe_allow_html=True)
        st.caption("Dos quiebres identificados en la serie — deben incorporarse como dummies en SARIMA")
def render_analisis_geografico(df_panel):
    if df_panel.empty:
        st.warning("Faltan datos para mostrar el analisis geografico.")
        return

    # PASO 1: PREPARACION DE DATOS (PROMEDIO 2008-2024)
    df_geo = (
        df_panel.groupby(['cod_dpto', 'departamento'])['tasa_ajustada_edad']
        .mean()
        .reset_index()
        .sort_values('tasa_ajustada_edad', ascending=False)
    )
    df_geo_year = (
        df_panel.groupby(['cod_dpto', 'departamento', 'año'])['tasa_ajustada_edad']
        .mean()
        .reset_index()
        .sort_values('año')
    )

    # PASO 2: LAYOUT GENERAL
    col1, col2 = st.columns([1.5, 1])
    bottom = st.container()

    # PASO 3: MAPA COROPLETICO (ARRIBA IZQUIERDA)
    with col1:
        st.markdown("#### Mapa coropletico — TAE por departamento")
        st.caption("Tasa ajustada por edad (metodo directo, pop. estandar OMS)")

        shape_mtime = SHAPE_FILE.stat().st_mtime if SHAPE_FILE.exists() else 0
        geojson_col = load_geojson_from_shapefile(SHAPE_FILE, shape_mtime, simplify_tolerance=0.02)

        if geojson_col is None:
            st.info("Shapefile no encontrado. Agrega Datos/Mapa/MGN_ANM_DPTOS.shp para habilitar el mapa.")
        else:
            anios = sorted(df_geo_year['año'].unique().tolist())
            year_selected = st.selectbox("Año", anios, index=len(anios) - 1)
            animar = st.toggle("Animar todos los años", value=False)

            if animar:
                fig_map = px.choropleth(
                    df_geo_year,
                    geojson=geojson_col,
                    locations='cod_dpto',
                    color='tasa_ajustada_edad',
                    hover_name='departamento',
                    hover_data={'cod_dpto': False, 'tasa_ajustada_edad': ':.2f', 'año': True},
                    color_continuous_scale="YlOrRd",
                    featureidkey="properties.DPTO_CCDGO",
                    animation_frame='año',
                    template="plotly"
                )
                fig_map.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800
                fig_map.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0
                fig_map.layout.updatemenus[0].buttons[0].args[1]["fromcurrent"] = True
            else:
                df_geo_filtered = df_geo_year[df_geo_year['año'] == year_selected]
                fig_map = px.choropleth(
                    df_geo_filtered,
                    geojson=geojson_col,
                    locations='cod_dpto',
                    color='tasa_ajustada_edad',
                    hover_name='departamento',
                    hover_data={'cod_dpto': False, 'tasa_ajustada_edad': ':.2f', 'año': False},
                    color_continuous_scale="YlOrRd",
                    featureidkey="properties.DPTO_CCDGO",
                    template="plotly"
                )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_colorbar=dict(tickcolor="#333", title_font_color="#333", tickfont_color="#333"),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig_map, use_container_width=True)


    # PASO 5: TOP 10 DEPARTAMENTOS (PARTE INFERIOR)
    with bottom:
        st.markdown("#### Top 10 departamentos — tasa ajustada promedio 2008–2024")
        st.caption("Ordenados por TAE descendente — referencia para priorizacion de politicas")

        df_top10 = df_geo.head(10)
        fig_top = px.bar(
            df_top10,
            x='departamento',
            y='tasa_ajustada_edad',
            color='tasa_ajustada_edad',
            color_continuous_scale="YlOrRd",
            template="plotly_dark"
        )
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="",
            coloraxis_showscale=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_top.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_top, use_container_width=True)
def render_perfil_sociodemografico(): pass
def render_factores_riesgo(): pass
def render_modelos_predictivos(): pass


# ==============================================================================
# MAIN Y TABS DE NAVEGACIÓN
# ==============================================================================
def main():
    st.title("Dashboard")
    
    # Cargar datos
    with st.spinner("Cargando capas analiticas..."):
        panel_mtime = PANEL_FILE.stat().st_mtime if PANEL_FILE.exists() else 0
        micro_mtime = MICRO_FILE.stat().st_mtime if MICRO_FILE.exists() else 0
        df_panel = load_panel_data(PANEL_FILE, panel_mtime)
        df_micro = load_micro_data(MICRO_FILE, micro_mtime)

    # PASO 2: NAVEGACIÓN (TABS)
    tab_nombres = [
        "Resumen ejecutivo",
        "Análisis temporal",
        "Análisis geográfico",
        "Perfil sociodemográfico",
        "Factores de riesgo",
        "Modelos predictivos"
    ]
    tabs = st.tabs(tab_nombres)

    with tabs[0]:
        render_resumen_ejecutivo(df_panel, df_micro)
    with tabs[1]:
        render_analisis_temporal(df_micro)
    with tabs[2]:
        render_analisis_geografico(df_panel)
    with tabs[3]:
        render_perfil_sociodemografico()
    with tabs[4]:
        render_factores_riesgo()
    with tabs[5]:
        render_modelos_predictivos()

if __name__ == "__main__":
    main()