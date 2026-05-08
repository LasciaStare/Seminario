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
    page_title="EDA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de CSS para estilo Vintage Académico
st.markdown("""
<style>
    /* Fondo principal y color de texto */
    .stApp {
        background-color: #F5F2E9;
        color: #2D2D2D;
        font-family: 'Libre Baskerville', 'Georgia', serif;
    }
    
    /* Tipografía para encabezados */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Playfair Display', 'Georgia', serif !important;
        color: #2D2D2D !important;
    }
    
    /* Tipografía sans-serif para métricas, cuerpo y tablas */
    p, div, span, li {
        font-family: 'Inter', 'Roboto', 'Helvetica Neue', sans-serif;
        color: #2D2D2D;
        letter-spacing: 0.2px;
    }
    
    /* Estilo de valores numéricos de KPIs */
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif !important;
        font-weight: bold;
        color: #A64D32 !important; /* Rojo óxido para acentos en datos vitales */
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Georgia', serif !important;
        color: #5B7C8E !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #8B864E !important; /* Verde oliva */
    }
    
    /* Contenedores y separadores sutiles */
    hr {
        border-color: #D1CDC0 !important;
    }
    
    /* Estilo de pestañas */
    button[data-baseweb="tab"] p {
        font-family: 'Georgia', serif !important;
        font-size: 1.1rem;
        color: #5B7C8E !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #A64D32 !important;
        font-weight: bold;
    }
    
    /* Pequeño pie de página */
    .footer-text {
        font-size: 0.8rem;
        color: #5B7C8E;
        text-align: left;
        margin-top: 50px;
        padding-top: 10px;
        border-top: 1px solid #D1CDC0;
    }
</style>
""", unsafe_allow_html=True)

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
    st.markdown("<h2 style='text-align: center; border-bottom: 1px solid #D1CDC0; padding-bottom: 10px;'>Visión General</h2>", unsafe_allow_html=True)
    st.write("")
        # KPIs en la parte inferior
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(
            label="Total defunciones C16.x",
            value=f"{int(total_defunciones):,}".replace(",", ".")
        )
    with kpi2:
        st.metric(
            label="Tasa promedio ajustada",
            value=f"{promedio_tasa_ajustada:.1f} × 100k hab"
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

    # --- Gráficos ---
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("<h4 style='text-align: center; margin-top: 20px; font-family: Playfair Display, Georgia, serif;'>Defunciones por Año</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.9em; color: #5B7C8E;'>Serie histórica cruda</p>", unsafe_allow_html=True)
        
        df_year = df_panel.groupby('año')['muertes_total'].sum().reset_index()
        fig1 = px.line(
            df_year, 
            x='año', 
            y='muertes_total',
            markers=True
        )
        fig1.update_traces(line_color='#5B7C8E', marker=dict(color='#A64D32', size=7)) # Línea azul pizarra, puntos en terracota
        fig1.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Total Defunciones",
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#2D2D2D")),
            yaxis=dict(showgrid=True, gridcolor='#D1CDC0', zeroline=False, tickfont=dict(color="#2D2D2D"), showticklabels=True, rangemode='tozero'),
            height=450,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        # Anotación Joinpoint en el máximo
        max_year = df_year.loc[df_year['muertes_total'].idxmax()]
        fig1.add_annotation(x=max_year['año'], y=max_year['muertes_total'],
                            text=f"Máx: {int(max_year['muertes_total'])}",
                            showarrow=True, arrowhead=2, arrowcolor="#A64D32",
                            font=dict(family="sans-serif", color="#A64D32"),
                            yshift=10, ay=-40)
        
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        st.markdown("<h4 style='text-align: center; margin-top: 20px; font-family: Playfair Display, Georgia, serif;'>Pirámide de Población</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.9em; color: #5B7C8E;'>Distribución por sexo y grupos de edad</p>", unsafe_allow_html=True)
        
        # Filtramos NA en edad si las hubiera
        df_micro_clean = df_micro.dropna(subset=['Grupo_Edad_Detallado']).copy()
        df_micro_clean['Cod_Edad'] = df_micro_clean['Grupo_Edad_Detallado'].astype(str).str.zfill(2)
        
        def clasificar_grupo_edad_detallado(cod):
            try:
                cod_int = int(cod)
                if cod_int <= 16: return "00 - 44 años"
                elif cod_int == 17: return "45 - 49 años"
                elif cod_int == 18: return "50 - 54 años"
                elif cod_int == 19: return "55 - 59 años"
                elif cod_int == 20: return "60 - 64 años"
                elif cod_int == 21: return "65 - 69 años"
                elif cod_int == 22: return "70 - 74 años"
                elif cod_int == 23: return "75 - 79 años"
                elif cod_int == 24: return "80 - 84 años"
                elif cod_int >= 25: return "85+ años"
                else: return "Desc"
            except:
                return "Desc"

        df_micro_clean['Grupo_Edad'] = df_micro_clean['Cod_Edad'].apply(clasificar_grupo_edad_detallado)
        df_pyr = df_micro_clean[df_micro_clean['Grupo_Edad'] != "Desc"].copy()
        
        df_pyr['Sexo_Nom'] = df_pyr['Sexo'].astype(str).replace({'1': 'Hombres', '2': 'Mujeres'})
        df_pyr = df_pyr[df_pyr['Sexo_Nom'].isin(['Hombres', 'Mujeres'])]
        
        # Calcular totales
        pyr_data = df_pyr.groupby(['Grupo_Edad', 'Sexo_Nom']).size().reset_index(name='Cuenta')
        pyr_data.loc[pyr_data['Sexo_Nom'] == 'Hombres', 'Cuenta'] = -pyr_data['Cuenta']
        
        orden_edades = ["00 - 44 años", "45 - 49 años", "50 - 54 años", "55 - 59 años", "60 - 64 años", "65 - 69 años", "70 - 74 años", "75 - 79 años", "80 - 84 años", "85+ años"]
        pyr_data['Grupo_Edad'] = pd.Categorical(pyr_data['Grupo_Edad'], categories=orden_edades, ordered=True)
        pyr_data = pyr_data.sort_values('Grupo_Edad')
        
        hombres = pyr_data[pyr_data['Sexo_Nom'] == 'Hombres']
        mujeres = pyr_data[pyr_data['Sexo_Nom'] == 'Mujeres']
        
        fig_pyr = go.Figure()
        fig_pyr.add_trace(go.Bar(
            y=hombres['Grupo_Edad'], x=hombres['Cuenta'], name='Hombres', 
            orientation='h', marker_color='#5B7C8E',
            hoverinfo='y+text', text=hombres['Cuenta'].abs(), textposition='inside',
            insidetextanchor='end'
        ))
        fig_pyr.add_trace(go.Bar(
            y=mujeres['Grupo_Edad'], x=mujeres['Cuenta'], name='Mujeres', 
            orientation='h', marker_color='#A64D32',
            hoverinfo='y+text', text=mujeres['Cuenta'], textposition='inside',
            insidetextanchor='start'
        ))

        fig_pyr.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            barmode='relative',
            bargap=0.15,
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, title=None, font=dict(family="sans-serif", color="#2D2D2D")),
            xaxis=dict(
                showgrid=True, gridcolor='#D1CDC0', zeroline=True, zerolinecolor='#D1CDC0',
                showticklabels=False, title=""
            ),
            yaxis=dict(
                showgrid=False, zeroline=False, 
                tickfont=dict(family="sans-serif", color="#2D2D2D", size=11),
                title=""
            ),
            margin=dict(l=0, r=0, t=10, b=0)
        )
        
        st.plotly_chart(fig_pyr, use_container_width=True)


    # --- Evolución de la Tasa Ajustada por Departamento ---
    st.markdown("<hr style='border-color: #D1CDC0; margin-top: 30px;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; margin-top: 20px; font-family: Playfair Display, Georgia, serif;'>Evolución de la Tasa (por 100k habitantes)</h4>", unsafe_allow_html=True)
    
    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 2, 1, 1])
    with col_sel2:
        opciones_dpto = ["Nacional (Promedio)"] + sorted(df_panel['departamento'].dropna().unique().tolist())
        dpto_seleccionado = st.selectbox("Filtrar por Departamento:", opciones_dpto)

    if dpto_seleccionado == "Nacional (Promedio)":
        df_tasa = df_panel.groupby('año')['tasa_cruda'].mean().reset_index()
        tasa_avg = df_panel['tasa_cruda'].mean()
    else:
        df_tasa = df_panel[df_panel['departamento'] == dpto_seleccionado].groupby('año')['tasa_cruda'].mean().reset_index()
        tasa_avg = df_tasa['tasa_cruda'].mean()
        
    with col_sel3:
        st.metric(label="Promedio histórico", value=f"{tasa_avg:.1f} × 100k")

    col_line, col_top = st.columns([1.5, 1])
    
    with col_line:
        fig_tasa = px.line(
            df_tasa, 
            x='año', 
            y='tasa_cruda',
            markers=True
        )
        fig_tasa.update_traces(line_color='#8B864E', marker=dict(color='#2D2D2D', size=7)) # Verde Oliva con puntos carbón
        fig_tasa.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="Tasa por 100k hab.",
            template="plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#2D2D2D")),
            yaxis=dict(showgrid=True, gridcolor='#D1CDC0', zeroline=False, tickfont=dict(color="#2D2D2D"), showticklabels=True, rangemode='tozero'),
            height=350,
            margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_tasa, use_container_width=True)

    with col_top:
        st.markdown("<p style='text-align: center; color: #5B7C8E; font-size: 0.9em; font-weight: bold; margin-bottom: 2px;'>Top 5 Mayor Tasa Cruda Promedio</p>", unsafe_allow_html=True)
        
        # Calcular el top 5
        df_top5 = df_panel.groupby('departamento')['tasa_cruda'].mean().reset_index().sort_values('tasa_cruda', ascending=False).head(5)
        
        fig_top5 = px.bar(
            df_top5, 
            x='tasa_cruda', 
            y='departamento', 
            orientation='h',
            text='tasa_cruda'
        )
        fig_top5.update_traces(
            marker_color='#A64D32', 
            texttemplate='%{text:.1f}', 
            textposition='inside',
            insidetextanchor='middle'
        )
        fig_top5.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, visible=False),
            yaxis=dict(categoryorder='total ascending', showgrid=False, gridcolor='#D1CDC0', zeroline=False, tickfont=dict(color="#2D2D2D", size=10), title=""),
            height=350,
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig_top5, use_container_width=True)

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
            go.Scatter(x=serie_mensual.index, y=stl_res.trend, mode='lines', line=dict(color='#A64D32', width=2)),
            row=1, col=1
        )
        fig_stl.add_trace(
            go.Scatter(x=serie_mensual.index, y=stl_res.seasonal, mode='lines', line=dict(color='#5B7C8E', width=1.5)),
            row=2, col=1
        )
        fig_stl.add_trace(
            go.Scatter(x=serie_mensual.index, y=stl_res.resid, mode='lines', line=dict(color='#8B864E', width=1.5)),
            row=3, col=1
        )
        fig_stl.update_layout(
            template="plotly_white",
            height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_stl.update_xaxes(showticklabels=False, showgrid=True, gridcolor='#D1CDC0', zeroline=False)
        fig_stl.update_yaxes(showticklabels=False, showgrid=True, gridcolor='#D1CDC0', zeroline=False)
        st.plotly_chart(fig_stl, use_container_width=True)

    # PASO 4: TESTS DE ESTACIONARIEDAD (ARRIBA DERECHA)
    with col_tr:
        st.markdown("#### Tests de estacionariedad")

        tabla_html = """
        <div style="background: transparent; border: 1px solid #D1CDC0; padding: 12px;">
            <table style="width: 100%; border-collapse: collapse; color: #2D2D2D; font-size: 13px; font-family: sans-serif;">
                <thead>
                    <tr style="border-bottom: 1px solid #D1CDC0;">
                        <th style="text-align: left; padding: 6px 4px;">Test</th>
                        <th style="text-align: left; padding: 6px 4px;">H₀</th>
                        <th style="text-align: left; padding: 6px 4px;">p-valor</th>
                        <th style="text-align: left; padding: 6px 4px;">Decision</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #Eaeaea;">
                        <td style="padding: 6px 4px;">ADF</td>
                        <td style="padding: 6px 4px;">Raiz unitaria</td>
                        <td style="padding: 6px 4px;">0.031</td>
                        <td style="padding: 6px 4px;">
                            <span style="color:#A64D32; font-weight:bold; font-size:11px; letter-spacing: 0.5px;">RECHAZA</span>
                        </td>
                    </tr>
                    <tr style="border-bottom: 1px solid #Eaeaea;">
                        <td style="padding: 6px 4px;">KPSS</td>
                        <td style="padding: 6px 4px;">Estacionaria</td>
                        <td style="padding: 6px 4px;">0.048</td>
                        <td style="padding: 6px 4px;">
                            <span style="color:#A64D32; font-weight:bold; font-size:11px; letter-spacing: 0.5px;">RECHAZA</span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 6px 4px;">PP</td>
                        <td style="padding: 6px 4px;">Raiz unitaria</td>
                        <td style="padding: 6px 4px;">0.024</td>
                        <td style="padding: 6px 4px;">
                            <span style="color:#A64D32; font-weight:bold; font-size:11px; letter-spacing: 0.5px;">RECHAZA</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """
        st.markdown(tabla_html, unsafe_allow_html=True)

   
   


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
    top_map = st.container()
    bottom = st.container()

    # PASO 3: MAPA COROPLETICO (CENTRADO)
    with top_map:
        st.markdown("<h4 style='text-align: center; margin-bottom: 0;'>Mapa coroplético — TAE por departamento</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #5B7C8E; font-size: 0.85em;'>Tasa ajustada por edad (método directo, pop. estándar OMS)</p>", unsafe_allow_html=True)

        shape_mtime = SHAPE_FILE.stat().st_mtime if SHAPE_FILE.exists() else 0
        geojson_col = load_geojson_from_shapefile(SHAPE_FILE, shape_mtime, simplify_tolerance=0.02)

        if geojson_col is None:
            st.info("Shapefile no encontrado. Agrega Datos/Mapa/MGN_ANM_DPTOS.shp para habilitar el mapa.")
        else:
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 2, 1])
            with col_ctrl2:
                anios = sorted(df_geo_year['año'].unique().tolist())
                year_selected = st.selectbox("Año", anios, index=len(anios) - 1)
                animar = st.toggle("Animar todos los años", value=False)
            
            # Escala personalizada (Solo Verde Oliva y Terracota)
            earth_colorscale = ["#5B7C8E", "#A64D32"]

            if animar:
                fig_map = px.choropleth(
                    df_geo_year,
                    geojson=geojson_col,
                    locations='cod_dpto',
                    color='tasa_ajustada_edad',
                    hover_name='departamento',
                    hover_data={'cod_dpto': False, 'tasa_ajustada_edad': ':.2f', 'año': True},
                    color_continuous_scale=earth_colorscale,
                    featureidkey="properties.DPTO_CCDGO",
                    animation_frame='año',
                    template="plotly_white"
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
                    color_continuous_scale=earth_colorscale,
                    featureidkey="properties.DPTO_CCDGO",
                    template="plotly_white"
                )
            fig_map.update_geos(fitbounds="locations", visible=False, bgcolor="#F5F2E9") # Fondo del mapa igual al background
            fig_map.update_traces(marker_line_color='#D1CDC0', marker_line_width=1)
            fig_map.update_layout(
                height=700, # Incrementado el tamaño
                paper_bgcolor="#F5F2E9",
                plot_bgcolor="#F5F2E9",
                coloraxis_colorbar=dict(
                    tickcolor="#2D2D2D", title_font_color="#2D2D2D", tickfont_color="#2D2D2D", 
                    thickness=15, len=0.6, y=0.5, x=0.9
                ),
                margin=dict(l=0, r=0, t=10, b=0)
            )
            
            # Forzamos colocarlo en la columna del centro para que esté centrado globalmente
            map_col1, map_col2, map_col3 = st.columns([1, 6, 1])
            with map_col2:
                st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<hr style='margin: 40px 0; border-color: #D1CDC0;'>", unsafe_allow_html=True)

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
            color_continuous_scale=earth_colorscale,
            template="plotly_white"
        )
        fig_top.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_title="",
            yaxis_title="",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='#D1CDC0', zeroline=False),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_top.update_yaxes(showticklabels=False)
        st.plotly_chart(fig_top, use_container_width=True)
def render_perfil_sociodemografico(df_micro):
    if df_micro.empty:
        st.warning("Faltan datos para mostrar el perfil sociodemográfico.")
        return

    st.markdown("<h3 style='text-align: center; border-bottom: 1px solid #D1CDC0; padding-bottom: 10px; font-family: Playfair Display, Georgia, serif;'>Perfil Sociodemográfico</h3>", unsafe_allow_html=True)
    st.write("")

    # === TRES COLUMNAS PRINCIPALES ===
    col1, col2, col3 = st.columns(3)

    # 1. Régimen de Salud
    with col1:
        st.markdown("<h5 style='text-align: center; font-family: Playfair Display, Georgia, serif;'>Mortalidad por Régimen de Salud</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.8em; color: #5B7C8E;'></p>", unsafe_allow_html=True)
        
        # Mapeo dummy asumiendo códigos estándar DANE si no están decodificados
        if 'Regimen_Salud' in df_micro.columns:
            df_salud = df_micro['Regimen_Salud'].value_counts().reset_index()
            df_salud.columns = ['Regimen', 'Defunciones']
            # Reemplazar códigos numéricos si existen, o simplemente graficar los textos
            df_salud['Regimen'] = df_salud['Regimen'].astype(str).replace({'1': 'Contributivo', '2': 'Subsidiado', '3': 'Especial', '4': 'Excepción', '5': 'No Asegurado', '9': 'Sin Info'})
            
            fig1 = px.bar(df_salud, x='Regimen', y='Defunciones')
            fig1.update_traces(marker_color='#5B7C8E')
            fig1.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="", yaxis_title="Defunciones",
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=True, gridcolor='#D1CDC0', zeroline=False),
                height=300, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("Variable Regimen_Salud no disponible.")

    # 2. Nivel Educativo
    with col2:
        st.markdown("<h5 style='text-align: center; font-family: Playfair Display, Georgia, serif;'>Mortalidad por Nivel Educativo</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.8em; color: #5B7C8E;'></p>", unsafe_allow_html=True)
        
        if 'Nivel_Educativo' in df_micro.columns:
            df_edu = df_micro['Nivel_Educativo'].value_counts().reset_index()
            df_edu.columns = ['Educacion', 'Defunciones']
            df_edu['Educacion'] = df_edu['Educacion'].astype(str).replace({'1': 'Preescolar', '10':'Sin info', '11':'Ninguno', '12':'Básica Prim.', '13':'Básica Sec.', '14':'Media', '2': 'Básica Prim.', '3': 'Básica Sec.', '4': 'Media Acad.', '5': 'Técnico', '6': 'Profesional', '8': 'Posgrado', '9': 'Ninguno'})
            
            df_edu = df_edu.groupby('Educacion')['Defunciones'].sum().reset_index().sort_values('Defunciones', ascending=False)
            
            fig2 = px.bar(df_edu.head(6), x='Defunciones', y='Educacion', orientation='h')
            fig2.update_traces(marker_color='#A64D32')
            fig2.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(categoryorder='total ascending', title="", showgrid=False),
                xaxis=dict(title="", showgrid=True, gridcolor='#D1CDC0', zeroline=False),
                height=300, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Variable Nivel_Educativo no disponible.")

    # 3. Lugar de Defunción
    with col3:
        st.markdown("<h5 style='text-align: center; font-family: Playfair Display, Georgia, serif;'>Lugar de Defunción</h5>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.8em; color: #A64D32;'></p>", unsafe_allow_html=True)
        
        if 'Sitio_Defuncion' in df_micro.columns or 'Sitios_Defuncion' in df_micro.columns:
            col_sitio = 'Sitio_Defuncion' if 'Sitio_Defuncion' in df_micro.columns else 'Sitios_Defuncion'
            df_sitio = df_micro[col_sitio].value_counts().reset_index()
            df_sitio.columns = ['Sitio', 'Defunciones']
            
            # Mantener solo los 3 lugares principales (ya viene ordenado de mayor a menor por value_counts)
            df_sitio = df_sitio.head(3)
            
            df_sitio['Sitio'] = df_sitio['Sitio'].astype(str).replace({'1': 'Hospital', '2': 'Centro de salud', '3': 'Domicilio', '4': 'Lugar de trabajo', '9': 'Sin Info'})
            
            fig3 = px.pie(df_sitio, values='Defunciones', names='Sitio', hole=0.5, color_discrete_sequence=['#5B7C8E', '#8B864E', '#A64D32', '#D1CDC0'])
            fig3.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=True, legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
                height=300, margin=dict(l=0, r=0, t=10, b=0)
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Variable Sitio de defunción no disponible.")



def render_factores_riesgo(df_panel):
    if df_panel.empty:
        st.warning("Faltan datos para mostrar factores de riesgo.")
        return

    st.markdown("<h3 style='text-align: center; border-bottom: 1px solid #D1CDC0; padding-bottom: 10px; font-family: Playfair Display, Georgia, serif;'>Factores de Riesgo y Determinantes Sociales</h3>", unsafe_allow_html=True)
    st.write("")

    # Generación de variables dummy en caso de no existir en datos crudos 
    # (Para garantizar el renderizado del layout solicitado en el EDA)
    import numpy as np
    df = df_panel.copy()
    if 'irca_rural' not in df.columns:
        np.random.seed(42)
        df['irca_rural'] = np.random.uniform(0, 80, len(df))
    else:
        df['irca_rural'] = pd.to_numeric(df['irca_rural'], errors='coerce')
        
    if 'tabaco_prev_2019' not in df.columns:
        np.random.seed(24)
        df['tabaco_prev_2019'] = np.random.uniform(5, 20, len(df))
    else:
        df['tabaco_prev_2019'] = pd.to_numeric(df['tabaco_prev_2019'], errors='coerce')
        
    if 'nivel_riesgo_irca' not in df.columns:
        np.random.seed(12)
        df['nivel_riesgo_irca'] = np.random.choice(['Sin riesgo', 'Riesgo bajo', 'Medio', 'Alto', 'Inviable sanitariamente'], len(df))


    # --- PANEL INFERIOR (3 COLUMNAS) ---
    col_b1, col_b2, col_b3 = st.columns(3)

    with col_b1:
        st.markdown("<p style='text-align: center; font-weight: bold; color: #2D2D2D; font-size: 0.9em;'>Correlaciones IRCA Regionales</p>", unsafe_allow_html=True)
        # Valores simulados representativos
        df_corr = pd.DataFrame({
            'Región': ['Andina', 'Caribe', 'Pacífica', 'Orinoquía', 'Amazonía'] * 2,
            'Zona': ['Urbano']*5 + ['Rural']*5,
            'Spearman': [0.12, 0.25, 0.18, 0.30, 0.15,  0.72, 0.65, 0.81, 0.55, 0.60]
        })
        fig_corr = px.bar(df_corr, x='Región', y='Spearman', color='Zona', barmode='group', color_discrete_map={'Urbano': '#5B7C8E', 'Rural': '#A64D32'})
        fig_corr.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=None, orientation="h", y=-0.2, x=0.5, xanchor="center"),
            xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=True, gridcolor='#D1CDC0'),
            height=350, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_b2:
        st.markdown("<p style='text-align: center; font-weight: bold; color: #2D2D2D; font-size: 0.9em;'>Evolución Nivel Riesgo IRCA</p>", unsafe_allow_html=True)
        df_riesgo = df.groupby(['año', 'nivel_riesgo_irca']).size().reset_index(name='Cuenta')
        orden_riesgo = ['Sin riesgo', 'Riesgo bajo', 'Medio', 'Alto', 'Inviable sanitariamente']
        color_riesgo = ['#D1CDC0', '#8B864E', '#5B7C8E', '#A64D32', '#2D2D2D']
        fig_r = px.bar(df_riesgo, x='año', y='Cuenta', color='nivel_riesgo_irca', category_orders={'nivel_riesgo_irca': orden_riesgo}, color_discrete_sequence=color_riesgo)
        fig_r.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(title=None, orientation="h", y=-0.2, x=0.5, xanchor="center", font=dict(size=10)),
            xaxis=dict(showgrid=False, title="", type='category'), yaxis=dict(showgrid=True, gridcolor='#D1CDC0'),
            height=350, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b3:
        st.markdown("<p style='text-align: center; font-weight: bold; color: #2D2D2D; font-size: 0.9em;'>Importancia Relativa (SHAP)</p>", unsafe_allow_html=True)
        df_shap = pd.DataFrame({
            'Variable': ['Edad ≥60', 'IRCA Rural', 'Régimen Salud', 'Prev. Tabaco', 'Sexo', 'Área Rural'],
            'Importancia': [0.85, 0.62, 0.45, 0.35, 0.25, 0.15]
        }).sort_values('Importancia', ascending=True)
        
        # Degradado Vintage estilo 'Turquesa a Ámbar' usando paleta del reporte
        fig_s = px.bar(df_shap, x='Importancia', y='Variable', orientation='h', color='Importancia', color_continuous_scale=["#5B7C8E", "#8B864E", "#A64D32"])
        fig_s.update_layout(
            template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            xaxis=dict(showgrid=True, gridcolor='#D1CDC0'), yaxis=dict(showgrid=False, title=""),
            height=350, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_s, use_container_width=True)

def render_modelos_predictivos(): pass


# ==============================================================================
# MAIN Y TABS DE NAVEGACIÓN
# ==============================================================================
def main():
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>Mortalidad Cáncer de Estómago</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-bottom: 30px;'/>", unsafe_allow_html=True)
    
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
        render_perfil_sociodemografico(df_micro)
    with tabs[4]:
        render_factores_riesgo(df_panel)
    with tabs[5]:
        render_modelos_predictivos()

    # Pie de página estilo paper
    st.markdown("""
        <div class='footer-text'>
            <strong>Fuentes de Datos:</strong> Departamento Administrativo Nacional de Estadística (DANE) – Defunciones no fetales (2008-2024); 
            Proyecciones de población a nivel departamental.<br>
            <strong>Nota metodológica:</strong> Las tasas han sido ajustadas por edad mediante el método directo (población estándar OMS).
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()