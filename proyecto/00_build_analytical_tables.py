import pandas as pd
import numpy as np
from pathlib import Path
import itertools

# ==============================================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================================
BASE_DIR = Path("proyecto/data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
ANALYTICAL_DIR = BASE_DIR / "analytical"

def build_mortalidad_slim():
    """
    Paso 2.2: Genera la tabla slim de registros de mortalidad en formato parquet.
    """
    print("Iniciando procesamiento: Tabla slim de mortalidad (mortalidad_raw_slim.parquet)...")
    ruta_mortalidad = RAW_DIR / "mortalidad" / "mortalidad_estomago_colombia_2008_2024.parquet"
    
    if not ruta_mortalidad.exists():
        print(f"  [!] Archivo no encontrado: {ruta_mortalidad}. Saltando...")
        return

    # Se asume que el CSV tiene unas 85k filas, lo leemos
    df_mortalidad = pd.read_parquet(ruta_mortalidad)
    
    columnas_slim = [
        "Anio_Defuncion", "Mes_Defuncion", "Departamento_Residencia", "Sexo", 
        "Grupo_Edad_Detallado", "Nivel_Educativo", "Regimen_Salud", 
        "Area_Residencia", "Sitio_Defuncion", "Asistencia_Medica"
    ]
    
    # Filtrar solo las columnas de interés
    df_slim = df_mortalidad[columnas_slim].copy()
    
    # Exportar a Parquet
    ruta_salida = ANALYTICAL_DIR / "mortalidad_raw_slim.parquet"
    df_slim.to_parquet(ruta_salida, index=False)
    print(f"  [✔] Tabla slim creada exitosamente en: {ruta_salida}")


def calcular_tasas_ajustadas(df_muertes, df_poblacion_long):
    """
    Requisito 3: Cálculo de Tasa Ajustada por Edad (TAE) mediante Estandarización Directa.
    1. Mapea la mortalidad a grupos quinquenales usando Grupo_Edad_Detallado (o el que se le pase).
    2. Agrupa la población DANE en los mismos quinquenios.
    3. Cruza muertes/población para Tasa Específica x (cod_dpto, año, grupo_edad).
    4. Multiplica por el peso estándar de la OMS y colapsa (sumando).
    Retorna DF con: [cod_dpto, año, tasa_ajustada_edad]
    """
    print("  ...Calculando Tasas Ajustadas por Edad (Estandarización Directa - OMS).")
    
    # === 1. MAPEO Y LIMPIEZA DE MORTALIDAD ===
    mapeo_edades_mortalidad = {
        '00': '0-4', '01': '0-4', '02': '0-4', '03': '0-4', '04': '0-4',
        '05': '0-4', '06': '0-4', '07': '0-4', '08': '0-4',
        '09': '5-9', '10': '10-14', '11': '15-19', '12': '20-24',
        '13': '25-29', '14': '30-34', '15': '35-39', '16': '40-44',
        '17': '45-49', '18': '50-54', '19': '55-59', '20': '60-64',
        '21': '65-69', '22': '70-74', '23': '75-79', 
        '24': '80+', '25': '80+', '26': '80+', '27': '80+', '28': '80+'
    }
    
    # Excluir edad desconocida ('29')
    # Dado que los valores en el df_muertes crudo (ej: '7', '8', '9') vienen sin padding a veces:
    df_mort = df_muertes.copy()
    df_mort['Grupo_Edad_Detallado'] = df_mort['Grupo_Edad_Detallado'].astype(str).str.zfill(2)
    df_mort = df_mort[df_mort['Grupo_Edad_Detallado'] != '29']
    
    # Mapeo a quinquenios
    df_mort['grupo_edad'] = df_mort['Grupo_Edad_Detallado'].map(mapeo_edades_mortalidad)
    
    # Agrupar las muertes: sumamos a nivel (cod_dpto, año, grupo_edad)
    df_mort_edad = df_mort.groupby(['Departamento_Residencia', 'Anio_Defuncion', 'grupo_edad']).size().reset_index(name='muertes')
    df_mort_edad.rename(columns={'Departamento_Residencia': 'cod_dpto', 'Anio_Defuncion': 'año'}, inplace=True)
    
    # Cast tipos para cruzar seguros
    df_mort_edad['cod_dpto'] = pd.to_numeric(df_mort_edad['cod_dpto'], errors='coerce')
    df_mort_edad['año'] = pd.to_numeric(df_mort_edad['año'], errors='coerce')
    
    # === 2. AGRUPACIÓN Y LIMPIEZA DE POBLACIÓN ===
    # df_poblacion_long viene con 'genero_edad' que es como "Hombres_5", "Mujeres_1"...
    df_pop = df_poblacion_long.copy()
    
    # Extraemos la edad desde el nombre de la columna "Hombres_X"
    df_pop['edad_simple'] = df_pop['genero_edad'].str.split('_').str[-1].astype(int)
    
    # Asignar la edad_simple al grupo_edad quinquenal correspondiente
    bins = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 150]
    labels = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", 
              "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"]
    df_pop['grupo_edad'] = pd.cut(df_pop['edad_simple'], bins=bins, labels=labels)
    
    # Sumar la población por (cod_dpto, año, grupo_edad) colapsando el género 
    df_pop_edad = df_pop.groupby(['DP', 'AÑO', 'grupo_edad'], observed=True)['poblacion'].sum().reset_index()
    df_pop_edad.rename(columns={'DP': 'cod_dpto', 'AÑO': 'año'}, inplace=True)
    
    df_pop_edad['cod_dpto'] = pd.to_numeric(df_pop_edad['cod_dpto'], errors='coerce')
    df_pop_edad['año'] = pd.to_numeric(df_pop_edad['año'], errors='coerce')
    
    # === 3. CALCULAR TASA ESPECÍFICA POR QUINQUENIO ===
    # Cruzar mortalidad con población a nivel grupo etario
    df_tasas = pd.merge(df_pop_edad, df_mort_edad, on=['cod_dpto', 'año', 'grupo_edad'], how='left')
    df_tasas['muertes'] = df_tasas['muertes'].fillna(0)
    
    # Tasa_Especifica (por grupo) = Muertes / Poblacion
    df_tasas['tasa_especifica'] = df_tasas['muertes'] / df_tasas['poblacion']
    
    # === 4. APLICAR PESOS DE LA POBLACIÓN ESTÁNDAR (OMS) ===
    pesos_oms = {
        "0-4": 8.86, "5-9": 8.69, "10-14": 8.60, "15-19": 8.47,
        "20-24": 8.22, "25-29": 7.93, "30-34": 7.61, "35-39": 7.15,
        "40-44": 6.59, "45-49": 6.04, "50-54": 5.37, "55-59": 4.55,
        "60-64": 3.72, "65-69": 2.96, "70-74": 2.21, "75-79": 1.52,
        "80+": 1.54
    }
    
    # Convertimos los pesos a un dataframe
    df_pesos = pd.DataFrame(list(pesos_oms.items()), columns=['grupo_edad', 'peso_oms'])
    
    df_tasas = pd.merge(df_tasas, df_pesos, on='grupo_edad', how='left')
    
    # Se multiplica (Tasa_Especifica) * (Peso/100) * 100,000 para volver a una tasa x 100.000 habitual
    # o de una (Tasa_Especifica * 100_000) * (Peso/100)
    df_tasas['tasa_ajustada_edad_grupo'] = (df_tasas['tasa_especifica'] * 100000) * (df_tasas['peso_oms'] / 100.0)
    
    # === 5. COLAPSAR AL PANEL MAESTRO ===
    # Sumar todas las tasas ajustadas segmentadas de todos los quinquenios del dpto_año
    df_tae = df_tasas.groupby(['cod_dpto', 'año'])['tasa_ajustada_edad_grupo'].sum().reset_index()
    df_tae.rename(columns={'tasa_ajustada_edad_grupo': 'tasa_ajustada_edad'}, inplace=True)
    
    return df_tae


def build_panel_maestro():
    """
    Paso 2.1: Genera el panel balanceado (panel_dpto_año.parquet) a nivel departamento-año 
    desde 2008 hasta 2024.
    """
    print("\nIniciando procesamiento: Panel maestro (panel_dpto_año.parquet)...")
    
    # --- PROCESAMIENTO DE POBLACIÓN (DANE) ---
    ruta_poblacion = RAW_DIR / "poblacion" / "PROYECCIONES_UNIFICADAS_EDAD.xlsx"
    if not ruta_poblacion.exists():
        print(f"  [!] Archivo no encontrado: {ruta_poblacion}. Saltando cálculo del panel...")
        return
        
    df_pop_raw = pd.read_excel(ruta_poblacion)
    
    print("  ...Procesando datos DANE de población a formato largo mediante pd.wide_to_long / melt.")
    # El reto indicaba construir la lógica para pasarlos de columnas wide (Hombres_0, Mujeres_15) a long
    # Nos centramos en las columnas Hombres_X y Mujeres_X
    cols_mantener = ['DP', 'DPNOM', 'AÑO']
    # Extraemos solo columnas que contengan 'Hombres_' o 'Mujeres_'
    cols_edades = [col for col in df_pop_raw.columns if col.startswith('Hombres_') or col.startswith('Mujeres_')]
    
    df_pop_filtrado = df_pop_raw[cols_mantener + cols_edades].copy()
    
    df_pop_long = df_pop_filtrado.melt(
        id_vars=cols_mantener, 
        value_vars=cols_edades, 
        var_name='genero_edad', 
        value_name='poblacion'
    )
    
    # Extraer el sexo (Hombres / Mujeres)
    df_pop_long['Sexo'] = df_pop_long['genero_edad'].apply(lambda x: "HOMBRES" if "Hombres" in str(x) else "MUJERES")
    
    # Agrupar a nivel Departamento, Año
    df_pop_agg = df_pop_long.groupby(['DP', 'DPNOM', 'AÑO']).agg(
        poblacion_total=('poblacion', 'sum')
    ).reset_index()
    
    # Extraer subtotales hombres y mujeres haciendo pivote
    df_pop_sexo = df_pop_long.pivot_table(
        index=['DP', 'AÑO'], 
        columns='Sexo', 
        values='poblacion', 
        aggfunc='sum'
    ).reset_index()
    df_pop_sexo.rename(columns={'HOMBRES': 'poblacion_hombre', 'MUJERES': 'poblacion_mujer'}, inplace=True)
    
    df_poblacion = pd.merge(df_pop_agg, df_pop_sexo, on=['DP', 'AÑO'], how='left')

    df_poblacion.rename(columns={
        'DP': 'cod_dpto',
        'DPNOM': 'departamento',
        'AÑO': 'año'
    }, inplace=True)
    
    # Convertimos explícitamente a numérico
    df_poblacion['cod_dpto'] = pd.to_numeric(df_poblacion['cod_dpto'], errors='coerce')
    df_poblacion['año'] = pd.to_numeric(df_poblacion['año'], errors='coerce')
    df_poblacion = df_poblacion.dropna(subset=['cod_dpto', 'año'])
    df_poblacion['cod_dpto'] = df_poblacion['cod_dpto'].astype(int)
    df_poblacion['año'] = df_poblacion['año'].astype(int)
    
    # Base balanceada de departamentos que existen en la población
    codigos_dpto = df_poblacion[['cod_dpto', 'departamento']].drop_duplicates(subset=['cod_dpto'])
    anios = list(range(2008, 2025))
    panel_base = pd.DataFrame(list(itertools.product(codigos_dpto['cod_dpto'], anios)), columns=['cod_dpto', 'año'])
    panel_base = pd.merge(panel_base, codigos_dpto, on='cod_dpto', how='left')

    # Unir variables poblacionales al panel
    df_panel = pd.merge(panel_base, df_poblacion.drop(columns=['departamento']), on=['cod_dpto', 'año'], how='left')
    
    # --- PROCESAMIENTO DE MORTALIDAD ---
    ruta_slim = ANALYTICAL_DIR / "mortalidad_raw_slim.parquet"
    if ruta_slim.exists():
        df_slim = pd.read_parquet(ruta_slim)
        print("  ...Agregando muertes (total, por sexo, urbano/rural).")
        
        # Sexo: '1' = Hombre, '2' = Mujer
        df_slim['es_hombre'] = np.where(df_slim['Sexo'] == '1', 1, 0)
        df_slim['es_mujer'] = np.where(df_slim['Sexo'] == '2', 1, 0)
        # Area: '1' = Urbano (Cabecera), '2' y '3' = Rural o centros poblados
        df_slim['es_urbano'] = np.where(df_slim['Area_Residencia'] == '1', 1, 0)
        df_slim['es_rural'] = np.where(df_slim['Area_Residencia'].isin(['2', '3']), 1, 0)
        
        df_mort_agg = df_slim.groupby(['Departamento_Residencia', 'Anio_Defuncion']).agg(
            muertes_total=('Anio_Defuncion', 'count'),
            muertes_hombre=('es_hombre', 'sum'),
            muertes_mujer=('es_mujer', 'sum'),
            muertes_urbano=('es_urbano', 'sum'),
            muertes_rural=('es_rural', 'sum')
        ).reset_index().rename(columns={'Departamento_Residencia': 'cod_dpto', 'Anio_Defuncion': 'año'})
        
        # Cast a int para evitar problemas de tipos en el merge
        df_mort_agg['cod_dpto'] = pd.to_numeric(df_mort_agg['cod_dpto'], errors='coerce')
        df_mort_agg['año'] = pd.to_numeric(df_mort_agg['año'], errors='coerce')
        df_mort_agg = df_mort_agg.dropna(subset=['cod_dpto', 'año'])
        df_mort_agg['cod_dpto'] = df_mort_agg['cod_dpto'].astype(int)
        df_mort_agg['año'] = df_mort_agg['año'].astype(int)
        
        # Merge al panel general
        df_panel = pd.merge(df_panel, df_mort_agg, on=['cod_dpto', 'año'], how='left')
        
        # Rellenar con 0 si no hubo muertes en un depto para el año dado
        for col in ['muertes_total', 'muertes_hombre', 'muertes_mujer', 'muertes_urbano', 'muertes_rural']:
            df_panel[col] = df_panel[col].fillna(0)
    else:
        print("  [!] No existe el archivo slim de mortalidad para agregar. Asegurese de procesarlo.")
        
    
    # Cálculo de Tasas
    df_panel['tasa_cruda'] = (df_panel['muertes_total'] / df_panel['poblacion_total']) * 100000
    
    # Enviar df_pop_long y df_base (mortalidad no slim sino df original) al cálculo ajustado
    df_mort_bruta = pd.read_parquet(RAW_DIR / "mortalidad" / "mortalidad_estomago_colombia_2008_2024.parquet")
    # Asegúrate de pasar el dataframe en bruto de mortalidad donde está "Grupo_Edad_Detallado"
    df_tae = calcular_tasas_ajustadas(df_mort_bruta, df_pop_long)
    
    # Unir la tasa calculada a nuestro panel general
    df_panel = pd.merge(df_panel, df_tae, on=['cod_dpto', 'año'], how='left')

    # --- PROCESAMIENTO DE IRCA ---
    ruta_irca = RAW_DIR / "irca" / "IRCA_DPTO.csv"
    if ruta_irca.exists():
        df_irca = pd.read_csv(ruta_irca)
        # Renombramos para cruzar
        df_irca.rename(columns={
            'DepartamentoCodigo': 'cod_dpto',
            'Año': 'año',
            'IRCA': 'irca_global',
            'IRCAurbano': 'irca_urbano',
            'IRCArural': 'irca_rural',
            'Nivel de riesgo': 'nivel_riesgo_agua'
        }, inplace=True)
        # Limpiar las comas en los años de IRCA (ej: '2,024' -> 2024)
        df_irca['año'] = df_irca['año'].astype(str).str.replace(',', '')
        
        # Convertir año a numérico para evitar errores en merge
        df_irca['año'] = pd.to_numeric(df_irca['año'], errors='coerce')
        df_irca['cod_dpto'] = pd.to_numeric(df_irca['cod_dpto'], errors='coerce')
        df_irca = df_irca.dropna(subset=['año', 'cod_dpto'])
        df_irca['año'] = df_irca['año'].astype(int)
        df_irca['cod_dpto'] = df_irca['cod_dpto'].astype(int)
        
        # Seleccionamos las que importan
        cols_irca = ['cod_dpto', 'año', 'irca_global', 'irca_urbano', 'irca_rural', 'nivel_riesgo_agua']
        df_panel = pd.merge(df_panel, df_irca[cols_irca], on=['cod_dpto', 'año'], how='left')
    
    # --- PROCESAMIENTO DE ENCSPA (Consumo Tabaco 2019) ---
    ruta_encspa = RAW_DIR / "encspa" / "Tabaco.xlsx"
    if ruta_encspa.exists():
        df_encspa = pd.read_excel(ruta_encspa)
        # Renombrar y seleccionar
        df_encspa = df_encspa.rename(columns={
            "Código DANE": "cod_dpto", 
            "Prevalencia Vida (%)": "prevalencia_tabaco"
        })[['cod_dpto', 'prevalencia_tabaco']]
        # Convertir y limpiar
        df_encspa['cod_dpto'] = pd.to_numeric(df_encspa['cod_dpto'], errors='coerce')
        df_encspa = df_encspa.dropna(subset=['cod_dpto'])
        df_encspa['cod_dpto'] = df_encspa['cod_dpto'].astype(int)
        
        # Merge Broadcast
        df_panel = pd.merge(df_panel, df_encspa, on='cod_dpto', how='left')

    # =========================================================================
    # Guardado Final
    # =========================================================================
    ruta_salida = ANALYTICAL_DIR / "panel_dpto_año.parquet"
    df_panel.to_parquet(ruta_salida, index=False)
    
    print(f"  [✔] Panel Maestro balanceado listo en {ruta_salida}")



if __name__ == "__main__":
    print("=== SCRIPT DE CONSTRUCCIÓN DE TABLAS ANALÍTICAS ===")
    
    # Asegurar que las carpetas existan operativamente
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ANALYTICAL_DIR.mkdir(parents=True, exist_ok=True)
    
    build_mortalidad_slim()
    build_panel_maestro()
    
    print("\n=== FINALIZADO ===")
