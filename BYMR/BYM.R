# ==============================================================================
# 0. LIBRERÍAS
# ==============================================================================
library(INLA)
library(nanoparquet)
library(dplyr)
library(readr)
library(scales)

# ==============================================================================
# 1. CARGAR Y PREPARAR DATOS
# ==============================================================================
panel <- read_parquet("BYM/panel_dpto_año.parquet")

panel_inla <- panel %>%
  filter(!is.na(poblacion_total), !is.na(muertes_total)) %>%
  
  # Índice temporal
  mutate(id_tiempo = año - min(año) + 1) %>%
  
  # Tasa de referencia por año y offset E
  group_by(año) %>%
  mutate(
    tasa_ref = sum(muertes_total, na.rm = TRUE) / sum(poblacion_total, na.rm = TRUE),
    E = poblacion_total * tasa_ref
  ) %>%
  ungroup() %>%
  
  # Imputar NAs en irca_global ANTES de escalar
  group_by(cod_dpto) %>%
  mutate(
    irca_global = ifelse(is.na(irca_global),
                         mean(irca_global, na.rm = TRUE),
                         irca_global)
  ) %>%
  ungroup() %>%
  
  # Escalar covariables DESPUÉS de imputar e índice espacial
  arrange(cod_dpto, año) %>%
  mutate(
    irca_z     = as.numeric(scale(irca_global)),
    tabaco_z   = as.numeric(scale(prevalencia_tabaco)),
    id_espacio = as.integer(factor(cod_dpto))
  )

# Verificar que no hay NAs en variables del modelo
vars_modelo <- c("muertes_total", "E", "id_espacio", "id_tiempo", "irca_z", "tabaco_z")
cat("NAs en variables del modelo:\n")
print(colSums(is.na(panel_inla[, vars_modelo])))

stopifnot(
  !any(is.na(panel_inla[, vars_modelo])),
  !any(panel_inla$E <= 0)
)
cat("✅ Sin NAs — listo para modelar\n\n")

# ==============================================================================
# 2. MODELADO Y COMPARACIÓN
# ==============================================================================
# Modelo completo (con IRCA)
formula_bym <- muertes_total ~ 1 + irca_z + tabaco_z +
  f(id_espacio, model = "bym2", graph = "BYM/colombia.graph", scale.model = TRUE) +
  f(id_tiempo,  model = "ar1")

modelo <- inla(
  formula_bym,
  family  = "poisson",
  data    = panel_inla,
  E       = panel_inla$E,
  control.predictor = list(compute = TRUE),
  control.compute   = list(dic = TRUE, waic = TRUE, cpo = TRUE)
)

# Modelo sin IRCA (al no ser significativo)
formula_sin_irca <- muertes_total ~ 1 + tabaco_z +
  f(id_espacio, model = "bym2", graph = "BYM/colombia.graph", scale.model = TRUE) +
  f(id_tiempo,  model = "ar1")

modelo_sin_irca <- inla(
  formula_sin_irca,
  family  = "poisson",
  data    = panel_inla,
  E       = panel_inla$E,
  control.predictor = list(compute = TRUE),
  control.compute   = list(dic = TRUE, waic = TRUE, cpo = TRUE)
)

cat("--- COMPARACIÓN DE MODELOS ---\n")
cat("DIC modelo completo:", modelo$dic$dic, "\n")
cat("DIC sin irca:       ", modelo_sin_irca$dic$dic, "\n\n")

# ==============================================================================
# 3. DIAGNÓSTICO DEL MODELO SELECCIONADO (Sin IRCA)
# ==============================================================================
cat("--- DIAGNÓSTICO MODELO SIN IRCA ---\n")
cat("DIC saturado:", modelo_sin_irca$dic$dic.sat, "\n")

# Análisis de CPO
cpo_vals <- modelo_sin_irca$cpo$cpo
cat("CPOs problemáticos (< 0.01):", sum(cpo_vals < 0.01, na.rm = TRUE), "\n")
cat("CPOs con NA:", sum(is.na(cpo_vals)), "\n")

# Posterior Predictive Check (PPC)
obs <- panel_inla$muertes_total
fit <- modelo_sin_irca$summary.fitted.values$mean * panel_inla$E
cat("Correlación obs vs fitted:", round(cor(obs, fit), 3), "\n")
cat("Media obs:", round(mean(obs), 2), "| Media fitted:", round(mean(fit), 2), "\n\n")

# ==============================================================================
# 4. EXTRACCIÓN DE RESULTADOS HISTÓRICOS
# ==============================================================================
panel_inla <- panel_inla %>%
  mutate(
    # SMR suavizado
    SMR      = modelo_sin_irca$summary.fitted.values$mean,
    SMR_low  = modelo_sin_irca$summary.fitted.values$`0.025quant`,
    SMR_high = modelo_sin_irca$summary.fitted.values$`0.975quant`,
    
    # Tasas (por 100,000 hab)
    tasa_observada      = (muertes_total / poblacion_total) * 100000,
    tasa_suavizada      = (SMR * E / poblacion_total) * 100000,
    tasa_suavizada_low  = (SMR_low * E / poblacion_total) * 100000,
    tasa_suavizada_high = (SMR_high * E / poblacion_total) * 100000
  )

cat("--- RESUMEN POR DEPARTAMENTO (Top 33) ---\n")
panel_inla %>%
  group_by(departamento) %>%
  summarise(
    tasa_obs_media       = round(mean(tasa_observada), 2),
    tasa_suavizada_media = round(mean(tasa_suavizada), 2),
    SMR_medio            = round(mean(SMR), 3),
    IC_low               = round(mean(SMR_low), 3),
    IC_high              = round(mean(SMR_high), 3)
  ) %>%
  arrange(desc(SMR_medio)) %>%
  print(n = 33)

cat("\n--- TENDENCIA NACIONAL ANUAL ---\n")
panel_inla %>%
  group_by(año) %>%
  summarise(
    tasa_obs       = round(mean(tasa_observada), 2),
    tasa_suavizada = round(mean(tasa_suavizada), 2)
  ) %>%
  print()

# ==============================================================================
# 5. PREDICCIONES PARA 2025
# ==============================================================================
ultimo_año <- max(panel_inla$año)

# Crear estructura para 2025
future_2025 <- panel_inla %>%
  filter(año == ultimo_año) %>%
  mutate(
    año            = 2025,
    id_tiempo      = max(panel_inla$id_tiempo) + 1,
    muertes_total  = NA,
    tasa_observada = NA
  )

# Tasa de referencia 2024 para cálculo del E en 2025
tasa_ref_2024 <- panel_inla %>%
  filter(año == ultimo_año) %>%
  summarise(tasa = sum(muertes_total) / sum(poblacion_total)) %>%
  pull(tasa)

# Consolidar panel predictivo
panel_pred <- bind_rows(panel_inla, future_2025) %>%
  mutate(
    E = ifelse(año == 2025, poblacion_total * tasa_ref_2024, E),
    tabaco_z = as.numeric(scale(prevalencia_tabaco))
  )

# Correr modelo predictivo
modelo_pred <- inla(
  formula_sin_irca, # Usamos la misma fórmula ganadora
  family  = "poisson",
  data    = panel_pred,
  E       = panel_pred$E,
  control.predictor = list(compute = TRUE, link = 1),
  control.compute   = list(dic = TRUE, waic = TRUE)
)

# Extraer predicciones usando índices robustos
idx_2025 <- which(panel_pred$año == 2025)

pred_2025 <- panel_pred %>%
  filter(año == 2025) %>%
  mutate(
    SMR_pred      = modelo_pred$summary.fitted.values$mean[idx_2025],
    SMR_pred_low  = modelo_pred$summary.fitted.values$`0.025quant`[idx_2025],
    SMR_pred_high = modelo_pred$summary.fitted.values$`0.975quant`[idx_2025],
    muertes_pred  = round(SMR_pred * E),
    tasa_pred     = (SMR_pred * E / poblacion_total) * 100000
  ) %>%
  select(departamento, muertes_pred, tasa_pred, SMR_pred, SMR_pred_low, SMR_pred_high) %>%
  arrange(desc(tasa_pred))

cat("\n--- PREDICCIONES 2025 POR DEPARTAMENTO ---\n")
print(pred_2025, n = 33)

cat("\n--- TOTAL MUERTES PREDICHAS (NACIONAL 2025) ---\n")
pred_2025 %>%
  summarise(
    total_muertes_pred      = sum(muertes_pred),
    total_muertes_pred_low  = round(sum(SMR_pred_low * filter(panel_pred, año == 2025)$E)),
    total_muertes_pred_high = round(sum(SMR_pred_high * filter(panel_pred, año == 2025)$E))
  ) %>%
  mutate(across(everything(), scales::comma)) %>%
  print()

# ==============================================================================
# 6. EXPORTAR RESULTADOS
# ==============================================================================
# Histórico
historico_bym <- panel_inla %>%
  select(departamento, cod_dpto, año, muertes_total, tasa_observada, 
         tasa_suavizada, SMR, SMR_low, SMR_high)
write_csv(historico_bym, "BYM/historico_bym_suavizado.csv")

# Predicciones Departamentales
write_csv(pred_2025, "BYM/predicciones_bym_2025_dpto.csv")

# Agregado Nacional
bym_nacional <- pred_2025 %>%
  summarise(
    modelo       = "BYM",
    muertes_pred = sum(muertes_pred),
    muertes_low  = round(sum(SMR_pred_low  * filter(panel_pred, año == 2025)$E)),
    muertes_high = round(sum(SMR_pred_high * filter(panel_pred, año == 2025)$E))
  )
write_csv(bym_nacional, "BYM/predicciones_bym_2025_nacional.csv")

cat("\n✅ Archivos exportados exitosamente.\n")s