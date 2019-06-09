setwd("~/mo826/machine-learning")
base = read.csv('dengue-ml-features-plus-labels.data.csv')

cor(base$weekofyear,base$total_cases)
cor(base$year, base$total_cases)
cor(base$ndvi_ne, base$total_cases)
cor(base$ndvi_nw, base$total_cases)
cor(base$ndvi_se, base$total_cases)
cor(base$ndvi_sw, base$total_cases)
cor(base$precipitation_amt_mm, base$total_cases)
cor(base$reanalysis_air_temp_k, base$total_cases)
cor(base$reanalysis_avg_temp_k, base$total_cases)
cor(base$reanalysis_dew_point_temp_k, base$total_cases)
cor(base$reanalysis_max_air_temp_k, base$total_cases)
cor(base$reanalysis_min_air_temp_k, base$total_cases)
cor(base$reanalysis_precip_amt_kg_per_m2, base$total_cases)
cor(base$reanalysis_relative_humidity_percent, base$total_cases)
cor(base$reanalysis_sat_precip_amt_mm, base$total_cases)
cor(base$reanalysis_specific_humidity_g_per_kg, base$total_cases)
cor(base$reanalysis_tdtr_k, base$total_cases)
cor(base$station_avg_temp_c, base$total_cases)
cor(base$station_diur_temp_rng_c, base$total_cases)
cor(base$station_max_temp_c, base$total_cases)
cor(base$station_min_temp_c, base$total_cases)
cor(base$station_precip_mm, base$total_cases)

cor(base$total_cases, base$total_cases)


PCA_base = princomp(base, cor = FALSE, scores = TRUE)
PCA_base$sdev
PCA_base$loadings.print
print("geste")
PCA_base$scores
print(PCA_base)
print(PCA_base$scores)


regressor = lm(formula = total_cases ~ ., data = base)
summary(regressor)
regressor = lm(formula = total_cases ~ station_min_temp_c+station_diur_temp_rng_c+reanalysis_tdtr_k+reanalysis_min_air_temp_k+weekofyear+reanalysis_air_temp_k, data = base)
summary(regressor)
regressor = lm(formula = total_cases ~ weekofyear+reanalysis_min_air_temp_k+reanalysis_tdtr_k, data = base)
summary(regressor)


base_selected = read.csv('dengue-ml-selected-features-fixed.data.csv')
install.packages('RoughSets')
library(RoughSets)
dt_base_selected = SF.asDecisionTable(dataset=base_selected,decision.attr=4)
intervalos = D.discretization.RST(dt_base_selected,nOfIntervals=3)
dt_base_selected2 = SF.applyDecTable(dt_base_selected,intervalos)
y = b0 + b1x1 
nofcases_selected = SF.asDecisionTable(dataset=base_selected)
intervalos2 = D.discretization.RST(nofcases_selected,nOfIntervals=5)

