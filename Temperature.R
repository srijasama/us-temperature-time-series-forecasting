# Group 6 Final R Code - Environmental Forecasting for USA
# --------------------------------------------------
# Purpose: Forecasting Temperature Trends and Exploring Environmental Impact

# 1. Load Required Libraries
# --------------------------------------------------
library(tidyverse)
library(lubridate)
library(forecast)
library(fable)
library(tsibble)
library(readr)
library(caret)
library(GGally)
library(zoo)

# 2. Load and Inspect Dataset
# --------------------------------------------------
data <- read_csv("temperature.csv")
glimpse(data)
summary(data)
sum(is.na(data))

# 3. Filter for USA Data (2000–2023)
# --------------------------------------------------
usa_data <- data %>% 
  filter(Country == "USA") %>% 
  arrange(Year)

# Fill missing years for temperature and interpolate
usa_yearly <- usa_data %>%
  group_by(Year) %>%
  summarise(Avg_Temperature_degC = mean(Avg_Temperature_degC, na.rm = TRUE))

full_years <- tibble(Year = 2000:2023)
usa_full <- left_join(full_years, usa_yearly, by = "Year")
usa_full$Avg_Temperature_degC <- na.approx(usa_full$Avg_Temperature_degC, na.rm = FALSE)

# Create a tsibble for forecasting
usa_ts <- usa_full %>%
  mutate(Year = year(ymd(paste0(Year, "-01-01")))) %>%
  as_tsibble(index = Year)

# Convert to time series
temperature_ts <- ts(usa_full$Avg_Temperature_degC, start = 2000, frequency = 1)

# Fit ARIMA(1,1,1) model
arima_model <- Arima(temperature_ts, order = c(1,1,1))
arima_forecast <- forecast(arima_model, h = 6)

# Fit ETS model (additive trend)
ets_model <- ets(temperature_ts, model = "AAN")
ets_forecast <- forecast(ets_model, h = 6)

# Define forecast years
forecast_years <- 2024:2029

# 4. Time Series Plot of Temperature
# --------------------------------------------------
ggplot(usa_full, aes(x = Year, y = Avg_Temperature_degC)) +
  geom_line(color = "blue", size = 1.1) +
  geom_point(color = "red", size = 2) +
  theme_minimal(base_size = 14) +
  labs(title = "USA: Average Temperature Over Time (2000–2023)",
       x = "Year", y = "Avg Temperature (°C)")

# 5. Histogram of Average Temperature
# --------------------------------------------------
ggplot(usa_data, aes(x = Avg_Temperature_degC)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "darkgreen", alpha = 0.7) +
  geom_density(aes(y = ..count..), color = "red", size = 1) +
  theme_minimal(base_size = 14) +
  labs(title = "USA: Distribution of Average Temperature (2000–2023)",
       x = "Avg Temperature (°C)", y = "Frequency")

# 6. Environmental Variables Time Series (Faceted)
# --------------------------------------------------
env_yearly <- usa_data %>%
  group_by(Year) %>%
  summarise(
    Avg_Temperature_degC = mean(Avg_Temperature_degC),
    CO2_Emissions_tons_per_capita = mean(CO2_Emissions_tons_per_capita),
    Rainfall_mm = mean(Rainfall_mm),
    Sea_Level_Rise_mm = mean(Sea_Level_Rise_mm),
    Renewable_Energy_pct = mean(Renewable_Energy_pct),
    Extreme_Weather_Events = mean(Extreme_Weather_Events)
  )

env_long <- env_yearly %>%
  pivot_longer(cols = -Year, names_to = "Variable", values_to = "Value")

ggplot(env_long, aes(x = Year, y = Value, color = Variable)) +
  geom_line(size = 0.8) +
  facet_wrap(~Variable, scales = "free_y", ncol = 2) +
  theme_minimal(base_size = 14) +  
  labs(title = "Environmental Indicators in the USA (2000–2023)",
       x = "Year", y = "Measured Value") +
  theme(legend.position = "none")

# 7. ARIMA and ETS Forecasting Combined Plot
# --------------------------------------------------

# Individual ARIMA Forecast Plot
# --------------------------------------------------
plot(temperature_ts, type = "o", pch = 19, col = "orange", lwd = 2,
     xlim = c(2000, 2029), ylim = c(5, 32),
     main = "ARIMA Forecast for USA Avg Temperature (2024–2029)",
     ylab = "Average Temperature (°C)", xlab = "Year")

lines(forecast_years, arima_forecast$mean, col = "red", lty = 2, lwd = 2)
points(forecast_years, arima_forecast$mean, col = "red", pch = 16)

legend("topright",
       legend = c("Observed", "ARIMA Forecast"),
       col = c("orange", "red"),
       pch = c(19, 16),
       lty = c(1, 2),
       lwd = 2)

# Individual ETS Forecast Plot
# --------------------------------------------------
plot(temperature_ts, type = "o", pch = 19, col = "orange", lwd = 2,
     xlim = c(2000, 2029), ylim = c(5, 32),
     main = "ETS Forecast for USA Avg Temperature (2024–2029)",
     ylab = "Average Temperature (°C)", xlab = "Year")

lines(forecast_years, ets_forecast$mean, col = "blue", lty = 3, lwd = 2)
points(forecast_years, ets_forecast$mean, col = "blue", pch = 17)

legend("topright",
       legend = c("Observed", "ETS Forecast"),
       col = c("orange", "blue"),
       pch = c(19, 17),
       lty = c(1, 3),
       lwd = 2)

plot(temperature_ts, type = "o", pch = 19, col = "orange", lwd = 2,
     xlim = c(2000, 2029), ylim = c(5, 32),
     main = "Forecasting USA Average Temperature: ARIMA vs ETS",
     ylab = "Average Temperature (°C)", xlab = "Year")

lines(forecast_years, arima_forecast$mean, col = "orangered", lty = 2, lwd = 2)
points(forecast_years, arima_forecast$mean, col = "orangered", pch = 16)

lines(forecast_years, ets_forecast$mean, col = "blue", lty = 3, lwd = 2)
points(forecast_years, ets_forecast$mean, col = "blue", pch = 4)

legend("topright",
       legend = c("Observed", "ARIMA Forecast", "ETS Forecast"),
       col = c("orange", "orangered", "blue"),
       pch = c(19, 16, 4),
       lty = c(1, 2, 3),
       lwd = 2)


# 9. Regression Analysis + Multicollinearity Check
# --------------------------------------------------
regression_data <- usa_data %>%
  select(Avg_Temperature_degC, CO2_Emissions_tons_per_capita,
         Rainfall_mm, Renewable_Energy_pct, Population, Forest_Area_pct)

reg_model <- lm(Avg_Temperature_degC ~ ., data = regression_data)
summary(reg_model)

# 10. Multivariate Visualization
# --------------------------------------------------
ggpairs(regression_data, title = "USA: Relationships Between Environmental Variables")

