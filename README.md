# Electricity Price Prediction Based on Weather and Market Signals

---

## Table of Contents

1. [Setup and Navigation](#setup-and-navigation)
2. [Energy Markets - Background](#energy-markets--background)
3. [Why Weather Matters](#why-weather-matters)
4. [Project Motivation and Management Summary](#project-motivation-and-management-summary)
5. [Data Sources](#data-sources)
6. [Feature Engineering](#feature-engineering)
7. [Model Selection](#model-selection)
8. [Sprints and Discoveries](#sprints-and-discoveries)
9. [Results](#results)
10. [Further Development](#further-development)

---

## Setup and Navigation

### Required Libraries

Install all dependencies before running the notebooks:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost 
```

### Data Files Required

| File | Source | Description |
|------|--------|-------------|
| `Energy_Day-ahead_prices_Hour.csv` | SMARD.de | Hourly DE-LU day-ahead prices 2019-2024 |
| `Weather_open-meteo.csv` | Open-Meteo | Hourly weather data, central Germany |
| `ICE_Dutch_TTF_Natural_Gas_Futures_Historical_Data.csv` | Investing.com | Daily TTF gas prices 2019-2024 |

### Notebook Execution Order

The project is structured across four notebooks. They must be run in order as each notebook depends on outputs from the previous one.

**Notebook 1: `starter.ipynb`**
Start here. Loads raw data, merges all sources, engineers all features, and saves `df_final.csv` and `df_final_with_gas.csv` to disk. Run this notebook once before anything else.

**Notebook 2: `eda.ipynb`**
Exploratory data analysis. Load after running Notebook 1. No model training - only visualizations and statistical analysis. 

**Notebook 3: `weather_model.ipynb`**
Sprint 1 model. Trains Linear Regression, Random Forest, and XGBoost using weather and lag features only (no gas price). Documents the first iteration and its limitations.

**Notebook 4: `weather_and_gas_model.ipynb`**
Sprint 2 model. Extends Sprint 1 by adding TTF gas price. Includes hyperparameter tuning, feature importance analysis, leakage verification, and hit rate evaluation. This is the final best model.

---

## Energy Markets - Background

![European Energy Market](https://cdn.prod.website-files.com/65b3e159d25a6097b6ca5815/66ba21f9940a9a05212ac51f_66ba21c28aeb0b3a69666ab0_Intraday%2520market_In_text.png)

### How Electricity Markets Work

Electricity markets are fundamentally different from commodity markets for oil, gas, or metals. Electricity cannot be stored at scale - every megawatt-hour generated must be consumed at the moment of production. This physical constraint creates a highly dynamic market where prices can change dramatically from hour to hour.

The European electricity market is organized through a network of transmission system operators (TSOs) coordinated by ENTSO-E (European Network of Transmission System Operators for Electricity). Germany operates within the DE-LU bidding zone, a shared market with Luxembourg that reflects the integrated nature of Central European power systems.

### The Merit Order

Electricity pricing follows the **merit order** principle. Every hour, power plants bid their available capacity into the market, sorted from cheapest to most expensive:

1. Solar and wind (near-zero marginal cost)
2. Nuclear (low marginal cost)
3. Coal and lignite (moderate marginal cost)
4. Natural gas (high marginal cost)
5. Oil and emergency reserves (very high marginal cost)

The most expensive plant required to meet demand in a given hour sets the price for all producers in that hour. This means that even cheap renewable energy receives the same price as the expensive gas plant at the margin. Consequently, when gas prices rise, electricity prices rise - even when most generation comes from wind and solar.

### Trading Timeframes

**Day-ahead market (focus of this project)**
The primary market for electricity delivery. Prices are set through an auction that closes at 12:00 noon for each hour of the following day. This is the most liquid and transparent market, with prices published hourly by SMARD and ENTSO-E. It is the standard reference for electricity price forecasting in academic research and industry practice.

**Intraday market**
Continuous trading up to 5 minutes before physical delivery. Used to balance short-term deviations from day-ahead schedules caused by forecast errors in renewable generation or unexpected plant outages. More volatile and less predictable than day-ahead.

**Forward and futures markets**
Week-ahead, monthly, quarterly, and yearly contracts traded on exchanges such as EEX (European Energy Exchange). Used by large utilities and industrial consumers for long-term price risk management. Less granular - one price covers an entire delivery period.

This project focuses exclusively on the **day-ahead hourly market**, which is the most relevant timeframe for operational energy management, short-term trading decisions, and demand-side flexibility programs.

### Negative Prices

A notable feature of the German electricity market is the occurrence of negative prices - hours when the market price falls below zero euros per MWh. This means producers pay consumers to take electricity.

Negative prices arise when renewable generation exceeds demand and inflexible plants (nuclear, coal) cannot reduce output quickly enough. Rather than shutting down and incurring restart costs, operators accept negative prices to maintain grid balance. Since 2023, negative prices have become more frequent as Germany has rapidly expanded solar and wind capacity.

From a forecasting perspective, negative prices represent the most challenging prediction task because they are driven by sudden renewable overproduction events that are difficult to anticipate from weather data alone.

---

## Why Weather Matters

![German Wind and Solar](https://core-p-001.sitecorecontenthub.cloud/api/public/content/c84f8a8ee19f43218a72ee9c85b496aa?v=68ac84cf&t=LG1)

Weather is one of the most important drivers of short-term electricity price variation in Germany for two distinct reasons: it affects both **supply** through renewable generation and **demand** through heating and cooling requirements.

### Supply Side: Renewable Generation

Germany has one of the largest installed renewable energy capacities in the world. As of 2024, renewables account for approximately 59% of German electricity generation, with wind and solar as the dominant sources.

**Wind power** is the single most important renewable driver of electricity prices. Wind turbine output scales with the cube of wind speed - a doubling of wind speed produces eight times the power output. On windy days, wind farms inject large volumes of cheap electricity into the grid, pushing expensive gas plants out of the merit order and suppressing prices significantly. The relationship is clearly visible in the data: high wind speed hours consistently show lower prices, and the correlation between wind power proxy and price is the strongest of all weather features.

**Solar radiation** drives photovoltaic generation. On sunny summer afternoons, solar output peaks and depresses prices - sometimes dramatically. This creates the characteristic midday price dip visible in spring and summer months, and contributes to the phenomenon of negative prices when solar generation peaks on low-demand days such as public holidays.

### Demand Side: Temperature Effects

Temperature affects electricity demand through two opposing mechanisms. Cold temperatures increase demand for electric heating and heat pumps, raising prices. Hot temperatures increase demand for air conditioning, also raising prices. Mild temperatures in the range of 15 to 20 degrees Celsius correspond to minimum demand and lowest prices.

This U-shaped relationship between temperature and price is one reason why raw temperature is less informative than derived features. The project addresses this by engineering separate heating degree and cooling degree features that explicitly capture the two demand-driving directions.

### Other Important Factors

Weather is a necessary but not sufficient predictor of electricity prices. Several non-weather factors have significant influence:

**Natural gas prices** are the most important non-weather driver. Since gas plants typically set the marginal price in the merit order during low-renewable hours, the price of gas directly determines the floor price of electricity during such periods. The 2021 to 2023 energy crisis demonstrated this relationship dramatically - when Russian pipeline gas was cut and European gas prices multiplied tenfold, electricity prices followed immediately regardless of weather conditions.

**Geopolitical events** can cause sudden structural breaks in price levels. The Russian invasion of Ukraine in February 2022 created an overnight regime shift in European energy markets that no weather-based model could have anticipated.

**Carbon prices** (EU ETS allowances) directly increase the cost of coal and gas generation, flowing through to electricity prices.

**Cross-border electricity flows** affect German prices through interconnections with France (nuclear), Scandinavia (hydro), and neighboring markets.

**Grid events and plant outages** can cause sudden price spikes independent of weather.

---

## Project Motivation and Management Summary

### Personal Motivation

I work at one of Europe's largest energy companies, where my current project involves partnering with a tech giant to improve energy market forecasting capabilities. The intersection of machine learning and energy price prediction is directly relevant to my professional work - particularly the question of how much predictive power can be extracted from publicly available weather and market data, and where the fundamental limits of such models lie. 
I can validate or dismiss my own findings based on domain knowledge. 

### Why

European electricity markets are undergoing a fundamental transformation driven by the rapid growth of renewable energy. As wind and solar penetration increases, price volatility intensifies and the relationship between weather and price becomes more pronounced. The ability to forecast prices accurately has direct economic value for energy producers, industrial consumers, grid operators, and traders. Improving forecast accuracy by even a small margin translates to significant cost savings at the scale of European energy markets.

### What

This project investigates the predictive power of weather data, price history, and natural gas prices for German day-ahead electricity price forecasting. It addresses three specific research questions: How much of hourly electricity price variation can be explained by weather features alone? How much additional signal is provided by price lag features and gas prices? Which machine learning models are best suited to this prediction task given the non-linear, time-varying nature of electricity prices?

### How

The project follows the CRISP-DM methodology across two iterative sprints. Data from three public sources covering 2019 to 2024 was collected, merged, and enriched with engineered features. Three machine learning models were trained on 2019 to 2022 data, validated on 2023, and evaluated on the unseen test year 2024. Each sprint produced new findings that motivated the next iteration. The full workflow - from raw data to trained models - is reproducible through the four Jupyter notebooks in this repository.

---

## Data Sources

### 1. Electricity Prices - SMARD.de

**Source**: Federal Network Agency of Germany (Bundesnetzagentur), SMARD Market Data Platform  
**URL**: https://www.smard.de/en/downloadcenter/download-market-data  
**Coverage**: January 2019 to December 2024  
**Resolution**: Hourly  
**Variable**: Day-ahead price, DE-LU bidding zone, EUR/MWh  

SMARD is the official German energy market data portal operated by the Federal Network Agency. It provides the authoritative source for German electricity market data, directly sourced from ENTSO-E transparency reporting. The DE-LU bidding zone represents Germany and Luxembourg as a single pricing area, which is the standard reference for German electricity market research.

### 2. Weather Data - Open-Meteo

**Source**: Open-Meteo Historical Weather API (ERA5 reanalysis data from Copernicus Climate Change Service)  
**URL**: https://open-meteo.com/en/docs/historical-weather-api   
**Coverage**: January 2019 to December 2024  
**Resolution**: Hourly  
**Location**: geographic center of Germany  
**Variables**: Temperature at 2m (°C), wind speed at 10m (km/h), shortwave radiation (W/m²), precipitation (mm)  

### 3. Natural Gas Prices - Investing.com

**Source**: Investing.com, ICE Dutch TTF Natural Gas Futures  
**Coverage**: January 2019 to December 2024  
**Resolution**: Daily trading days  
**Variable**: Settlement price, EUR/MWh  

TTF (Title Transfer Facility) is the European benchmark natural gas price, traded on the Intercontinental Exchange (ICE). It is the standard reference for European gas pricing and is the primary input cost for gas-fired power plants that set the marginal electricity price during low-renewable hours. Daily gas prices were forward-filled to match the hourly electricity price data, reflecting that gas prices are constant within each trading day.

---

## Feature Engineering

Features are organized into four logical groups, each capturing a distinct driver of electricity price variation.

### Group 1: Weather Features

These features capture the supply-side impact of weather on renewable electricity generation.

| Feature | Derivation | Physical Meaning |
|---------|-----------|-----------------|
| wind_power | wind_speed³ | Proportional to wind turbine power output (cubic relationship from turbine physics) |
| solar_proxy | radiation.clip(lower=0) | Solar PV generation proxy, clipped to remove any negative values |
| precipitation | raw mm | Proxy for cloud cover and reduced solar output |

Raw wind speed is transformed to its cube because wind turbine power output scales with the cube of wind speed according to the Betz law - doubling wind speed produces eight times the power. This domain-specific transformation makes the feature more physically meaningful and improves model performance compared to using raw wind speed.

### Group 2: Temperature Demand Features

Rather than using raw temperature, two derived features encode the heating and cooling demand that temperature creates.

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| heating_degree | max(18 - temperature, 0) | Degrees below 18°C - proxy for heating demand |
| cooling_degree | max(temperature - 24, 0) | Degrees above 24°C - proxy for cooling/AC demand |

This decomposition addresses the U-shaped relationship between temperature and electricity demand. A single temperature variable has low linear correlation with price because both cold and hot temperatures raise demand. The two derived features make this non-linear relationship explicit and linear, improving performance particularly for the Linear Regression model.

### Group 3: Calendar Features

These features capture the temporal patterns in electricity demand driven by human activity schedules.

| Feature | Values | Physical Meaning |
|---------|--------|-----------------|
| hour | 0-23 | Hour of day - captures intraday demand cycle |
| month | 1-12 | Month of year - captures seasonal patterns |
| is_weekend | 0 or 1 | Weekend indicator - industrial demand is lower on weekends |

The hour feature captures the strong intraday price pattern: cheap at night (low demand), rising in the morning as industry starts, peaking in the evening after solar generation drops. The month feature captures seasonal heating and cooling cycles. The weekend indicator captures the significant demand reduction on Saturdays and Sundays when industrial and commercial electricity consumption falls substantially.

### Group 4: Lag Features

Lag features provide the model with information about recent price history, capturing price momentum and regime persistence.

| Feature | Derivation | Economic Meaning |
|---------|-----------|-----------------|
| price_lag_24 | price shifted 24 hours | Same hour yesterday - captures day-to-day price momentum |
| price_lag_168 | price shifted 168 hours | Same hour last week - captures weekly demand cycle in price terms |

Lag features are legitimate in a day-ahead forecasting context because yesterday's prices are always known when predicting tomorrow. The 24-hour lag captures short-term price momentum - if gas prices spiked yesterday, they are likely elevated today. The 168-hour lag (7 days × 24 hours) captures the weekly demand cycle: Monday 8am tends to be similar to the previous Monday 8am in terms of industrial activity patterns.

The introduction of lag features was the single largest improvement in model performance, lifting R² from negative values to approximately 0.45-0.57 across all models.

### Group 5: Gas Price

| Feature | Source | Economic Meaning |
|---------|--------|-----------------|
| gas_price | TTF daily futures, EUR/MWh | Marginal cost of gas-fired generation - sets the price floor during low-renewable hours |

Gas price was added in Sprint 2 after Sprint 1 revealed systematic underprediction of absolute price levels. Because gas plants typically set the marginal price in the merit order, the gas price directly determines the baseline electricity price level that weather features cannot capture. Adding this feature improved validation R² from 0.537 to 0.637.

---

## Model Selection

Three models were chosen to represent a progression from simple to complex, enabling systematic comparison and providing a clear narrative of methodological choices.

### Linear Regression

Linear Regression assumes that the target variable is a weighted linear combination of the input features. It is the simplest possible model and serves as the baseline against which more sophisticated models are compared.

Despite its simplicity, Linear Regression has a specific advantage in this problem: it can extrapolate beyond the range of training data. When 2023 prices were systematically higher than 2019-2022 training prices, Linear Regression maintained reasonable predictions by extrapolating its learned weights. Tree-based models, by contrast, cannot predict values outside the range seen during training.

Linear Regression requires feature scaling because it assigns weights to features directly. Features with large numerical ranges (such as wind_power, which can reach 100,000) would dominate over binary features (such as is_weekend) without normalization. StandardScaler was applied to training data and the same transformation applied to validation and test sets.

### Random Forest

Random Forest constructs many decision trees in parallel, each trained on a random subset of the training data and a random subset of features. The final prediction is the average across all trees. This ensemble approach reduces overfitting compared to a single decision tree while maintaining the ability to capture non-linear relationships.

Random Forest is well-suited to this problem because electricity price relationships are inherently non-linear - the merit order creates threshold effects, and the interaction between wind, solar, and time-of-day creates complex joint effects that linear models cannot capture. Tree-based models handle these naturally through hierarchical splits.

An important property of Random Forest is that it does not require feature scaling. Trees split on thresholds rather than distances, so the absolute scale of features does not affect the result.

### XGBoost

XGBoost (Extreme Gradient Boosting) constructs trees sequentially, where each tree is trained to correct the errors of the previous ensemble. This boosting approach typically produces better performance than bagging (Random Forest) on tabular data problems because it directly optimizes the prediction error at each step.

XGBoost is the standard model for tabular regression problems in both industry and competitive machine learning. It handles non-linear relationships, feature interactions, and missing values natively. Its sequential construction also makes it more sensitive to the ordering and magnitude of errors, which is relevant for electricity prices where extreme values (during crisis periods) should be learned carefully.

Hyperparameter tuning via GridSearchCV with TimeSeriesSplit cross-validation identified `max_depth=4` as optimal, reducing overfitting compared to the default `max_depth=6`. TimeSeriesSplit was used instead of standard k-fold cross-validation to preserve temporal order within the training set - standard k-fold would mix future observations into training folds, constituting data leakage.

### Why Not Neural Networks

Deep learning models were considered but excluded for several well-reasoned grounds. The dataset contains approximately 35,000 training observations - a size at which tree-based models consistently match or outperform neural networks on tabular data. Neural networks require substantially more hyperparameter tuning (architecture, learning rate, dropout, batch size), making them harder to interpret and defend. Feature importance, which is a valuable analytical output in this project, is straightforward for tree models but requires additional techniques such as SHAP for neural networks. Finally, the academic literature on electricity price forecasting consistently shows that gradient boosted trees achieve competitive performance on day-ahead markets without the complexity overhead of neural architectures.

---

## Sprints and Discoveries

This project followed the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology across two iterative sprints. Each sprint produced findings that directly motivated the design of the next iteration.

### Sprint 1: Weather-Only Model

**Focus**: Establish a baseline using only weather features, calendar features, and price lag features. Evaluate whether weather data carries meaningful predictive signal for day-ahead electricity prices.

**Key findings**:

The initial models trained without lag features produced negative R² values on validation data. Investigation revealed the cause: training data (2019-2021) had a mean price of approximately €55/MWh while validation data (2022) had a mean of €235/MWh - a fourfold increase driven by the energy crisis. A weather-only model cannot extrapolate to this price regime shift.

Adding price lag features (lag_24 and lag_168) resolved this problem by providing the model with the current price level as context. R² improved from negative values to 0.45-0.57 across all three models.

Feature importance analysis revealed that `price_lag_24` accounted for approximately 78% of XGBoost's predictive power. Wind power was the most important weather feature. This finding raised the question of whether the model had learned genuine price dynamics or was simply predicting "tomorrow will be like today."

A leakage check removed `price_lag_24` and showed R² dropped from 0.455 to -0.263 on validation - a large but not complete collapse. The remaining features (weather, calendar, lag_168) still produced positive test R² of 0.128, confirming they carry genuine signal. The lag feature is legitimate rather than leakage because yesterday's prices are genuinely available at prediction time in day-ahead forecasting.

**Conclusion from Sprint 1**: Weather features explain intraday price patterns well but cannot explain the absolute price level, which varies across years with fuel market conditions. Gas price was identified as the key missing feature.

### Sprint 2: Adding Gas Price and Hyperparameter Tuning

**Focus**: Add TTF natural gas price to the feature set and systematically tune XGBoost hyperparameters.

**Key findings**:

Adding TTF gas price as a daily feature (forward-filled to hourly) improved validation R² from 0.537 to 0.637 - the largest single improvement in Sprint 2. The feature importance distribution shifted: gas_price took approximately 10% of importance from all other features, confirming it provides genuinely new information beyond what price lags already capture.

Visual inspection of predictions confirmed the mechanism: the systematic underprediction of absolute price levels observed in Sprint 1 was largely resolved. The model now correctly tracks both the shape of intraday price movements and the level at which those movements occur.

Hyperparameter tuning via GridSearchCV identified `max_depth=4` as optimal, compared to the default `max_depth=6`. Shallower trees reduced overfitting and improved test R² from 0.585 to 0.613. The improvement was modest but consistent across both validation and test sets, confirming that simpler trees generalize better for this dataset size and feature set.

Hit rate analysis revealed that hourly directional accuracy is a misleading metric for this problem - a naive model predicting the average price for each hour of the day scored 81.1%, higher than XGBoost at 75.3%. This is because the daily price cycle (prices rise in the morning, fall at night) is so consistent that it dominates the directional accuracy metric. A more meaningful metric compares the same hour across consecutive days, asking whether today's 8am price is higher or lower than yesterday's 8am price.

**Conclusion from Sprint 2**: Gas price is the critical missing feature for absolute price level prediction. Hyperparameter tuning provides incremental improvement. The final tuned XGBoost achieves test R² of 0.613, explaining 61.3% of hourly electricity price variation on unseen 2024 data.

---

## Evaluation Metrics
 
Three complementary metrics were used to evaluate model performance. Each captures a different aspect of prediction quality and together they provide a complete picture of how well the model performs in both statistical and practical terms.
 
### Mean Absolute Error (MAE)
 
MAE is the average absolute difference between predicted and actual prices across all hours in the evaluation set:
 
```
MAE = mean(|actual - predicted|)
```
 
A MAE of €21/MWh means the model's predictions are on average €21 away from the actual price. MAE is expressed in the same unit as the target variable (EUR/MWh), making it directly interpretable in business terms. It treats all errors equally regardless of their sign or magnitude.
 
MAE was chosen as the primary point estimate metric because it is intuitive for stakeholders without a statistical background and is robust to the extreme price outliers that occur during crisis periods. A single hour at €800/MWh inflates error metrics that weight large errors more heavily (such as RMSE) but has limited impact on MAE.
 
### Root Mean Squared Error (RMSE)
 
RMSE is the square root of the mean squared error:
 
```
RMSE = sqrt(mean((actual - predicted)²))
```
 
Squaring the errors before averaging means that large errors are penalized more than small errors. RMSE is always greater than or equal to MAE - the larger the gap between them, the more the model struggles with extreme price events.
 
In the context of this project, the difference between MAE and RMSE reveals the model's sensitivity to crisis-period outliers. An RMSE significantly higher than MAE indicates that while typical predictions are reasonable, the model makes large errors during extreme price events - which is consistent with the finding that weather features cannot explain geopolitically-driven price spikes.
 
### R-Squared (R²)
 
R² measures how much of the total price variation the model explains compared to a naive baseline that always predicts the mean price:
 
```
R² = 1 - (sum of squared residuals / total sum of squares)
```
 
Intuitively, R² answers the question: by what percentage does the model reduce prediction error compared to always guessing the average price?
 
| R² Value | Interpretation |
|----------|---------------|
| 1.0 | Perfect predictions |
| 0.613 | Model explains 61.3% of price variation |
| 0.0 | No better than predicting the mean price every hour |
| Negative | Worse than predicting the mean price every hour |
 
R² is dimensionless and comparable across different datasets and time periods, making it the primary metric for comparing Sprint 1 and Sprint 2 results and for benchmarking against academic literature. A negative R² - observed in early Sprint 1 models - indicates the model was actively harmful relative to a trivial baseline, which motivated the feature engineering iterations described in the sprint section.
 
Academic literature on German day-ahead electricity price forecasting reports R² values of 0.6 to 0.8 for models with rich feature sets. The final tuned XGBoost achieves R² of 0.613, placing it at the lower end of the professional benchmark range - appropriate given the deliberately limited feature set used in this project.
 
### Hit Rate - Directional Accuracy
 
Hit rate is a practical trading metric that measures the percentage of hours for which the model correctly predicted whether the price would move up or down. Unlike MAE and R², which measure the accuracy of the predicted price level, hit rate measures the accuracy of the predicted price direction.
 
```
hit rate = proportion of hours where sign(actual change) == sign(predicted change)
```
 
Hit rate is relevant to trading because directional correctness is often more actionable than absolute price accuracy. A trader who knows that tomorrow evening will be more expensive than tomorrow morning can buy in the morning and sell in the evening without needing to know the exact prices involved.
 
A random model that guesses direction with a coin flip achieves 50% hit rate. Any model must exceed 50% to demonstrate genuine directional skill. In professional energy trading, hit rates above 60% are considered good and above 65% are considered strong.
 
**An important methodological finding regarding hit rate**: The naive definition of directional accuracy - comparing each hour to the previous hour - proved misleading for this problem. A model that simply memorizes the average price for each hour of the day (always predicting that 7am is higher than 6am, that midnight is lower than 11pm) scored 81.1% on this metric, higher than XGBoost at 75.3%. This reveals that the metric was capturing the predictable daily price cycle rather than genuine model skill.
 
The correct definition compares the same hour across consecutive days - asking whether today's 8am price is higher or lower than yesterday's 8am price. This question is genuinely difficult because it requires understanding day-to-day variation driven by weather, gas prices, and demand patterns. The daily cycle cancels out because both the actual and predicted values are compared at the same point in the intraday cycle. This corrected metric provides a meaningful assessment of the model's practical trading value.
 
---

## Results

### Final Model Performance

| Model | Validation MAE | Validation R² | Test MAE | Test R² |
|-------|---------------|---------------|----------|---------|
| Linear Regression | 21.40 | 0.600 | 21.74 | 0.558 |
| Random Forest | 23.13 | 0.550 | 21.61 | 0.578 |
| XGBoost (default) | 20.76 | 0.637 | 21.09 | 0.585 |
| XGBoost (tuned) | 20.32 | 0.656 | 20.77 | 0.613 |

### Progressive Improvement Across Sprints

| Stage | Best Test R² | Key Addition |
|-------|-------------|-------------|
| Weather only, no lags | -0.080 | Baseline |
| Weather + lag features | 0.569 | price_lag_24, price_lag_168 |
| Weather + lags + gas | 0.585 | TTF gas price |
| Tuned XGBoost | 0.613 | Hyperparameter optimization |

### Interpretation

The tuned XGBoost model explains 61.3% of hourly electricity price variation on completely unseen 2024 data. 

---

## Further Development

The following extensions are identified as the highest-value directions for improving model performance and practical applicability.

**Carbon price (EU ETS allowances)**: CO₂ allowance prices directly increase the marginal cost of coal and gas generation. Available free from Investing.com or the EU ETS registry. 

**Electricity demand (load) data**: ENTSO-E publishes hourly actual and forecast load data for Germany. Actual demand is a direct price driver - higher demand pushes more expensive plants into the merit order. This is among the strongest predictors in professional forecasting systems and its absence is the most significant gap in the current feature set.

**Wind and solar generation data**: Rather than using weather as a proxy for renewable generation, actual generation data from ENTSO-E would provide a more precise signal. Forecast generation data (published day-ahead) would be the operationally correct version for a real forecasting system.

**Cross-border electricity flows**: Germany's price is significantly influenced by imports from French nuclear and Norwegian hydro, and by export demand from neighboring markets. Net position data is available from ENTSO-E.

**Separate crisis and non-crisis regimes**: The structural break created by the energy crisis means that a single model trained across all periods must learn two fundamentally different price formation regimes simultaneously. A regime-switching approach - using a classification model to identify the current price regime and routing to a regime-specific regression model - could improve accuracy in both normal and crisis periods.

**Rolling window retraining**: Rather than a fixed training set, retraining the model periodically on the most recent two to three years of data would allow it to adapt to evolving market conditions, including the ongoing growth of renewable capacity and changes in the generation mix.

**Extended hyperparameter search**: The grid search in Sprint 2 covered a limited parameter space. A more comprehensive Bayesian optimization search over a wider range of XGBoost parameters (subsample, colsample_bytree, min_child_weight, gamma) could yield further improvement.

---

## Methodological Notes

**Data leakage prevention**: All features use only information available at prediction time. The StandardScaler was fit exclusively on training data and applied without refitting to validation and test sets. The temporal train/validation/test split was chosen based on domain knowledge of the energy crisis timeline, not to maximize reported performance.

**TimeSeriesSplit for cross-validation**: Standard k-fold cross-validation was not used for hyperparameter tuning because it would mix future observations into training folds. TimeSeriesSplit preserves chronological order within the training set, ensuring that each validation fold contains only observations that follow the corresponding training fold in time.

**Treatment of extreme prices**: Prices above 300 EUR/MWh and below -100 EUR/MWh were retained in the training data to ensure the model is exposed to crisis-period behavior. These extreme values do inflate RMSE metrics, but removing them would produce an optimistic evaluation that does not reflect real market conditions.

---

## Acknowledgements

This project was developed with the assistance of large language model tools including Claude Sonnet (Anthropic), ChatGPT 4.5 (OpenAI), and OpenAI Codex. These tools were used for code generation assistance, researching of domain concepts, and iterative debugging of implementation. Business motivation, analytical and planning decisions, sprint planning, feature engineering choices, model selection rationale, and interpretations of results are the author's own.

This project is based on publicly available data. Data provided by SMARD (Bundesnetzagentur), Open-Meteo (ERA5/ECMWF), and Investing.com (ICE TTF). ENTSO-E for the transparency platform providing the underlying electricity market data underlying SMARD publications.
