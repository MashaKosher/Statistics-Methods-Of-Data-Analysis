import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ПРАКТИЧЕСКОЕ ЗАДАНИЕ №8 - ПРОГНОЗИРОВАНИЕ ВРЕМЕННЫХ РЯДОВ")
print("=" * 80)

# -------------------------------
# ЗАДАНИЕ 1 - Анализ тренда
# -------------------------------
print("\n" + "=" * 50)
print("ЗАДАНИЕ 1 - Анализ тренда")
print("=" * 50)

# Данные из таблицы 1
data_task1 = {
    'Год': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Расходы': [7, 8, 8, 10, 11, 12, 14, 16, 17, 19]
}

df1 = pd.DataFrame(data_task1)

# Расчет автокорреляции
def calculate_autocorrelation(series, lag):
    return series.autocorr(lag=lag)

# Коэффициенты автокорреляции
r1 = calculate_autocorrelation(df1['Расходы'], 1)
r2 = calculate_autocorrelation(df1['Расходы'], 2)

print(f"Коэффициент автокорреляции 1-го порядка: {r1:.4f}")
print(f"Коэффициент автокорреляции 2-го порядка: {r2:.4f}")

# Построение линейного тренда
X = df1[['Год']]
y = df1['Расходы']

model = LinearRegression()
model.fit(X, y)

trend_line = model.predict(X)
r_squared = model.score(X, y)

print(f"\nУравнение линейного тренда: y = {model.intercept_:.2f} + {model.coef_[0]:.2f}*t")
print(f"Коэффициент детерминации R²: {r_squared:.4f}")

# Визуализация
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.scatter(df1['Год'], df1['Расходы'], color='blue', label='Фактические данные')
plt.plot(df1['Год'], trend_line, color='red', label='Линейный тренд')
plt.xlabel('Год')
plt.ylabel('Расходы, тыс. у.е.')
plt.title('Линейный тренд расходов')
plt.legend()
plt.grid(True)

# Степенной тренд (логарифмическая трансформация)
df1['ln_Год'] = np.log(df1['Год'])
df1['ln_Расходы'] = np.log(df1['Расходы'])

model_power = LinearRegression()
model_power.fit(df1[['ln_Год']], df1['ln_Расходы'])

power_trend = np.exp(model_power.predict(df1[['ln_Год']]))

plt.subplot(2, 2, 2)
plt.scatter(df1['Год'], df1['Расходы'], color='blue', label='Фактические данные')
plt.plot(df1['Год'], power_trend, color='green', label='Степенной тренд')
plt.xlabel('Год')
plt.ylabel('Расходы, тыс. у.е.')
plt.title('Степенной тренд расходов')
plt.legend()
plt.grid(True)

# -------------------------------
# ЗАДАНИЕ ДЛЯ САМОСТОЯТЕЛЬНОЙ РАБОТЫ №1
# -------------------------------
print("\n" + "=" * 50)
print("ЗАДАНИЕ ДЛЯ САМОСТОЯТЕЛЬНОЙ РАБОТЫ №1")
print("=" * 50)

# Данные для самостоятельной работы
data_independent = {
    # 'Вариант 1': [13.5, 12.7, 12.0, 11.9, 11.5, 11.2, 10.8, 10.7, 10.6, 10.5],
    # 'Вариант 2': [251, 249, 248, 246, 242, 239, 235, 230, 228, 225],
    'Вариант 3': [0.91, 0.87, 0.85, 0.82, 0.79, 0.75, 0.70, 0.66, 0.62, 0.60]
    # 'Вариант 4': [2.54, 2.50, 2.45, 2.40, 2.37, 2.30, 2.27, 2.19, 2.05, 2.00],
    # 'Вариант 5': [6.3, 6.21, 6.15, 6.00, 5.80, 5.45, 5.05, 4.85, 4.50, 4.20],
    # 'Вариант 6': [18.2, 17.5, 17.1, 16.8, 16.1, 15.7, 15.2, 14.5, 14.3, 14.0],
    # 'Вариант 7': [134, 130, 128, 126, 122, 120, 117, 112, 108, 105],
    # 'Вариант 8': [64, 61, 58, 52, 49, 45, 40, 37, 34, 30],
    # 'Вариант 9': [4.25, 4.20, 4.18, 4.11, 4.05, 4.00, 3.91, 3.85, 3.77, 3.70],
    # 'Вариант 10': [1.8, 1.78, 1.70, 1.64, 1.59, 1.51, 1.45, 1.42, 1.40, 1.37]
}

years = list(range(1, 11))

# Анализ тренда для каждого варианта
for i, (variant, data) in enumerate(data_independent.items(), 1):
    r1_var = calculate_autocorrelation(pd.Series(data), 1)
    
    X_var = np.array(years).reshape(-1, 1)
    y_var = np.array(data)
    
    model_var = LinearRegression()
    model_var.fit(X_var, y_var)
    r_squared_var = model_var.score(X_var, y_var)
    
    trend_direction = "возрастающий" if model_var.coef_[0] > 0 else "убывающий"
    
    print(f"{variant}: r1 = {r1_var:.3f}, R² = {r_squared_var:.3f}, {trend_direction} тренд")

# -------------------------------
# ЗАДАНИЕ 2 - Сглаживание и прогнозирование
# -------------------------------
print("\n" + "=" * 50)
print("ЗАДАНИЕ 2 - Сглаживание и прогнозирование")
print("=" * 50)

# Скользящее среднее
def moving_average(data, window):
    return data.rolling(window=window).mean()

# Экспоненциальное сглаживание
def exponential_smoothing(data, alpha):
    result = [data[0]]
    for i in range(1, len(data)):
        result.append(alpha * data[i] + (1 - alpha) * result[i-1])
    return result

# Данные для сглаживания
expenses = df1['Расходы'].values

# Скользящее среднее (окно = 3)
ma_3 = moving_average(pd.Series(expenses), 3)

# Экспоненциальное сглаживание (alpha = 0.25)
es_025 = exponential_smoothing(expenses, 0.25)

# Прогноз на 11-й период
# Для скользящего среднего
ma_forecast = ma_3.iloc[-3:].mean()

# Для экспоненциального сглаживания
last_es = es_025[-1]
es_forecast = 0.25 * expenses[-1] + 0.75 * last_es

print(f"Прогноз на 11-й период:")
print(f"Скользящее среднее: {ma_forecast:.2f} тыс. у.е.")
print(f"Экспоненциальное сглаживание: {es_forecast:.2f} тыс. у.е.")

# Визуализация сглаживания
plt.subplot(2, 2, 3)
plt.plot(df1['Год'], expenses, 'o-', label='Фактические данные', color='blue')
plt.plot(df1['Год'][2:], ma_3.dropna(), 's-', label='Скользящее среднее (окно=3)', color='red')
plt.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Год')
plt.ylabel('Расходы, тыс. у.е.')
plt.title('Сглаживание скользящим средним')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(df1['Год'], expenses, 'o-', label='Фактические данные', color='blue')
plt.plot(df1['Год'], es_025, 's-', label='Экспоненциальное сглаживание (α=0.25)', color='green')
plt.axvline(x=10.5, color='gray', linestyle='--', alpha=0.7)
plt.xlabel('Год')
plt.ylabel('Расходы, тыс. у.е.')
plt.title('Экспоненциальное сглаживание')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -------------------------------
# ЗАДАНИЕ 3 - Анализ сезонности
# -------------------------------
print("\n" + "=" * 50)
print("ЗАДАНИЕ 3 - Анализ сезонности")
print("=" * 50)

# Данные о потреблении электроэнергии
energy_data = {
    'Период': ['I кв. 2001', 'II кв. 2001', 'III кв. 2001', 'IV кв. 2001',
               'I кв. 2002', 'II кв. 2002', 'III кв. 2002', 'IV кв. 2002',
               'I кв. 2003', 'II кв. 2003', 'III кв. 2003', 'IV кв. 2003',
               'I кв. 2004', 'II кв. 2004', 'III кв. 2004', 'IV кв. 2004'],
    'Потребление': [6.0, 4.4, 5.0, 9.0, 7.2, 4.8, 6.0, 10.0, 
                   8.0, 5.6, 6.4, 11.0, 9.0, 6.6, 7.0, 10.8]
}

df_energy = pd.DataFrame(energy_data)
df_energy['t'] = range(1, 17)

# Визуализация временного ряда
plt.figure(figsize=(12, 6))
plt.plot(df_energy['t'], df_energy['Потребление'], 'o-', linewidth=2)
plt.xlabel('Период')
plt.ylabel('Потребление электроэнергии, млрд. кВт-ч')
plt.title('Потребление электроэнергии по кварталам (2001-2004)')
plt.xticks(df_energy['t'], df_energy['Период'], rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Аддитивная декомпозиция временного ряда
def additive_seasonal_decomposition(data, period=4):
    n = len(data)
    
    # Скользящее среднее
    moving_avg = []
    for i in range(2, n-1):
        moving_avg.append(np.mean(data[i-2:i+2]))
    
    # Центрированное скользящее среднее
    centered_moving_avg = []
    for i in range(len(moving_avg)-1):
        centered_moving_avg.append((moving_avg[i] + moving_avg[i+1]) / 2)
    
    # Оценка сезонной компоненты
    seasonal_components = []
    for i in range(2, n-2):
        seasonal_components.append(data[i] - centered_moving_avg[i-2])
    
    # Группировка по кварталам
    seasonal_by_quarter = [[] for _ in range(period)]
    for i, comp in enumerate(seasonal_components):
        quarter = (i + 2) % period  # +2 потому что начинаем с 3-го периода
        seasonal_by_quarter[quarter].append(comp)
    
    # Средние сезонные компоненты
    avg_seasonal = [np.mean(quarter_data) for quarter_data in seasonal_by_quarter]
    
    # Корректирующий коэффициент
    k = np.mean(avg_seasonal)
    
    # Скорректированные сезонные компоненты
    adjusted_seasonal = [comp - k for comp in avg_seasonal]
    
    return centered_moving_avg, seasonal_components, adjusted_seasonal

# Применяем декомпозицию
centered_ma, seasonal_est, final_seasonal = additive_seasonal_decomposition(df_energy['Потребление'].values)

print("Сезонные компоненты по кварталам:")
quarters = ['I квартал', 'II квартал', 'III квартал', 'IV квартал']
for i, (quarter, comp) in enumerate(zip(quarters, final_seasonal)):
    print(f"{quarter}: {comp:.3f}")

# Элиминируем сезонность
df_energy['Сезонная компонента'] = [final_seasonal[i % 4] for i in range(len(df_energy))]
df_energy['Без сезонности'] = df_energy['Потребление'] - df_energy['Сезонная компонента']

# Строим тренд для данных без сезонности
X_energy = df_energy[['t']]
y_deseasonalized = df_energy['Без сезонности']

model_energy = LinearRegression()
model_energy.fit(X_energy, y_deseasonalized)

trend_energy = model_energy.predict(X_energy)

print(f"\nТренд для данных без сезонности: y = {model_energy.intercept_:.2f} + {model_energy.coef_[0]:.2f}*t")
print(f"R² тренда: {model_energy.score(X_energy, y_deseasonalized):.4f}")

# Прогноз на 2005 год
forecast_2005 = []
for i in range(17, 21):  # периоды 17-20 (2005 год)
    quarter = (i - 1) % 4  # определяем квартал
    trend_value = model_energy.intercept_ + model_energy.coef_[0] * i
    seasonal_value = final_seasonal[quarter]
    forecast = trend_value + seasonal_value
    forecast_2005.append(forecast)

print("\nПрогноз потребления электроэнергии на 2005 год:")
quarters_2005 = ['I кв. 2005', 'II кв. 2005', 'III кв. 2005', 'IV кв. 2005']
for quarter, forecast in zip(quarters_2005, forecast_2005):
    print(f"{quarter}: {forecast:.2f} млрд. кВт-ч")

# -------------------------------
# ЗАДАНИЕ ДЛЯ САМОСТОЯТЕЛЬНОЙ РАБОТЫ №3
# -------------------------------
print("\n" + "=" * 50)
print("ЗАДАНИЕ ДЛЯ САМОСТОЯТЕЛЬНОЙ РАБОТЫ №3")
print("=" * 50)

# Данные об экспорте
export_data = {
    'Период': list(range(1, 25)),
    'Экспорт': [4087, 4737, 5768, 6005, 5639, 6745, 6311, 7107, 
                5741, 7087, 7310, 8600, 6975, 6891, 7527, 7971,
                5875, 6140, 6248, 6041, 4626, 6501, 6284, 6707]
}

df_export = pd.DataFrame(export_data)

# Анализ сезонности для экспорта
centered_ma_export, seasonal_est_export, final_seasonal_export = additive_seasonal_decomposition(df_export['Экспорт'].values)

print("Сезонные компоненты экспорта по кварталам:")
for i, (quarter, comp) in enumerate(zip(quarters, final_seasonal_export)):
    print(f"{quarter}: {comp:.1f} млрд. дол.")

# Прогноз на 2006 год (периоды 25-28)
df_export['Сезонная компонента'] = [final_seasonal_export[i % 4] for i in range(len(df_export))]
df_export['Без сезонности'] = df_export['Экспорт'] - df_export['Сезонная компонента']

# Тренд для экспорта
X_export = df_export[['Период']]
y_export_deseasonalized = df_export['Без сезонности']

model_export = LinearRegression()
model_export.fit(X_export, y_export_deseasonalized)

# Прогноз на 2006 год
forecast_2006 = []
for i in range(25, 29):  # периоды 25-28 (2006 год)
    quarter = (i - 1) % 4
    trend_value = model_export.intercept_ + model_export.coef_[0] * i
    seasonal_value = final_seasonal_export[quarter]
    forecast = trend_value + seasonal_value
    forecast_2006.append(forecast)

print("\nПрогноз экспорта на 2006 год:")
quarters_2006 = ['I кв. 2006', 'II кв. 2006', 'III кв. 2006', 'IV кв. 2006']
for quarter, forecast in zip(quarters_2006, forecast_2006):
    print(f"{quarter}: {forecast:.0f} млрд. дол.")

# -------------------------------
# ИТОГОВАЯ ВИЗУАЛИЗАЦИЯ
# -------------------------------
plt.figure(figsize=(15, 10))

# График потребления электроэнергии с прогнозом
plt.subplot(2, 2, 1)
plt.plot(df_energy['t'], df_energy['Потребление'], 'o-', label='Фактические данные', linewidth=2)
plt.plot(range(17, 21), forecast_2005, 's-', label='Прогноз 2005', linewidth=2)
plt.xlabel('Период')
plt.ylabel('Потребление электроэнергии, млрд. кВт-ч')
plt.title('Потребление электроэнергии с прогнозом на 2005 год')
plt.legend()
plt.grid(True)

# График экспорта с прогнозом
plt.subplot(2, 2, 2)
plt.plot(df_export['Период'], df_export['Экспорт'], 'o-', label='Фактические данные', linewidth=2)
plt.plot(range(25, 29), forecast_2006, 's-', label='Прогноз 2006', linewidth=2)
plt.xlabel('Период')
plt.ylabel('Экспорт, млрд. дол.')
plt.title('Экспорт с прогнозом на 2006 год')
plt.legend()
plt.grid(True)

# Сезонные компоненты электроэнергии
plt.subplot(2, 2, 3)
plt.bar(quarters, final_seasonal, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Сезонные компоненты потребления электроэнергии')
plt.ylabel('Сезонная компонента')
plt.grid(True, alpha=0.3)

# Сезонные компоненты экспорта
plt.subplot(2, 2, 4)
plt.bar(quarters, final_seasonal_export, color=['lightblue', 'lightgreen', 'lightcoral', 'gold'])
plt.title('Сезонные компоненты экспорта')
plt.ylabel('Сезонная компонента, млрд. дол.')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("ВЫВОДЫ:")
print("1. Все временные ряды демонстрируют наличие тренда и сезонности")
print("2. Аддитивная модель хорошо описывает сезонные колебания")
print("3. Прогнозные значения учитывают как трендовую, так и сезонную компоненты")
print("=" * 80)