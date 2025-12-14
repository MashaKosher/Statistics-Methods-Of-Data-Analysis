import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА - СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("=" * 80)

# -------------------------------
# ЛАБОРАТОРНАЯ РАБОТА 5
# -------------------------------
print("\n" + "=" * 50)
print("ЛАБОРАТОРНАЯ РАБОТА 5")
print("=" * 50)

data_lab5 = {
    't': range(1, 21),
    'tx1': [12, 17, 14, 13, 16, 15, 13, 11, 15, 13, 12, 15, 13, 16, 17, 15, 11, 14, 13, 15],
    'tx2': [2, 5, 6, 4, 3, 2, 6, 5, 4, 6, 5, 3, 2, 5, 5, 4, 5, 4, 2, 3],
    'tx3': [8, 12, 11, 9, 12, 9, 10, 13, 10, 11, 14, 14, 8, 11, 10, 13, 12, 12, 14, 11],
    'ty': [139, 182, 164, 150, 176, 168, 173, 145, 175, 157, 142, 151, 148, 186, 201, 169, 160, 151, 129, 163]
}

df5 = pd.DataFrame(data_lab5)
X5 = df5[['tx1', 'tx2', 'tx3']]
X5 = sm.add_constant(X5)
y5 = df5['ty']

# Построение модели
model5 = sm.OLS(y5, X5).fit()

print("МОДЕЛЬ 5: ty = β₀ + β₁·tx1 + β₂·tx2 + β₃·tx3 + ε")
print("Коэффициенты модели:")
print(f"const: {model5.params['const']:.4f}")
print(f"tx1:   {model5.params['tx1']:.4f}")
print(f"tx2:   {model5.params['tx2']:.4f}")
print(f"tx3:   {model5.params['tx3']:.4f}")

print(f"\nR² = {model5.rsquared:.4f}")
print(f"Adj. R² = {model5.rsquared_adj:.4f}")
print(f"F-statistic = {model5.fvalue:.4f}")
print(f"p-value (F-test) = {model5.f_pvalue:.4f}")

# -------------------------------
# ЛАБОРАТОРНАЯ РАБОТА 6
# -------------------------------
print("\n" + "=" * 50)
print("ЛАБОРАТОРНАЯ РАБОТА 6")
print("=" * 50)

# Создаем данные из содержимого файла
data_lab6 = """Country,Population_millions,GDP_billion_USD,CO2_emissions_tons_per_capita,Energy_consumption_kWh_per_capita,Healthcare_expenditure_percent_GDP,Urbanization_percent
Belarus,9.5,63.1,6.4,3250,6.1,79.5
Russia,144.1,1829.0,11.9,6500,5.3,74.8
Ukraine,41.2,200.1,4.7,2850,7.1,69.5
Poland,37.8,679.4,8.1,4100,4.9,60.5
Lithuania,2.8,56.5,4.5,3850,6.8,68.1
Latvia,1.9,34.9,3.8,3650,6.2,68.4
Estonia,1.3,31.0,8.0,5200,6.7,69.2
Germany,83.2,4223.1,8.5,7100,11.1,77.5
France,67.4,2937.5,4.6,7300,11.2,81.0
United_Kingdom,67.9,3131.4,5.6,5400,10.9,83.9
Italy,59.1,2106.3,5.4,5200,8.8,70.4
Spain,47.4,1394.1,5.0,5850,9.7,80.8
Netherlands,17.4,909.9,8.8,6800,10.1,92.2
Belgium,11.6,529.6,7.6,7850,10.5,98.0
Czech_Republic,10.7,281.8,9.8,6200,7.8,73.8
Slovakia,5.5,105.9,6.1,5100,6.7,53.7
Hungary,9.7,181.8,4.9,4350,6.8,71.4
Romania,19.1,249.7,3.9,2650,5.6,54.0
Bulgaria,6.9,84.1,6.7,4150,8.2,75.7
Croatia,3.9,60.4,4.2,3850,7.8,57.3
Slovenia,2.1,54.2,6.1,6850,8.5,55.1
Austria,9.0,446.3,7.3,8200,10.4,58.7
Switzerland,8.7,812.9,4.3,7400,11.3,73.8
Norway,5.4,482.2,7.7,22500,9.7,82.7
Sweden,10.4,541.2,3.8,13200,9.9,88.2
Denmark,5.8,396.0,5.0,6100,10.1,88.1
Finland,5.5,297.3,7.4,15400,9.2,85.5
Iceland,0.4,24.2,4.4,54000,8.0,94.1
Ireland,5.0,498.6,7.7,5900,7.1,63.7
Portugal,10.3,238.3,4.1,4650,9.5,66.3
Greece,10.7,188.8,5.8,3850,7.7,79.7
Turkey,84.3,761.4,4.5,3200,4.3,75.1
United_States,331.9,23315.1,14.2,12100,17.1,82.7
Canada,38.2,1988.3,15.6,14800,10.8,81.6
Mexico,128.9,1293.0,3.7,2150,5.4,80.7
Brazil,214.3,1869.2,2.3,2450,9.6,87.1
Argentina,45.4,491.5,4.3,3050,9.5,92.1
Chile,19.1,317.1,4.4,3850,9.1,87.8
Uruguay,3.5,59.3,2.0,3150,9.2,95.5
Colombia,50.9,314.5,1.8,1350,7.7,81.4
Peru,33.0,202.0,1.9,1250,5.2,78.1
Ecuador,17.6,107.4,2.3,1450,8.5,64.2
Venezuela,28.4,48.0,5.8,2850,1.2,88.2
Paraguay,7.1,40.7,1.2,1550,6.7,62.2
Bolivia,11.8,40.9,1.8,850,6.9,69.8
China,1439.3,17734.1,7.4,4300,5.4,61.4
India,1380.0,3173.4,1.9,900,3.5,35.0
Japan,125.8,4937.4,8.7,7450,10.9,91.8
South_Korea,51.8,1810.9,11.6,10350,8.1,81.4
Indonesia,273.5,1289.0,2.3,900,2.9,56.0
Thailand,69.8,543.5,3.8,2650,3.8,50.4
Vietnam,97.3,362.6,3.2,1850,6.1,37.3
Philippines,109.0,394.1,1.3,750,4.4,47.4
Malaysia,32.4,401.0,7.8,4650,3.8,76.6
Singapore,5.9,372.1,8.6,8350,4.1,100.0
Australia,25.7,1553.3,15.4,9950,9.3,86.2
New_Zealand,5.1,249.9,6.9,9300,9.7,86.7"""

from io import StringIO
df6 = pd.read_csv(StringIO(data_lab6))

X6 = df6[['Population_millions', 'CO2_emissions_tons_per_capita',
          'Energy_consumption_kWh_per_capita', 'Healthcare_expenditure_percent_GDP',
          'Urbanization_percent']]
X6 = sm.add_constant(X6)
y6 = df6['GDP_billion_USD']

# Построение модели
model6 = sm.OLS(y6, X6).fit()

print("МОДЕЛЬ 6: GDP = β₀ + β₁·Population + β₂·CO2 + β₃·Energy + β₄·Health + β₅·Urban + ε")
print("Коэффициенты модели:")
print(f"const:      {model6.params['const']:.4f}")
print(f"Population: {model6.params['Population_millions']:.4f}")
print(f"CO2:        {model6.params['CO2_emissions_tons_per_capita']:.4f}")
print(f"Energy:     {model6.params['Energy_consumption_kWh_per_capita']:.4f}")
print(f"Health:     {model6.params['Healthcare_expenditure_percent_GDP']:.4f}")
print(f"Urban:      {model6.params['Urbanization_percent']:.4f}")

print(f"\nR² = {model6.rsquared:.4f}")
print(f"Adj. R² = {model6.rsquared_adj:.4f}")
print(f"F-statistic = {model6.fvalue:.4f}")
print(f"p-value (F-test) = {model6.f_pvalue:.4f}")

# -------------------------------
# СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ
# -------------------------------
print("\n" + "=" * 80)
print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ МОДЕЛЕЙ")
print("=" * 80)

# Создаем таблицу для сравнения
comparison = pd.DataFrame({
    'Параметр': ['R²', 'Adj. R²', 'F-statistic', 'AIC', 'BIC', 'No. Observations'],
    'Модель 5': [
        model5.rsquared,
        model5.rsquared_adj,
        model5.fvalue,
        model5.aic,
        model5.bic,
        model5.nobs
    ],
    'Модель 6': [
        model6.rsquared,
        model6.rsquared_adj,
        model6.fvalue,
        model6.aic,
        model6.bic,
        model6.nobs
    ]
})

print(comparison.to_string(index=False, float_format="%.4f"))

# -------------------------------
# ПРОВЕРКА НА АВТОКОРРЕЛЯЦИЮ
# -------------------------------
print("\n" + "=" * 50)
print("ПРОВЕРКА НА АВТОКОРРЕЛЯЦИЮ (Тест Дарбина-Уотсона)")
print("=" * 50)

dw5 = durbin_watson(model5.resid)
dw6 = durbin_watson(model6.resid)

print(f"Модель 5 - DW статистика: {dw5:.4f}")
print(f"Модель 6 - DW статистика: {dw6:.4f}")

# Табличные значения Дарбина-Уотсона
print("\nТАБЛИЧНЫЕ ЗНАЧЕНИЯ ДАРБИНА-УОТСОНА (α=0.05):")
print("Для Модели 5 (n=20, k=3): dL=0.998, dU=1.676")
print("Для Модели 6 (n=55, k=5): dL=1.378, dU=1.721")

print("\nИНТЕРПРЕТАЦИЯ ДЛЯ МОДЕЛИ 5:")
if dw5 < 0.998:
    print("Положительная автокорреляция ✗")
elif dw5 > 2.402:  # 4 - dL
    print("Отрицательная автокорреляция ✗")
elif 1.676 < dw5 < 2.324:  # зона неопределенности для положительной
    print("Нет автокорреляции ✓")
else:
    print("Зона неопределенности")

print("\nИНТЕРПРЕТАЦИЯ ДЛЯ МОДЕЛИ 6:")
if dw6 < 1.378:
    print("Положительная автокорреляция ✗")
elif dw6 > 2.622:  # 4 - dL
    print("Отрицательная автокорреляция ✗")
elif 1.721 < dw6 < 2.279:  # зона неопределенности для положительной
    print("Нет автокорреляции ✓")
else:
    print("Зона неопределенности")

# -------------------------------
# ПРОВЕРКА НА ГЕТЕРОСКЕДАСТИЧНОСТЬ
# -------------------------------
print("\n" + "=" * 50)
print("ПРОВЕРКА НА ГЕТЕРОСКЕДАСТИЧНОСТЬ (Тест Бреуша-Пагана)")
print("=" * 50)

# Тест Бреуша-Пагана
bp5 = het_breuschpagan(model5.resid, model5.model.exog)
bp6 = het_breuschpagan(model6.resid, model6.model.exog)

print("Модель 5 - Тест Бреуша-Пагана:")
print(f"  LM статистика: {bp5[0]:.4f}")
print(f"  p-value: {bp5[1]:.4f}")
print(f"  F-statistic: {bp5[2]:.4f}")
print(f"  F p-value: {bp5[3]:.4f}")

print("\nМодель 6 - Тест Бреуша-Пагана:")
print(f"  LM статистика: {bp6[0]:.4f}")
print(f"  p-value: {bp6[1]:.4f}")
print(f"  F-statistic: {bp6[2]:.4f}")
print(f"  F p-value: {bp6[3]:.4f}")

print("\nТАБЛИЧНЫЕ ЗНАЧЕНИЯ (χ², α=0.05):")
print("Критическое значение χ² для:")
print("  Модель 5 (df=3): 7.815")
print("  Модель 6 (df=5): 11.070")

print("\nИНТЕРПРЕТАЦИЯ:")
if bp5[1] > 0.05:
    print("Модель 5: Нет гетероскедастичности (гомоскедастичность) ✓")
else:
    print("Модель 5: Обнаружена гетероскедастичность ✗")

if bp6[1] > 0.05:
    print("Модель 6: Нет гетероскедастичности (гомоскедастичность) ✓")
else:
    print("Модель 6: Обнаружена гетероскедастичность ✗")

# -------------------------------
# ПРОВЕРКА НА МУЛЬТИКОЛЛИНЕАРНОСТЬ
# -------------------------------
print("\n" + "=" * 50)
print("ПРОВЕРКА НА МУЛЬТИКОЛЛИНЕАРНОСТЬ (VIF)")
print("=" * 50)

def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Variable'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

print("Модель 5 - VIF значения:")
vif5 = calculate_vif(X5)
for i, row in vif5.iterrows():
    print(f"  {row['Variable']}: {row['VIF']:.4f}")

print("\nМодель 6 - VIF значения:")
vif6 = calculate_vif(X6)
for i, row in vif6.iterrows():
    print(f"  {row['Variable']}: {row['VIF']:.4f}")
    
print("\nКРИТЕРИИ ОЦЕНКИ VIF:")
print("VIF < 5: слабая мультиколлинеарность ✓")
print("5 ≤ VIF < 10: умеренная мультиколлинеарность ⚠")
print("VIF ≥ 10: сильная мультиколлинеарность ✗")

multicoll5 = any(vif5['VIF'] >= 5)
multicoll6 = any(vif6['VIF'] >= 5)

print(f"\nМодель 5: {'Обнаружена мультиколлинеарность ✗' if multicoll5 else 'Нет мультиколлинеарности ✓'}")
print(f"Модель 6: {'Обнаружена мультиколлинеарность ✗' if multicoll6 else 'Нет мультиколлинеарности ✓'}")

# -------------------------------
# ВИЗУАЛИЗАЦИЯ ОСТАТКОВ
# -------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Модель 5
axes[0,0].scatter(model5.fittedvalues, model5.resid, alpha=0.7)
axes[0,0].axhline(y=0, color='red', linestyle='--')
axes[0,0].set_title('Модель 5: Остатки vs Предсказанные')
axes[0,0].set_xlabel('Предсказанные значения')
axes[0,0].set_ylabel('Остатки')

axes[0,1].plot(model5.resid, marker='o')
axes[0,1].axhline(y=0, color='red', linestyle='--')
axes[0,1].set_title('Модель 5: Остатки по наблюдениям')
axes[0,1].set_xlabel('Наблюдения')
axes[0,1].set_ylabel('Остатки')

sm.qqplot(model5.resid, line='45', ax=axes[0,2])
axes[0,2].set_title('Модель 5: Q-Q plot')

# Модель 6
axes[1,0].scatter(model6.fittedvalues, model6.resid, alpha=0.7)
axes[1,0].axhline(y=0, color='red', linestyle='--')
axes[1,0].set_title('Модель 6: Остатки vs Предсказанные')
axes[1,0].set_xlabel('Предсказанные значения')
axes[1,0].set_ylabel('Остатки')

axes[1,1].plot(model6.resid, marker='o')
axes[1,1].axhline(y=0, color='red', linestyle='--')
axes[1,1].set_title('Модель 6: Остатки по наблюдениям')
axes[1,1].set_xlabel('Наблюдения')
axes[1,1].set_ylabel('Остатки')

sm.qqplot(model6.resid, line='45', ax=axes[1,2])
axes[1,2].set_title('Модель 6: Q-Q plot')

plt.tight_layout()
plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# ОБЩИЕ ВЫВОДЫ
# -------------------------------
print("\n" + "=" * 80)
print("ОБЩИЕ ВЫВОДЫ ПО АНАЛИЗУ МОДЕЛЕЙ")
print("=" * 80)

print("МОДЕЛЬ 5 (Лабораторная работа 5):")
print(f"• Качество модели: R² = {model5.rsquared:.4f}")
print(f"• Автокорреляция: {'Отсутствует' if 1.676 < dw5 < 2.324 else 'Присутствует'} (DW = {dw5:.4f})")
print(f"• Гетероскедастичность: {'Отсутствует' if bp5[1] > 0.05 else 'Присутствует'} (p = {bp5[1]:.4f})")
print(f"• Мультиколлинеарность: {'Отсутствует' if not multicoll5 else 'Присутствует'}")

print("\nМОДЕЛЬ 6 (Лабораторная работа 6):")
print(f"• Качество модели: R² = {model6.rsquared:.4f}")
print(f"• Автокорреляция: {'Отсутствует' if 1.721 < dw6 < 2.279 else 'Присутствует'} (DW = {dw6:.4f})")
print(f"• Гетероскедастичность: {'Отсутствует' if bp6[1] > 0.05 else 'Присутствует'} (p = {bp6[1]:.4f})")
print(f"• Мультиколлинеарность: {'Отсутствует' if not multicoll6 else 'Присутствует'}")

print("\nРЕКОМЕНДАЦИИ:")
issues5 = []
if not (1.676 < dw5 < 2.324):
    issues5.append("автокорреляция")
if bp5[1] <= 0.05:
    issues5.append("гетероскедастичность")
if multicoll5:
    issues5.append("мультиколлинеарность")

issues6 = []
if not (1.721 < dw6 < 2.279):
    issues6.append("автокорреляция")
if bp6[1] <= 0.05:
    issues6.append("гетероскедастичность")
if multicoll6:
    issues6.append("мультиколлинеарность")

if issues5:
    print(f"Для Модели 5 требуются корректирующие меры для: {', '.join(issues5)}")
else:
    print("Модель 5 соответствует основным предположениям МНК ✓")

if issues6:
    print(f"Для Модели 6 требуются корректирующие меры для: {', '.join(issues6)}")
else:
    print("Модель 6 соответствует основным предположениям МНК ✓")

# -------------------------------
# ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ
# -------------------------------
print("\n" + "=" * 80)
print("ДОПОЛНИТЕЛЬНАЯ СТАТИСТИЧЕСКАЯ ИНФОРМАЦИЯ")
print("=" * 80)

print("МОДЕЛЬ 5 - Статистическая значимость коэффициентов:")
for param_name in model5.params.index:
    coef = model5.params[param_name]
    pval = model5.pvalues[param_name]
    significance = "✓" if pval < 0.05 else "✗"
    print(f"  {param_name}: {coef:.4f} (p-value: {pval:.4f}) {significance}")

print("\nМОДЕЛЬ 6 - Статистическая значимость коэффициентов:")
for param_name in model6.params.index:
    coef = model6.params[param_name]
    pval = model6.pvalues[param_name]
    significance = "✓" if pval < 0.05 else "✗"
    print(f"  {param_name}: {coef:.4f} (p-value: {pval:.4f}) {significance}")