# Jupyter notebook (as a .py script) for ПЗ 7 — Проверка моделей на автокорреляцию, гетероскедастичность и мультиколлинеарность
# Следую ТЗ: Задание ПЗ 7.pdf. Ссылка на документ: fileciteturn2file0

# Эта тетрадь автоматически загружает data.csv (в рабочей директории), строит OLS
# для выбранной целевой переменной (по умолчанию — первая колонка),
# выполняет диагностические тесты и при необходимости предлагает корректировки.

# 0) Установка и импорт библиотек
# Ячейка для запуска
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_breusch_godfrey
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from scipy import stats
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data.csv'))

# 1) Загрузка данных
# Запустите эту ячейку; если файл data.csv находится в другом месте, скорректируйте путь.
try:
    df = pd.read_csv('data.csv')
except Exception as e:
    raise FileNotFoundError("Не найден файл 'data.csv' в рабочей директории. Положите data.csv рядом с этим ноутбуком.")

print('Размер данных:', df.shape)
print('\nПервые 5 строк:')
print(df.head())

# 2) Определение зависимой и независимых переменных
# По умолчанию: первая колонка — зависимая (y), остальные — X.
cols = df.columns.tolist()
if len(cols) < 2:
    raise ValueError('В файле должно быть как минимум 2 столбца (y и хотя бы один x).')

y_col = cols[0]
X_cols = cols[1:]
print(f"Автоматически выбранная зависимая: {y_col}")
print(f"Автоматически выбранные регрессоры: {X_cols}")

# 3) Подготовка данных и оценка исходной OLS модели
X = sm.add_constant(df[X_cols])
y = df[y_col]
model_ols = sm.OLS(y, X).fit()
print(model_ols.summary())

# 4) Остатки — графики и ACF/PACF
resid = model_ols.resid
fitted = model_ols.fittedvalues

# Остатки по наблюдениям
plt.figure(figsize=(10,4))
plt.plot(resid, marker='o', linestyle='-', label='resid')
plt.axhline(0, color='k', linestyle='--')
plt.title('Остатки модели по наблюдениям')
plt.legend();

# График остатков vs предсказанных
plt.figure(figsize=(6,4))
plt.scatter(fitted, resid)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

# ACF и PACF остатков
plt.figure(figsize=(10,4))
plot_acf(resid, lags=20, alpha=0.05)
plt.title('ACF of residuals')

plt.figure(figsize=(10,4))
plot_pacf(resid, lags=20, alpha=0.05, method='ywm')
plt.title('PACF of residuals')

# 5) Тесты автокорреляции: Durbin-Watson и Breusch-Godfrey
dw = durbin_watson(resid)
print(f'Durbin-Watson: {dw:.4f}  (прибл. 2 = отсутствие автокорреляции)')

# Breusch-Godfrey (lag=1..4)
for k in [1,2,4]:
    bg_test = acorr_breusch_godfrey(model_ols, nlags=k)
    lm_stat, lm_pvalue, f_stat, f_pvalue = bg_test
    print(f'Breusch-Godfrey (nlags={k}): LM stat={lm_stat:.4f}, p-value={lm_pvalue:.4f}, F stat={f_stat:.4f}, p-value={f_pvalue:.4f}')

# 6) Тесты на гетероскедастичность
# Breusch-Pagan
bp_test = het_breuschpagan(resid, model_ols.model.exog)
bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue = bp_test
print('\nBreusch-Pagan: LM stat={:.4f}, p-value={:.4f}, F stat={:.4f}, p-value={:.4f}'.format(bp_lm, bp_lm_pvalue, bp_fvalue, bp_f_pvalue))

# White test (using statsmodels)
white_test = het_white(resid, model_ols.model.exog)
w_lm, w_lm_pvalue, w_fvalue, w_f_pvalue = white_test
print('White test: LM stat={:.4f}, p-value={:.4f}, F stat={:.4f}, p-value={:.4f}'.format(w_lm, w_lm_pvalue, w_fvalue, w_f_pvalue))

# Goldfeld-Quandt
try:
    from statsmodels.stats.diagnostic import het_goldfeldquandt
    gq_test = het_goldfeldquandt(y, X)
    gq_f, gq_p, gq_label = gq_test
    print('\nGoldfeld-Quandt: F = {:.4f}, p-value (approx) = {:.4f}, alternative = {}'.format(gq_f, gq_p, gq_label))
except Exception as e:
    print('\nGoldfeld-Quandt test failed:', e)

# Park test: regress log(resid^2) on log(X_i)
resid_sq = resid**2
park_df = pd.DataFrame({'resid_sq': resid_sq})
for col in X_cols:
    # avoid nonpositive
    park_df['log_'+col] = np.log(np.abs(df[col]) + 1e-8)
park_df['log_resid_sq'] = np.log(park_df['resid_sq'] + 1e-8)
park_formula = 'log_resid_sq ~ ' + ' + '.join(['log_'+c for c in X_cols])
park_res = smf.ols(park_formula, data=park_df).fit()
print('\nPark test (regress log(resid^2) on log(Xs)):\n', park_res.summary())

# Glejser test: regress abs(resid) on predictors or their transforms
abs_resid = np.abs(resid)
glejser_df = df.copy()
glejser_df['abs_resid'] = abs_resid
glejser_formula = 'abs_resid ~ ' + ' + '.join(X_cols)
glejser_res = smf.ols(glejser_formula, data=glejser_df).fit()
print('\nGlejser-like test (abs residuals on X):\n', glejser_res.summary())

# 7) Мультиколлинеарность: парные корреляции и VIF
print('\nПарные корреляции между регрессорами:')
print(df[X_cols].corr())

# VIF
Xvif = sm.add_constant(df[X_cols])
vifs = pd.DataFrame()
vifs['variable'] = Xvif.columns
vifs['VIF'] = [variance_inflation_factor(Xvif.values, i) for i in range(Xvif.shape[1])]
print('\nVIFs:')
print(vifs)

# Condition number
cond_no = np.linalg.cond(Xvif.values)
print('\nCondition number of X matrix: {:.4f}'.format(cond_no))

# 8) Исправления при нарушениях
print('\n=== Коррекция автокорреляции и гетероскедастичности (автоматически) ===')

# Если автокорреляция есть (dw далеко от 2 или BG significant), предложим AR(1) GLS
autocorr_flag = (abs(dw - 2) > 0.2) or any([acorr_breusch_godfrey(model_ols, nlags=k)[1] < 0.05 for k in (1,2)])
hetero_flag = (bp_lm_pvalue < 0.05) or (w_lm_pvalue < 0.05)

print('autocorr_flag =', autocorr_flag, 'hetero_flag =', hetero_flag)

# 8a) Robust SEs if heterosked.
if hetero_flag:
    print('\nГетероскедастичность обнаружена: выводы с робастными стандартными ошибками (HC3)')
    print(model_ols.get_robustcov_results(cov_type='HC3').summary())

# 8b) GLSAR for AR(1) autocorrelation
if autocorr_flag:
    print('\nАвтокорреляция обнаружена: пробуем GLSAR(1)')
    try:
        glsar = sm.GLSAR(y, X, rho=1)
        glsar_res = glsar.iterative_fit(maxiter=10)
        print(glsar_res.summary())
    except Exception as e:
        print('GLSAR failed:', e)

# 8c) WLS using weights inversely proportional to estimated var(resid)
if hetero_flag:
    # Estimate variance function by regressing resid^2 on X and use predicted value as var
    aux = sm.OLS(resid_sq, X).fit()
    sigma2_hat = aux.fittedvalues.clip(lower=1e-8)
    weights = 1.0 / sigma2_hat
    wls_model = sm.WLS(y, X, weights=weights).fit()
    print('\nWLS (weights=1/hat_var) results:')
    print(wls_model.summary())

# 9) Краткие интерпретации по результатам (печатаются автоматически)
print('\n=== Автоматический вывод интерпретаций ===')

if abs(dw - 2) > 0.2:
    if dw < 1.5:
        print('- Durbin-Watson указывает на положительную автокорреляцию (DW={:.3f}).'.format(dw))
    elif dw > 2.5:
        print('- Durbin-Watson указывает на отрицательную автокорреляцию (DW={:.3f}).'.format(dw))
else:
    print('- Durbin-Watson не показывает сильной автокорреляции (DW={:.3f}).'.format(dw))

if any([acorr_breusch_godfrey(model_ols, nlags=k)[1] < 0.05 for k in (1,2,4)]):
    print('- Breusch-Godfrey: автокорреляция обнаружена для некоторого порядка лагов.')
else:
    print('- Breusch-Godfrey не обнаружил автокорреляцию (на проверенных лагах).')

if hetero_flag:
    print('- Обнаружена гетероскедастичность (Breusch-Pagan или White significant). Рекомендуется использовать робастные ошибки (HC) или WLS/GLS).')
else:
    print('- Явных признаков гетероскедастичности не обнаружено (BP/White не значимы).')

# Multicollinearity
high_vif = vifs[vifs['VIF']>10]
if not high_vif.empty:
    print('\n- Высокие VIF обнаружены для переменных:')
    print(high_vif)
    print('  Рекомендации: исключить переменные, объединить/центрировать, применить PCA или ridge-регрессию.')
else:
    print('\n- VIF в пределах нормы (нет сильной мультиколлинеарности).')

# 10) Сохранение результатов в CSV (остатки, fitted, diagnostics)
out = df.copy()
out['fitted'] = fitted
nout = out.copy()
out['resid'] = resid
out['resid_sq'] = resid_sq
out.to_csv('diagnostics_output.csv', index=False)
print('\nDiagnostics saved to diagnostics_output.csv')

