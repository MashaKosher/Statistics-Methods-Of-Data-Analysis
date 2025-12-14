# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import openpyxl
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

# 1. ЗАГРУЗКА ДАННЫХ
print("=" * 80)
print("1. ЗАГРУЗКА И ОЗНАКОМЛЕНИЕ С ДАННЫМИ")
print("=" * 80)

# try:
df = pd.read_excel('./Attrition-EDA.xlsm')

print(f"Размер датасета: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# Просмотр структуры данных
print("\nПервые 5 строк датасета:")
print(df.head())

print("\nИнформация о типах данных:")
print(df.info())

print("\nБазовые статистики числовых переменных:")
print(df.describe())

# 1.5 ПРЕОБРАЗОВАНИЕ ТИПОВ ДАННЫХ И ОБРАБОТКА ПРОПУСКОВ
print("\n" + "=" * 80)
print("1.5 ПРЕОБРАЗОВАНИЕ ТИПОВ ДАННЫХ И ОБРАБОТКА ПРОПУСКОВ")
print("=" * 80)

# Проверяем типы данных
print("\nТипы данных перед обработкой:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

# Функция для преобразования типов данных
def convert_column_types(df):
    df_clean = df.copy()
    
    # Список числовых колонок (на основе вашего кода)
    numeric_cols = [
        'Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
        'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
        'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
        'StockOptionLevel', 'WorkLifeBalance'
    ]
    
    # Преобразуем числовые колонки
    for col in numeric_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                print(f"  Преобразуем {col} из {df_clean[col].dtype} в числовой тип...")
                # Заменяем строковые представления пропусков
                df_clean[col] = df_clean[col].replace(['', ' ', 'NA', 'N/A', 'null', 'NULL'], np.nan)
                # Преобразуем в числовой тип
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Бинарные/категориальные колонки
    binary_prefixes = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
    binary_cols = [col for col in df_clean.columns if any(col.startswith(prefix) for prefix in binary_prefixes)]
    
    for col in binary_cols:
        if col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                print(f"  Преобразуем {col} из {df_clean[col].dtype} в числовой тип...")
                # Для бинарных переменных: если это строка, пробуем преобразовать в число
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                except:
                    # Если не получается, создаем бинарные признаки
                    unique_vals = df_clean[col].unique()
                    if len(unique_vals) == 2:
                        df_clean[col] = df_clean[col].map({unique_vals[0]: 0, unique_vals[1]: 1})
    
    return df_clean

# Преобразуем типы данных
df = convert_column_types(df)

# Проверяем пропуски после преобразования
print("\nПроверка пропусков после преобразования типов:")
missing_after = df.isnull().sum()
missing_cols = missing_after[missing_after > 0]
if len(missing_cols) > 0:
    print("Колонки с пропусками:")
    for col, count in missing_cols.items():
        percentage = (count / len(df)) * 100
        print(f"  {col}: {count} пропусков ({percentage:.2f}%)")
        
    # Заполняем пропуски
    print("\nЗаполнение пропусков...")
    
    # Для числовых колонок заполняем медианой
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  {col}: заполнено {df[col].isnull().sum()} пропусков медианой ({median_val:.2f})")
    
    # Для категориальных колонок заполняем модой
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"  {col}: заполнено {df[col].isnull().sum()} пропусков модой ('{mode_val}')")
else:
    print("Пропусков нет!")

# Проверяем типы данных после обработки
print("\nТипы данных после обработки:")
for col in df.columns[:20]:  # Выводим первые 20 колонок
    print(f"  {col}: {df[col].dtype}")
if len(df.columns) > 20:
    print(f"  ... и еще {len(df.columns) - 20} колонок")

# Проверяем уникальные значения для целевой переменной
if 'Attrition' in df.columns:
    print(f"\nУникальные значения Attrition: {df['Attrition'].unique()}")
    # Убедимся, что Attrition имеет правильные значения
    df['Attrition'] = df['Attrition'].astype(str).str.strip().str.title()
    print(f"Уникальные значения Attrition после очистки: {df['Attrition'].unique()}")

# 2. ОЧИСТКА ДАННЫХ
print("\n" + "=" * 80)
print("2. ОЧИСТКА И ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 80)

# 2.1 Проверка пропущенных значений
print("\n2.1 Проверка пропущенных значений:")
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

missing_df = pd.DataFrame({
    'Количество_пропусков': missing_values,
    'Процент_пропусков': missing_percentage
})

missing_data = missing_df[missing_df['Количество_пропусков'] > 0]
if len(missing_data) > 0:
    print(missing_data)
else:
    print("Пропущенных значений нет!")

# Визуализация пропущенных значений (если есть)
if missing_values.sum() > 0:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Матрица пропущенных значений', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2.2 Проверка дубликатов
print("\n2.2 Проверка дубликатов:")
duplicates = df.duplicated().sum()
print(f"Количество полных дубликатов: {duplicates}")

# 2.3 Анализ баланса классов
print("\n2.3 Анализ баланса классов (Attrition):")
attrition_counts = df['Attrition'].value_counts()
attrition_percentage = df['Attrition'].value_counts(normalize=True) * 100

print(f"Распределение Attrition:")
for val, count in attrition_counts.items():
    print(f"  {val}: {count} ({attrition_percentage[val]:.2f}%)")

# Визуализация баланса классов
plt.figure(figsize=(10, 6))
colors = ['#4CAF50', '#F44336']  # Зеленый для No, Красный для Yes
bars = plt.bar(attrition_counts.index, attrition_counts.values, color=colors, alpha=0.8)

plt.title('Распределение целевой переменной Attrition', fontsize=16, fontweight='bold')
plt.xlabel('Attrition', fontweight='bold')
plt.ylabel('Количество сотрудников', fontweight='bold')
plt.grid(axis='y', alpha=0.3)

# Добавление значений на столбцы
for bar, count in zip(bars, attrition_counts.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{count}\n({count/len(df)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('class_balance_attrition.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("ПРОВЕРКА И ОЧИСТКА ДАННЫХ")
print("=" * 80)

# Проверяем Attrition
if 'Attrition' in df.columns:
    print(f"Уникальные значения Attrition до очистки: {df['Attrition'].unique()}")
    
    # Очищаем и стандартизируем Attrition
    df['Attrition'] = df['Attrition'].astype(str).str.strip().str.title()
    print(f"Уникальные значения Attrition после очистки: {df['Attrition'].unique()}")
    
    # Проверяем, что есть оба значения
    attrition_counts = df['Attrition'].value_counts()
    print(f"Распределение Attrition: {attrition_counts.to_dict()}")
else:
    print("ВНИМАНИЕ: Столбец 'Attrition' не найден в данных!")

# Проверяем числовые колонки на NaN и inf
print("\nПроверка числовых колонок на проблемы:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:10]:  # Проверяем первые 10
    nan_count = df[col].isna().sum()
    inf_count = np.isinf(df[col]).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  {col}: {nan_count} NaN, {inf_count} inf/ -inf")
        
        # Заменяем inf на NaN, затем заполняем
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    Заполнено медианой: {median_val:.2f}")

# 3. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ
print("\n" + "=" * 80)
print("3. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ")
print("=" * 80)

# 3.1 Анализ числовых переменных
print("\n3.1 Анализ числовых переменных:")

# Разделим переменные на группы
ordinal_vars = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 
                'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 
                'StockOptionLevel', 'WorkLifeBalance']

continuous_vars = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome',
                   'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
                   'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
                   'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

binary_vars = [col for col in df.columns if col.startswith(('BusinessTravel', 'Department', 
               'EducationField', 'Gender', 'JobRole', 'MaritalStatus'))]

# ФИКС: Проверяем и преобразуем числовые колонки
print("\nПроверка типов данных для числовых переменных...")
for var in continuous_vars + ordinal_vars:
    if var in df.columns:
        if df[var].dtype == 'object':
            print(f"  Преобразуем {var} из {df[var].dtype} в числовой тип...")
            # Пробуем преобразовать в числовой тип, ошибки превращаем в NaN
            df[var] = pd.to_numeric(df[var], errors='coerce')
            
            # Заполняем пропущенные значения медианой
            if df[var].isnull().sum() > 0:
                median_val = df[var].median()
                df[var].fillna(median_val, inplace=True)
                print(f"    Заполнено {df[var].isnull().sum()} пропусков медианой: {median_val}")

# Визуализация распределения ключевых непрерывных переменных
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
key_continuous = ['Age', 'MonthlyIncome', 'DailyRate', 'TotalWorkingYears', 
                  'YearsAtCompany', 'DistanceFromHome', 'PercentSalaryHike', 
                  'NumCompaniesWorked', 'TrainingTimesLastYear']

# ФИКС: Фильтруем только существующие переменные
key_continuous = [var for var in key_continuous if var in df.columns]

for idx, var in enumerate(key_continuous):
    row = idx // 3
    col = idx % 3
    
    # Проверяем, что переменная числовая
    if pd.api.types.is_numeric_dtype(df[var]):
        # Гистограмма с KDE
        ax = axes[row, col]
        
        # ФИКС: Убедимся, что данные числовые
        data_to_plot = pd.to_numeric(df[var], errors='coerce').dropna()
        
        if len(data_to_plot) > 0:
            sns.histplot(data_to_plot, kde=True, ax=ax, bins=30, color='skyblue', edgecolor='black')
            ax.axvline(data_to_plot.mean(), color='red', linestyle='--', linewidth=2, label='Среднее')
            ax.axvline(data_to_plot.median(), color='green', linestyle='--', linewidth=2, label='Медиана')
            
            ax.set_title(f'Распределение {var}', fontweight='bold')
            ax.set_xlabel('')
            ax.legend(fontsize=8)
            
            # Добавление статистик
            stats_text = f'Mean: {data_to_plot.mean():.1f}\nMedian: {data_to_plot.median():.1f}\nStd: {data_to_plot.std():.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Нет данных для {var}', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'Распределение {var}', fontweight='bold')
    else:
        # Если переменная не числовая
        ax = axes[row, col]
        ax.text(0.5, 0.5, f'{var} не числовая\nТип: {df[var].dtype}', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title(f'Распределение {var}', fontweight='bold')

# Скрываем пустые оси, если переменных меньше 9
if len(key_continuous) < 9:
    for idx in range(len(key_continuous), 9):
        row = idx // 3
        col = idx % 3
        axes[row, col].set_visible(False)

plt.suptitle('Распределение ключевых непрерывных переменных', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('continuous_vars_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.2 Анализ порядковых переменных
print("\n3.2 Анализ порядковых (ординальных) переменных:")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
for idx, var in enumerate(ordinal_vars[:9]):  # Первые 9 порядковых переменных
    row = idx // 3
    col = idx % 3
    
    ax = axes[row, col]
    value_counts = df[var].value_counts().sort_index()
    
    bars = ax.bar(range(len(value_counts)), value_counts.values, 
                  color=plt.cm.Set3(np.arange(len(value_counts))/len(value_counts)))
    ax.set_title(f'{var} Distribution', fontweight='bold')
    ax.set_xlabel('Уровень/Оценка')
    ax.set_ylabel('Количество')
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index)
    
    # Добавление процентов
    total = len(df)
    for i, (x_pos, count) in enumerate(zip(range(len(value_counts)), value_counts.values)):
        percentage = (count / total) * 100
        ax.text(x_pos, count + 0.02 * max(value_counts.values), 
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)

plt.suptitle('Распределение порядковых переменных', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('ordinal_vars_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 3.3 Сравнение по группам Attrition
print("\n3.3 Сравнение ключевых показателей по группам Attrition:")

# Подготовка данных для сравнения
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
comparison_vars = ['Age', 'MonthlyIncome', 'DailyRate', 'TotalWorkingYears', 
                   'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                   'YearsAtCompany', 'DistanceFromHome']

# Фильтруем только существующие и числовые переменные
valid_comparison_vars = []
for var in comparison_vars:
    if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
        valid_comparison_vars.append(var)

print(f"Анализируем {len(valid_comparison_vars)} переменных из {len(comparison_vars)}")

for idx, var in enumerate(valid_comparison_vars):
    if idx < 9:  # Максимум 9 графиков
        row = idx // 3
        col = idx % 3
        
        ax = axes[row, col]
        
        try:
            # Создаем DataFrame для boxplot
            plot_data = []
            groups = []
            
            for attrition_val in ['Yes', 'No']:
                # Фильтруем данные для каждой группы
                mask = (df['Attrition'] == attrition_val) if 'Attrition' in df.columns else pd.Series([True] * len(df))
                subset = df.loc[mask, var]
                
                # Удаляем NaN и бесконечные значения
                subset_clean = subset.replace([np.inf, -np.inf], np.nan).dropna()
                
                if len(subset_clean) > 0:
                    plot_data.append(subset_clean)
                    groups.append(attrition_val)
            
            if len(plot_data) == 2 and len(plot_data[0]) > 0 and len(plot_data[1]) > 0:
                # Boxplot
                bp = ax.boxplot(plot_data, labels=groups, patch_artist=True, widths=0.6)
                
                # Настройка цветов
                colors = ['#FF6B6B', '#4ECDC4']  # Красный для Yes, бирюзовый для No
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                # Медианы
                for median in bp['medians']:
                    median.set(color='black', linewidth=2)
                
                ax.set_title(f'{var} по Attrition', fontweight='bold')
                ax.set_ylabel(var)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Добавление t-test статистики
                from scipy import stats
                try:
                    yes_data = df[df['Attrition'] == 'Yes'][var].replace([np.inf, -np.inf], np.nan).dropna()
                    no_data = df[df['Attrition'] == 'No'][var].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(yes_data) > 1 and len(no_data) > 1:
                        t_stat, p_value = stats.ttest_ind(yes_data, no_data, equal_var=False, nan_policy='omit')
                        stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                        ax.text(0.5, 0.95, f'p = {p_value:.4f} {stars}', 
                                transform=ax.transAxes, ha='center', va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                        
                        # Вывод статистики в консоль
                        print(f"{var}: Yes (n={len(yes_data)}): mean={yes_data.mean():.2f}, "
                              f"No (n={len(no_data)}): mean={no_data.mean():.2f}, p={p_value:.4f}")
                except Exception as e:
                    print(f"  Ошибка t-теста для {var}: {str(e)}")
            else:
                ax.text(0.5, 0.5, 'Недостаточно данных', 
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{var} по Attrition', fontweight='bold')
                ax.set_ylabel(var)
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Ошибка: {str(e)[:30]}...', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{var} по Attrition', fontweight='bold')
            ax.set_ylabel(var)
    else:
        break

# Скрываем пустые оси
for idx in range(len(valid_comparison_vars), 9):
    row = idx // 3
    col = idx % 3
    axes[row, col].set_visible(False)

plt.suptitle('Сравнение показателей по Attrition (с t-тестами)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('attrition_comparison_ttest.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
print("\n" + "=" * 80)
print("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("=" * 80)

# Создаем числовую версию целевой переменной
df_numeric = df.copy()

# Убедимся, что Attrition имеет правильные значения
if 'Attrition' in df_numeric.columns:
    # Приводим к стандартному виду
    df_numeric['Attrition'] = df_numeric['Attrition'].astype(str).str.strip().str.title()
    
    # Создаем числовую версию
    attrition_map = {'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0}
    df_numeric['Attrition_numeric'] = df_numeric['Attrition'].map(attrition_map)
    
    # Проверяем результат
    print(f"Уникальные значения Attrition_numeric: {df_numeric['Attrition_numeric'].unique()}")
    
    # Если есть NaN в Attrition_numeric, заполняем наиболее частым значением
    if df_numeric['Attrition_numeric'].isnull().sum() > 0:
        print(f"Предупреждение: {df_numeric['Attrition_numeric'].isnull().sum()} строк с некорректными значениями Attrition")
        mode_val = df_numeric['Attrition_numeric'].mode()[0] if not df_numeric['Attrition_numeric'].mode().empty else 0
        df_numeric['Attrition_numeric'].fillna(mode_val, inplace=True)
        print(f"  Заполнено значением: {mode_val}")
else:
    print("ОШИБКА: Столбец 'Attrition' не найден!")
    # Создаем фиктивный столбец для продолжения анализа
    df_numeric['Attrition_numeric'] = 0

# Выбираем числовые колонки для корреляционного анализа
# Фильтруем только существующие и числовые колонки
numeric_for_corr = []
for var in continuous_vars + ordinal_vars:
    if var in df_numeric.columns and pd.api.types.is_numeric_dtype(df_numeric[var]):
        numeric_for_corr.append(var)

# Добавляем целевую переменную
if 'Attrition_numeric' in df_numeric.columns:
    numeric_for_corr.append('Attrition_numeric')

print(f"\nИспользуем {len(numeric_for_corr)} числовых переменных для корреляционного анализа")

# Проверяем, что достаточно данных
if len(numeric_for_corr) < 2:
    print("ОШИБКА: Недостаточно числовых переменных для корреляционного анализа!")
else:
    # Создаем корреляционную матрицу
    corr_matrix = df_numeric[numeric_for_corr].corr()
    
    print(f"\nРазмер корреляционной матрицы: {corr_matrix.shape}")
    
    # 4.1 Тепловая карта всех корреляций
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot=False)
    
    plt.title('Тепловая карта корреляций между числовыми переменными', 
              fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('full_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4.2 Корреляции с Attrition
    if 'Attrition_numeric' in corr_matrix.columns:
        print("\n4.2 Переменные с наибольшей корреляцией с Attrition:")
        attrition_corr = corr_matrix['Attrition_numeric'].sort_values(ascending=False)
        
        # Топ положительных и отрицательных корреляций
        print("\nТоп-10 положительных корреляций:")
        top_pos = attrition_corr.head(11)  # 11 потому что первое - сам Attrition
        print(top_pos.to_string())
        
        print("\nТоп-10 отрицательных корреляций:")
        top_neg = attrition_corr.tail(10).sort_values(ascending=True)
        print(top_neg.to_string())
        
        # Визуализация топ-15 корреляций с Attrition
        # Исключаем сам Attrition_numeric
        top_corr_values = attrition_corr.drop('Attrition_numeric', errors='ignore')
        
        if len(top_corr_values) > 0:
            # Берем топ 8 положительных и топ 7 отрицательных
            n_pos = min(8, len(top_corr_values[top_corr_values > 0]))
            n_neg = min(7, len(top_corr_values[top_corr_values < 0]))
            
            top_pos = top_corr_values[top_corr_values > 0].head(n_pos)
            top_neg = top_corr_values[top_corr_values < 0].tail(n_neg)
            
            top_corr = pd.concat([top_pos, top_neg]).sort_values()
            
            plt.figure(figsize=(12, 8))
            
            colors = ['red' if x < 0 else 'green' for x in top_corr.values]
            bars = plt.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
            
            plt.yticks(range(len(top_corr)), top_corr.index)
            plt.xlabel('Коэффициент корреляции Пирсона', fontweight='bold')
            plt.title('Топ корреляций с Attrition', fontsize=16, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            plt.grid(axis='x', alpha=0.3)
            
            # Добавление значений
            for i, (bar, val) in enumerate(zip(bars, top_corr.values)):
                plt.text(val + (0.01 if val >= 0 else -0.03), i, 
                         f'{val:.3f}', 
                         va='center', 
                         color='black' if abs(val) < 0.15 else 'white',
                         fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('top_correlations_with_attrition.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("\nНет корреляций для визуализации")
    else:
        print("\nAttrition_numeric не найден в корреляционной матрице")

# 5. АНАЛИЗ КАТЕГОРИАЛЬНЫХ (БИНАРНЫХ) ПЕРЕМЕННЫХ
print("\n" + "=" * 80)
print("5. АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ")
print("=" * 80)

# 5.1 Анализ влияния бинарных переменных на Attrition
print("\n5.1 Влияние бинарных переменных на Attrition:")

# Функция для расчета отношения шансов (Odds Ratio) - ИСПРАВЛЕННАЯ ВЕРСИЯ
def calculate_odds_ratio(df, variable, target='Attrition'):
    try:
        # Создаем временный DataFrame
        temp_df = df[[variable, target]].copy()
        
        # Убедимся, что target в правильном формате
        if temp_df[target].dtype == 'object':
            # Преобразуем target в бинарный
            temp_df[target] = temp_df[target].astype(str).str.strip().str.title()
            target_map = {'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0, 'yes': 1, 'no': 0}
            temp_df[target] = temp_df[target].map(target_map)
        
        # Удаляем строки с NaN в целевой переменной
        temp_df = temp_df.dropna(subset=[target])
        
        if len(temp_df) < 10:  # Минимальное количество данных
            return None
        
        # Для переменной: если это число, используем как есть
        if pd.api.types.is_numeric_dtype(temp_df[variable]):
            # Проверяем, что значения 0 и 1
            unique_vals = temp_df[variable].dropna().unique()
            unique_vals = [v for v in unique_vals if not pd.isna(v)]
            
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                # Уже бинарная
                pass
            else:
                # Преобразуем в бинарную (медиана как порог)
                median_val = temp_df[variable].median()
                temp_df[variable] = (temp_df[variable] > median_val).astype(int)
        else:
            # Если не числовая, преобразуем в бинарную
            unique_vals = temp_df[variable].dropna().unique()
            unique_vals = [str(v) for v in unique_vals if not pd.isna(v)]
            
            if len(unique_vals) >= 2:
                # Берем два самых частых значения
                value_counts = temp_df[variable].value_counts()
                top_values = value_counts.index[:2].tolist()
                
                # Создаем mapping
                val_map = {str(top_values[0]): 0, str(top_values[1]): 1}
                for val in unique_vals:
                    if str(val) not in val_map:
                        val_map[str(val)] = 0  # остальные относим к группе 0
                
                temp_df[variable] = temp_df[variable].astype(str).map(val_map)
            else:
                return None
        
        # Удаляем строки с NaN в переменной
        temp_df = temp_df.dropna(subset=[variable])
        
        if len(temp_df) < 10:  # Слишком мало данных
            return None
        
        # Создаем таблицу сопряженности
        contingency_table = pd.crosstab(temp_df[variable], temp_df[target])
        
        # Убедимся, что таблица 2x2
        if contingency_table.shape != (2, 2):
            # Если не 2x2, возможно нужно перекодировать
            if len(contingency_table) == 2 and len(contingency_table.columns) == 2:
                pass  # Уже 2x2
            else:
                return None
        
        # Извлекаем значения
        try:
            a = contingency_table.loc[1, 1] if 1 in contingency_table.index else 0
            b = contingency_table.loc[1, 0] if 1 in contingency_table.index else 0
            c = contingency_table.loc[0, 1] if 0 in contingency_table.index else 0
            d = contingency_table.loc[0, 0] if 0 in contingency_table.index else 0
        except KeyError:
            return None
        
        # Расчет odds ratio с защитой от деления на 0
        if b == 0 or c == 0:
            odds_ratio = np.nan
        else:
            odds_ratio = (a * d) / (b * c)
        
        # Расчет rates
        attrition_rate_1 = a / (a + b) if (a + b) > 0 else 0
        attrition_rate_0 = c / (c + d) if (c + d) > 0 else 0
        
        return {
            'variable': variable,
            'odds_ratio': odds_ratio,
            'attrition_rate_1': attrition_rate_1,
            'attrition_rate_0': attrition_rate_0,
            'count_1': int(a + b),
            'count_0': int(c + d)
        }
    
    except Exception as e:
        # print(f"  Ошибка при расчете odds ratio для {variable}: {str(e)}")
        return None

# Анализ всех бинарных переменных
odds_ratios = []
print("Расчет odds ratios для бинарных переменных...")
for idx, var in enumerate(binary_vars):
    if var in df.columns:
        result = calculate_odds_ratio(df, var)
        if result is not None and not pd.isna(result['odds_ratio']):
            odds_ratios.append(result)

# Создаем DataFrame с результатами - с проверкой на пустоту
if odds_ratios:
    odds_df = pd.DataFrame(odds_ratios)
    
    # Проверяем, есть ли столбец odds_ratio
    if 'odds_ratio' in odds_df.columns and len(odds_df) > 0:
        # Удаляем строки с NaN в odds_ratio
        odds_df_clean = odds_df.dropna(subset=['odds_ratio'])
        
        if len(odds_df_clean) > 0:
            odds_df = odds_df_clean.sort_values('odds_ratio', ascending=False)
            
            print(f"\nНайдено {len(odds_df)} переменных с корректными odds ratios")
            print("\nТоп-10 отношений шансов (Odds Ratios) для бинарных переменных:")
            print(odds_df[['variable', 'odds_ratio', 'attrition_rate_1', 'attrition_rate_0']].head(10).to_string())
        else:
            print("\nНет корректных значений odds ratio для отображения")
            odds_df = pd.DataFrame()  # Создаем пустой DataFrame
    else:
        print("\nНе удалось рассчитать odds ratios")
        odds_df = pd.DataFrame()
else:
    print("\nНе удалось рассчитать ни одного odds ratio")
    odds_df = pd.DataFrame()

# 5.2 Визуализация отношения шансов
if not odds_df.empty and len(odds_df) > 0 and 'odds_ratio' in odds_df.columns:
    plt.figure(figsize=(14, 10))
    
    # Берем топ 10 или меньше
    n_to_show = min(15, len(odds_df))
    top_odds = odds_df.head(n_to_show).sort_values('odds_ratio')
    
    # Создаем горизонтальную диаграмму
    y_pos = np.arange(len(top_odds))
    colors = ['red' if x < 1 else 'green' for x in top_odds['odds_ratio']]
    
    bars = plt.barh(y_pos, top_odds['odds_ratio'], color=colors, alpha=0.7)
    
    plt.yticks(y_pos, top_odds['variable'])
    plt.xlabel('Отношение шансов (Odds Ratio)', fontweight='bold')
    plt.title('Влияние бинарных переменных на Attrition (Odds Ratio)', 
              fontsize=16, fontweight='bold')
    plt.axvline(x=1, color='black', linestyle='--', linewidth=2)
    plt.grid(axis='x', alpha=0.3)
    
    # Добавление аннотаций
    for i, (bar, odds, rate1, rate0) in enumerate(zip(bars, top_odds['odds_ratio'], 
                                                       top_odds['attrition_rate_1'], 
                                                       top_odds['attrition_rate_0'])):
        annotation = f'OR: {odds:.2f}\nRate(1): {rate1:.1%}\nRate(0): {rate0:.1%}'
        x_pos = odds + 0.1 if odds < 10 else odds - 0.5
        plt.text(x_pos, i, 
                 annotation, 
                 va='center', fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('odds_ratios_binary_vars.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("\nПропускаем визуализацию odds ratios: недостаточно данных")
    
    # Вместо этого создаем простую визуализацию Attrition по бинарным переменным
    plt.figure(figsize=(14, 10))
    
    # Берем первые 6 бинарных переменных
    vars_to_plot = binary_vars[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, var in enumerate(vars_to_plot):
        if idx < len(axes) and var in df.columns:
            ax = axes[idx]
            
            try:
                # Создаем сводную таблицу
                if pd.api.types.is_numeric_dtype(df[var]):
                    # Для числовых бинарных
                    temp_df = df[[var, 'Attrition']].copy()
                    temp_df = temp_df.dropna()
                    
                    if len(temp_df) > 0:
                        # Преобразуем в бинарные группы
                        if temp_df[var].nunique() > 2:
                            # Если больше 2 значений, используем медиану как порог
                            median_val = temp_df[var].median()
                            temp_df[var] = (temp_df[var] > median_val).astype(int)
                        
                        pivot = temp_df.groupby(var)['Attrition'].value_counts(normalize=True).unstack()
                        
                        if 'Yes' in pivot.columns and len(pivot) == 2:
                            pivot['Yes'].plot(kind='bar', ax=ax, color=['skyblue', 'salmon'], alpha=0.7)
                            ax.set_xlabel(var)
                            ax.set_ylabel('Процент Attrition')
                            ax.set_title(f'Attrition по {var}')
                            ax.grid(axis='y', alpha=0.3)
                            
                            # Добавить значения на столбцы
                            for i, val in enumerate(pivot['Yes']):
                                ax.text(i, val + 0.01, f'{val:.1%}', 
                                        ha='center', va='bottom', fontweight='bold')
                        else:
                            ax.text(0.5, 0.5, 'Недостаточно данных\nили не бинарная', 
                                    transform=ax.transAxes, ha='center', va='center')
                            ax.set_title(f'{var}')
                    else:
                        ax.text(0.5, 0.5, 'Нет данных', 
                                transform=ax.transAxes, ha='center', va='center')
                        ax.set_title(f'{var}')
                else:
                    ax.text(0.5, 0.5, 'Не числовая\nпеременная', 
                            transform=ax.transAxes, ha='center', va='center')
                    ax.set_title(f'{var}')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Ошибка\n{str(e)[:20]}', 
                        transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{var}')
    
    # Скрыть пустые оси
    for idx in range(len(vars_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Распределение Attrition по бинарным переменным', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('binary_vars_attrition.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. КЛАСТЕРНЫЙ АНАЛИЗ
print("\n" + "=" * 80)
print("6. КЛАСТЕРНЫЙ АНАЛИЗ")
print("=" * 80)

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 6.1 Подготовка данных для кластеризации
print("\n6.1 Подготовка данных для кластеризации...")

# Выбираем ключевые переменные для кластеризации
cluster_features = [
    'Age',
    'MonthlyIncome', 
    'TotalWorkingYears',
    'YearsAtCompany',
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'WorkLifeBalance',
    'JobLevel',
    'StockOptionLevel',
    'DistanceFromHome',
    'PercentSalaryHike'
]

# Проверяем, что все переменные существуют
available_features = [f for f in cluster_features if f in df.columns]
print(f"Используем {len(available_features)} признаков для кластеризации")

X_cluster = df[available_features].copy()

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# 6.2 Определение оптимального числа кластеров
print("\n6.2 Определение оптимального числа кластеров...")

# Метод локтя и силуэтный анализ
wcss = []  # Within-cluster sum of squares
silhouette_scores = []
db_scores = []  # Davies-Bouldin scores
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
    
    if k > 1:
        silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        
        db_score = davies_bouldin_score(X_scaled, kmeans.labels_)
        db_scores.append(db_score)
    
    print(f"k={k}: WCSS={kmeans.inertia_:.2f}, ", end="")
    if k > 1:
        print(f"Silhouette={silhouette_avg:.3f}, DB={db_score:.3f}")
    else:
        print()

# Визуализация методов определения k
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Метод локтя
axes[0].plot(K_range, wcss, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Количество кластеров (k)', fontweight='bold')
axes[0].set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontweight='bold')
axes[0].set_title('Метод локтя', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Силуэтный анализ
axes[1].plot(range(2, 11), silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Количество кластеров (k)', fontweight='bold')
axes[1].set_ylabel('Средний силуэтный коэффициент', fontweight='bold')
axes[1].set_title('Силуэтный анализ', fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin Index
axes[2].plot(range(2, 11), db_scores, 'go-', linewidth=2, markersize=8)
axes[2].set_xlabel('Количество кластеров (k)', fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index', fontweight='bold')
axes[2].set_title('Davies-Bouldin Index (меньше = лучше)', fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('Определение оптимального числа кластеров', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('optimal_clusters_determination.png', dpi=300, bbox_inches='tight')
plt.show()

# Выбираем оптимальное k (по силуэтному коэффициенту)
optimal_k = range(2, 11)[np.argmax(silhouette_scores)]
print(f"\nОптимальное число кластеров (по силуэтному коэффициенту): {optimal_k}")

# 6.3 Применение K-means с оптимальным k
print(f"\n6.3 Применение K-means с k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=300)
cluster_labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = cluster_labels

# 6.4 Снижение размерности для визуализации
print("\n6.4 Снижение размерности и визуализация кластеров...")

# PCA для визуализации
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Объясненная дисперсия PCA: {pca.explained_variance_ratio_.sum():.3f}")
print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")

# t-SNE для нелинейного снижения размерности
print("Применение t-SNE...")
try:
    # Для новых версий scikit-learn (>= 0.24)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, learning_rate=200)
except TypeError:
    try:
        # Для версий 0.19-0.23
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, learning_rate=200)
    except TypeError:
        # Минимальная конфигурация
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        print("Используем упрощенную конфигурацию t-SNE")

try:
    X_tsne = tsne.fit_transform(X_scaled)
    print("t-SNE успешно выполнен")
except Exception as e:
    print(f"Ошибка при выполнении t-SNE: {e}")
    print("Используем PCA для визуализации вместо t-SNE")
    X_tsne = X_pca  # Используем PCA как запасной вариант

# Визуализация кластеров
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# PCA визуализация
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
axes[0].set_title('Кластеры в PCA пространстве', fontweight='bold')
plt.colorbar(scatter1, ax=axes[0], label='Кластер')

# t-SNE визуализация
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], 
                          c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
axes[1].set_xlabel('t-SNE Component 1')
axes[1].set_ylabel('t-SNE Component 2')
axes[1].set_title('Кластеры в t-SNE пространстве', fontweight='bold')
plt.colorbar(scatter2, ax=axes[1], label='Кластер')

# Распределение Attrition по кластерам
cluster_attrition = pd.crosstab(df['Cluster'], df['Attrition'], normalize='index') * 100
cluster_attrition.plot(kind='bar', stacked=True, ax=axes[2], 
                      color=['#4CAF50', '#F44336'])
axes[2].set_xlabel('Кластер', fontweight='bold')
axes[2].set_ylabel('Процент (%)', fontweight='bold')
axes[2].set_title('Распределение Attrition по кластерам', fontweight='bold')
axes[2].legend(title='Attrition')
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle(f'Кластерный анализ (k={optimal_k})', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()
# 6.5 Анализ характеристик кластеров
print("\n6.5 Характеристики кластеров:")

# Средние значения по кластерам
cluster_profiles = df.groupby('Cluster')[available_features].mean()
print("\nСредние значения по кластерам для ключевых переменных:")
print(cluster_profiles.round(2))

# Размеры кластеров
print("\nРазмеры кластеров:")
print(df['Cluster'].value_counts().sort_index())

# Визуализация профилей кластеров (Heatmap)
plt.figure(figsize=(14, 8))
sns.heatmap(cluster_profiles.T, annot=True, cmap='YlOrRd', 
            fmt='.1f', linewidths=0.5, cbar_kws={'label': 'Среднее значение'})
plt.title('Профили кластеров (средние значения признаков)', fontsize=16, fontweight='bold')
plt.xlabel('Кластер')
plt.ylabel('Признак')
plt.tight_layout()
plt.savefig('cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. СНИЖЕНИЕ РАЗМЕРНОСТИ И ВИЗУАЛИЗАЦИЯ
print("\n" + "=" * 80)
print("7. PCA АНАЛИЗ И СНИЖЕНИЕ РАЗМЕРНОСТИ")
print("=" * 80)

from sklearn.decomposition import PCA

# 7.1 Полный PCA анализ
print("\n7.1 PCA анализ всех числовых переменных...")

# Используем все числовые переменные для PCA
all_numeric_vars = continuous_vars + ordinal_vars
X_pca_full = df[all_numeric_vars].copy()

# Масштабирование
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_pca_full)

# PCA со всеми компонентами
pca_full = PCA()
X_pca_transformed = pca_full.fit_transform(X_pca_scaled)

# Анализ объясненной дисперсии
explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# 7.2 Визуализация объясненной дисперсии
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scree plot
axes[0].plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-', linewidth=2)
axes[0].set_xlabel('Номер главной компоненты', fontweight='bold')
axes[0].set_ylabel('Объясненная дисперсия', fontweight='bold')
axes[0].set_title('Scree Plot (объясненная дисперсия)', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Кумулятивная дисперсия
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-', linewidth=2)
axes[1].axhline(y=0.8, color='g', linestyle='--', label='80% variance')
axes[1].axhline(y=0.9, color='b', linestyle='--', label='90% variance')
axes[1].axhline(y=0.95, color='purple', linestyle='--', label='95% variance')
axes[1].set_xlabel('Количество главных компонент', fontweight='bold')
axes[1].set_ylabel('Кумулятивная объясненная дисперсия', fontweight='bold')
axes[1].set_title('Кумулятивная объясненная дисперсия', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA анализ: объясненная дисперсия', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Определение числа компонент для 80%, 90%, 95% дисперсии
n_components_80 = np.where(cumulative_variance >= 0.8)[0][0] + 1
n_components_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
n_components_95 = np.where(cumulative_variance >= 0.95)[0][0] + 1

print(f"\nДля объяснения 80% дисперсии нужно {n_components_80} компонент")
print(f"Для объяснения 90% дисперсии нужно {n_components_90} компонент")
print(f"Для объяснения 95% дисперсии нужно {n_components_95} компонент")

# 7.3 Анализ нагрузок на первые две компоненты
print("\n7.3 Анализ нагрузок на главные компоненты...")

# Создаем DataFrame с нагрузками
loadings = pd.DataFrame(
    pca_full.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=all_numeric_vars
)

# Сортируем по абсолютной величине нагрузок
print("\nТоп-10 переменных с наибольшими нагрузками на PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False).head(10))

print("\nТоп-10 переменных с наибольшими нагрузками на PC2:")
print(loadings['PC2'].abs().sort_values(ascending=False).head(10))

# Визуализация нагрузок
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Нагрузки на PC1
top_pc1 = loadings['PC1'].abs().sort_values(ascending=False).head(15)
colors_pc1 = ['red' if x < 0 else 'green' for x in loadings.loc[top_pc1.index, 'PC1']]
axes[0].barh(range(len(top_pc1)), loadings.loc[top_pc1.index, 'PC1'], color=colors_pc1)
axes[0].set_yticks(range(len(top_pc1)))
axes[0].set_yticklabels(top_pc1.index)
axes[0].set_xlabel('Нагрузка', fontweight='bold')
axes[0].set_title(f'Нагрузки на PC1 ({explained_variance[0]*100:.1f}% variance)', fontweight='bold')
axes[0].axvline(x=0, color='black', linewidth=0.5)
axes[0].grid(axis='x', alpha=0.3)

# Нагрузки на PC2
top_pc2 = loadings['PC2'].abs().sort_values(ascending=False).head(15)
colors_pc2 = ['red' if x < 0 else 'green' for x in loadings.loc[top_pc2.index, 'PC2']]
axes[1].barh(range(len(top_pc2)), loadings.loc[top_pc2.index, 'PC2'], color=colors_pc2)
axes[1].set_yticks(range(len(top_pc2)))
axes[1].set_yticklabels(top_pc2.index)
axes[1].set_xlabel('Нагрузка', fontweight='bold')
axes[1].set_title(f'Нагрузки на PC2 ({explained_variance[1]*100:.1f}% variance)', fontweight='bold')
axes[1].axvline(x=0, color='black', linewidth=0.5)
axes[1].grid(axis='x', alpha=0.3)

plt.suptitle('Нагрузки переменных на главные компоненты', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pca_loadings_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7.4 Biplot (комбинированная визуализация)
print("\n7.4 Biplot визуализация...")

# Создаем biplot
plt.figure(figsize=(14, 10))

# Точечная диаграмма
scatter = plt.scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], 
                     c=df_numeric['Attrition_numeric'], cmap='coolwarm', 
                     alpha=0.6, s=50, edgecolor='k', linewidth=0.5)

# Векторы нагрузок
scale_factor = 5  # Масштаб для векторов
for i, feature in enumerate(all_numeric_vars):
    plt.arrow(0, 0, loadings.loc[feature, 'PC1'] * scale_factor, 
              loadings.loc[feature, 'PC2'] * scale_factor,
              color='black', alpha=0.5, head_width=0.05)
    plt.text(loadings.loc[feature, 'PC1'] * scale_factor * 1.15,
             loadings.loc[feature, 'PC2'] * scale_factor * 1.15,
             feature, color='darkblue', fontsize=9, fontweight='bold')

plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)', fontweight='bold')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)', fontweight='bold')
plt.title('Biplot: PCA проекция с нагрузками переменных', fontsize=16, fontweight='bold')
plt.colorbar(scatter, label='Attrition (0=No, 1=Yes)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('pca_biplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. ПОСТРОЕНИЕ ПРОГНОЗИРУЮЩЕЙ МОДЕЛИ
print("\n" + "=" * 80)
print("8. ПОСТРОЕНИЕ ПРОГНОЗИРУЮЩЕЙ МОДЕЛИ ДЛЯ ATTRITION")
print("=" * 80)

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_curve, auc, roc_auc_score, precision_recall_curve,
                           f1_score, precision_score, recall_score)

# Проверка данных перед моделированием
print("\n" + "=" * 80)
print("ПРОВЕРКА ДАННЫХ ПЕРЕД МОДЕЛИРОВАНИЕМ")
print("=" * 80)

print(f"Общий размер датасета: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# Проверяем целевую переменную
if 'Attrition' in df.columns:
    print(f"\nАнализ целевой переменной Attrition:")
    print(f"  Тип данных: {df['Attrition'].dtype}")
    print(f"  Уникальные значения: {df['Attrition'].unique()}")
    
    # Подсчет распределения
    attrition_counts = df['Attrition'].value_counts()
    for val, count in attrition_counts.items():
        percentage = count / len(df) * 100
        print(f"  '{val}': {count} ({percentage:.1f}%)")
    
    # Проверяем на NaN
    nan_count = df['Attrition'].isna().sum()
    if nan_count > 0:
        print(f"  ВНИМАНИЕ: Найдено {nan_count} NaN значений в Attrition!")
        
        # Заполняем наиболее частым значением
        mode_val = df['Attrition'].mode()[0] if not df['Attrition'].mode().empty else 'No'
        print(f"  Заполняем NaN значением '{mode_val}'...")
        df['Attrition'].fillna(mode_val, inplace=True)
    
    # Проверяем на корректность значений
    valid_values = ['Yes', 'No', 'YES', 'NO', 'yes', 'no', '1', '0']
    invalid_count = df[~df['Attrition'].astype(str).isin(valid_values)].shape[0]
    if invalid_count > 0:
        print(f"  ВНИМАНИЕ: Найдено {invalid_count} некорректных значений в Attrition!")
        
        # Приводим к стандартному виду
        print("  Приводим к стандартному виду...")
        df['Attrition'] = df['Attrition'].astype(str).str.strip().str.title()
        
        # Проверяем еще раз
        invalid_values = df[~df['Attrition'].isin(['Yes', 'No'])]['Attrition'].unique()
        if len(invalid_values) > 0:
            print(f"  Некорректные значения после обработки: {invalid_values}")
            print("  Заменяем некорректные значения на 'No'...")
            df['Attrition'] = df['Attrition'].apply(lambda x: x if x in ['Yes', 'No'] else 'No')

print("\nПроверка числовых признаков на проблемы:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
problem_cols = []

for col in numeric_cols:
    nan_count = df[col].isna().sum()
    inf_count = np.isinf(df[col]).sum()
    
    if nan_count > 0 or inf_count > 0:
        problem_cols.append((col, nan_count, inf_count))

if len(problem_cols) > 0:
    print(f"  Найдено {len(problem_cols)} проблемных колонок:")
    for col, nan_count, inf_count in problem_cols[:10]:  # Показываем первые 10
        print(f"    {col}: {nan_count} NaN, {inf_count} inf/-inf")
    
    print("\n  Исправление проблемных колонок...")
    for col in numeric_cols:
        # Заменяем inf на NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Заполняем NaN медианой
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
    
    print("  Проблемы исправлены!")
else:
    print("  Проблемных колонок не найдено")

print("\nПроверка категориальных признаков:")
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"  Найдено {len(categorical_cols)} категориальных колонок:")
    for col in categorical_cols[:5]:  # Показываем первые 5
        print(f"    {col}: {df[col].dtype}, уникальных значений: {df[col].nunique()}")
else:
    print("  Категориальных колонок не найдено")

# 8.1 Подготовка данных для моделирования
print("\n8.1 Подготовка данных для моделирования...")

# Создаем бинарную целевую переменную
print("Преобразование целевой переменной Attrition...")
print(f"Уникальные значения Attrition: {df['Attrition'].unique()}")

# Приводим Attrition к стандартному виду
df['Attrition'] = df['Attrition'].astype(str).str.strip().str.title()

# Создаем бинарную целевую переменную
attrition_map = {'Yes': 1, 'No': 0, 'YES': 1, 'NO': 0, 'yes': 1, 'no': 0, '1': 1, '0': 0, 1: 1, 0: 0}
y = df['Attrition'].map(attrition_map)

# Проверяем результат
print(f"\nРаспределение целевой переменной:")
print(f"  Всего записей: {len(y)}")
print(f"  Attrition=1 (Yes): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"  Attrition=0 (No): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"  NaN значений: {y.isna().sum()}")

# Удаляем строки с NaN в целевой переменной
if y.isna().sum() > 0:
    print(f"\nУдаляем {y.isna().sum()} строк с NaN в целевой переменной...")
    valid_indices = y.dropna().index
    df = df.loc[valid_indices].copy()
    y = y.loc[valid_indices].copy()
    
    print(f"  Осталось записей: {len(y)}")
    print(f"  Attrition=1 (Yes): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    print(f"  Attrition=0 (No): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")

# Подготовка признаков
print("\nПодготовка матрицы признаков...")
X = df.drop(['Attrition', 'Cluster', 'Attrition_numeric'] if 'Attrition_numeric' in df.columns else ['Attrition', 'Cluster'], 
            axis=1, errors='ignore')

# Проверяем наличие нечисловых колонок
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    print(f"Найдены нечисловые колонки ({len(non_numeric_cols)}): {list(non_numeric_cols)}")
    X = pd.get_dummies(X, drop_first=True)
    print(f"После one-hot encoding: {X.shape[1]} признаков")
else:
    print("Все колонки уже числовые")

print(f"\nРазмер матрицы признаков: {X.shape}")
print(f"Баланс классов: {y.value_counts().to_dict()}")

# Проверяем на NaN в признаках
nan_in_X = X.isna().sum().sum()
if nan_in_X > 0:
    print(f"\nПредупреждение: в матрице признаков найдено {nan_in_X} NaN значений")
    print("Заполняем NaN медианами по столбцам...")
    for col in X.columns:
        if X[col].isna().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

# Проверяем на бесконечные значения
inf_in_X = np.isinf(X.values).sum()
if inf_in_X > 0:
    print(f"\nПредупреждение: в матрице признаков найдено {inf_in_X} inf/-inf значений")
    print("Заменяем inf/-inf на NaN, затем заполняем медианами...")
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].isna().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)

# Разделение данных
print("\nРазделение данных на тренировочную и тестовую выборки...")
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Размер тренировочной выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Текучесть в тренировочной выборке: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
    print(f"Текучесть в тестовой выборке: {y_test.mean():.3f} ({y_test.sum()}/{len(y_test)})")
    
except ValueError as e:
    print(f"Ошибка при разделении данных: {e}")
    print("Пробуем разделение без stratify...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"Размер тренировочной выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Текучесть в тренировочной выборке: {y_train.mean():.3f} ({y_train.sum()}/{len(y_train)})")
    print(f"Текучесть в тестовой выборке: {y_test.mean():.3f} ({y_test.sum()}/{len(y_test)})")

# 8.2 Обучение и сравнение нескольких моделей
print("\n8.2 Обучение и сравнение моделей...")

# Определяем модели
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced_subsample'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
}

# Результаты
results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Кросс-валидация
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"ROC-AUC (кросс-валидация): {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    
    # Обучение на всей тренировочной выборке
    model.fit(X_train, y_train)
    
    # Прогнозы
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Метрики
    accuracy = np.mean(y_pred == y_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC (тест): {roc_auc:.3f}")
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Сохранение результатов
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'cv_score': cv_scores.mean(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc if 'roc_auc' in locals() else None
    }

# 8.3 Визуализация результатов моделей
print("\n8.3 Визуализация результатов моделей...")

# Создаем фигуру для визуализации
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.3.1 Сравнение метрик моделей
metrics_comparison = pd.DataFrame({
    'Model': list(results.keys()),
    'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] is not None else 0 for m in results],
    'Accuracy': [results[m]['accuracy'] for m in results],
    'Precision': [results[m]['precision'] for m in results],
    'Recall': [results[m]['recall'] for m in results],
    'F1-Score': [results[m]['f1'] for m in results]
}).set_index('Model')

# Heatmap сравнения метрик
sns.heatmap(metrics_comparison, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 0])
axes[0, 0].set_title('Сравнение метрик моделей', fontweight='bold')

# 8.3.2 ROC-кривые
axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Случайная модель')
for name, result in results.items():
    if result['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

axes[0, 1].set_xlim([0.0, 1.0])
axes[0, 1].set_ylim([0.0, 1.05])
axes[0, 1].set_xlabel('False Positive Rate', fontweight='bold')
axes[0, 1].set_ylabel('True Positive Rate', fontweight='bold')
axes[0, 1].set_title('ROC-кривые моделей', fontweight='bold')
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(True, alpha=0.3)

# 8.3.3 Precision-Recall кривые
for name, result in results.items():
    if result['y_pred_proba'] is not None:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, result['y_pred_proba'])
        axes[0, 2].plot(recall_vals, precision_vals, label=name, linewidth=2)

axes[0, 2].set_xlabel('Recall', fontweight='bold')
axes[0, 2].set_ylabel('Precision', fontweight='bold')
axes[0, 2].set_title('Precision-Recall кривые', fontweight='bold')
axes[0, 2].legend(loc="upper right")
axes[0, 2].grid(True, alpha=0.3)

# 8.3.4 Матрицы ошибок для лучшей модели по F1-Score
best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
best_result = results[best_model_name]

cm = confusion_matrix(y_test, best_result['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
axes[1, 0].set_xlabel('Предсказанный класс', fontweight='bold')
axes[1, 0].set_ylabel('Истинный класс', fontweight='bold')
axes[1, 0].set_title(f'Матрица ошибок: {best_model_name}\nF1-Score: {best_result["f1"]:.3f}', fontweight='bold')

# 8.3.5 Важность признаков для tree-based моделей
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_result['model'].feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    axes[1, 1].barh(range(len(feature_importance)), 
                    feature_importance['importance'].values,
                    color='skyblue')
    axes[1, 1].set_yticks(range(len(feature_importance)))
    axes[1, 1].set_yticklabels(feature_importance['feature'])
    axes[1, 1].set_xlabel('Важность', fontweight='bold')
    axes[1, 1].set_title(f'Топ-15 важных признаков: {best_model_name}', fontweight='bold')
    axes[1, 1].invert_yaxis()
else:
    # Для логистической регрессии - коэффициенты
    if best_model_name == 'Logistic Regression':
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': best_result['model'].coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False).head(15)
        
        colors = ['red' if x < 0 else 'green' for x in coef_df['coefficient']]
        axes[1, 1].barh(range(len(coef_df)), coef_df['coefficient'].values, color=colors)
        axes[1, 1].set_yticks(range(len(coef_df)))
        axes[1, 1].set_yticklabels(coef_df['feature'])
        axes[1, 1].set_xlabel('Коэффициент', fontweight='bold')
        axes[1, 1].set_title(f'Топ-15 коэффициентов: {best_model_name}', fontweight='bold')
        axes[1, 1].invert_yaxis()
        axes[1, 1].axvline(x=0, color='black', linewidth=0.5)
    else:
        axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Важность признаков', fontweight='bold')
        axes[1, 1].axis('off')

# 8.3.6 Сводная таблица лучшей модели
summary_text = f"""
Лучшая модель: {best_model_name}

Метрики:
- ROC-AUC: {best_result['roc_auc']:.3f}
- Accuracy: {best_result['accuracy']:.3f}
- Precision: {best_result['precision']:.3f}
- Recall: {best_result['recall']:.3f}
- F1-Score: {best_result['f1']:.3f}

Кросс-валидация (ROC-AUC):
Среднее: {best_result['cv_score']:.3f}
"""
axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 2].axis('off')
axes[1, 2].set_title('Сводка по лучшей модели', fontweight='bold')

plt.suptitle('Результаты моделирования Attrition', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig('model_results_summary.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. СВОДНЫЙ АНАЛИЗ И ВЫВОДЫ
print("\n" + "=" * 80)
print("9. СВОДНЫЙ АНАЛИЗ И ВЫВОДЫ")
print("=" * 80)

# Пересчитываем ключевые статистики ПОСЛЕ всех изменений данных
print("\n9.1 КЛЮЧЕВЫЕ ВЫВОДЫ И РЕКОМЕНДАЦИИ\n")

# Пересчитываем статистики Attrition
print("1. БАЗОВАЯ СТАТИСТИКА:")
if 'Attrition' in df.columns:
    # Текущая статистика после всех обработок
    current_attrition_counts = df['Attrition'].value_counts()
    current_attrition_percentage = df['Attrition'].value_counts(normalize=True) * 100
    
    # Проверяем формат Attrition (числовой или строковый)
    attrition_values = df['Attrition'].unique()
    
    # Определяем метки для Yes/No
    if df['Attrition'].dtype == 'object':
        # Строковый формат
        yes_label = 'Yes' if 'Yes' in df['Attrition'].values else ('YES' if 'YES' in df['Attrition'].values else 'Yes')
        no_label = 'No' if 'No' in df['Attrition'].values else ('NO' if 'NO' in df['Attrition'].values else 'No')
    else:
        # Числовой формат
        yes_label = 1
        no_label = 0
    
    # Получаем значения
    if yes_label in current_attrition_counts.index:
        attrition_yes = current_attrition_counts[yes_label]
        attrition_yes_pct = current_attrition_percentage[yes_label]
    else:
        # Если нет 'Yes', ищем альтернативные обозначения
        attrition_yes = 0
        attrition_yes_pct = 0
        for val in current_attrition_counts.index:
            if str(val).lower() in ['yes', 'y', '1', 1]:
                attrition_yes = current_attrition_counts[val]
                attrition_yes_pct = current_attrition_percentage[val]
                break
    
    print(f"   • Общая текучесть: {attrition_yes_pct:.1f}% ({attrition_yes} сотрудников)")
else:
    print("   • Общая текучесть: Данные недоступны (столбец Attrition не найден)")
    attrition_yes = 0
    attrition_yes_pct = 0

print(f"   • Размер выборки: {len(df)} сотрудников")
print(f"   • Количество признаков: {df.shape[1]}")

print("\n2. КЛЮЧЕВЫЕ ФАКТОРЫ, ВЛИЯЮЩИЕ НА ATTRITION:")

# Факторы из корреляционного анализа
if 'attrition_corr' in locals() and attrition_corr is not None and len(attrition_corr) > 0:
    print("   А) Числовые переменные (наибольшая корреляция):")
    
    # Исключаем сам Attrition_numeric если он есть
    if 'Attrition_numeric' in attrition_corr.index:
        relevant_corr = attrition_corr.drop('Attrition_numeric')
    else:
        relevant_corr = attrition_corr
    
    # Сортируем по абсолютному значению
    top_corr_abs = relevant_corr.abs().sort_values(ascending=False).head(5)
    
    for i, (var, corr_abs) in enumerate(top_corr_abs.items(), 1):
        # Получаем реальное значение корреляции
        corr_value = relevant_corr[var]
        direction = "увеличивает" if corr_value > 0 else "уменьшает"
        print(f"      {i}. {var}: {corr_value:.3f} ({direction} вероятность увольнения)")
else:
    print("   А) Числовые переменные: данные корреляционного анализа недоступны")

# Бинарные переменные
if 'odds_df' in locals() and not odds_df.empty and len(odds_df) > 0:
    print("\n   Б) Бинарные переменные (наибольшее влияние):")
    top_binary = odds_df.head(3)
    
    for i, row in top_binary.iterrows():
        if 'odds_ratio' in row:
            if row['odds_ratio'] > 1:
                effect = "повышает"
                effect_text = f"в {row['odds_ratio']:.1f} раз"
            else:
                effect = "снижает"
                effect_text = f"в {1/row['odds_ratio']:.1f} раз"
            
            # Укорачиваем название переменной если оно слишком длинное
            var_name = row['variable']
            if len(var_name) > 30:
                var_name = var_name[:27] + "..."
            
            print(f"      {i+1}. {var_name}: OR={row['odds_ratio']:.2f} ({effect} вероятность {effect_text})")
else:
    print("\n   Б) Бинарные переменные: данные анализа odds ratios недоступны")

# Кластерный анализ
print("\n3. КЛАСТЕРНЫЙ АНАЛИЗ:")
if 'Cluster' in df.columns:
    print(f"   • Оптимальное число кластеров: {optimal_k}")
    print("   • Распределение текучести по кластерам:")
    
    cluster_analysis = []
    for cluster in sorted(df['Cluster'].unique()):
        cluster_size = (df['Cluster'] == cluster).sum()
        
        # Рассчитываем текучесть для кластера
        cluster_data = df[df['Cluster'] == cluster]
        if 'Attrition' in cluster_data.columns:
            # Определяем формат Attrition
            if cluster_data['Attrition'].dtype == 'object':
                attrition_count = (cluster_data['Attrition'].str.contains('Yes', case=False, na=False) | 
                                  (cluster_data['Attrition'] == 'YES') | 
                                  (cluster_data['Attrition'] == 'yes')).sum()
            else:
                # Числовой формат
                attrition_count = (cluster_data['Attrition'] == 1).sum()
            
            attrition_rate = (attrition_count / cluster_size * 100) if cluster_size > 0 else 0
            cluster_analysis.append((cluster, cluster_size, attrition_rate))
    
    # Сортируем по текучести (от высокой к низкой)
    cluster_analysis.sort(key=lambda x: x[2], reverse=True)
    
    for cluster, size, rate in cluster_analysis:
        print(f"      Кластер {cluster}: {size} чел. ({rate:.1f}% текучести)")
else:
    print("   • Кластерный анализ: не выполнен")

print("\n4. РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ:")
if 'best_model_name' in locals() and 'best_result' in locals():
    print(f"   • Лучшая модель: {best_model_name}")
    print(f"   • F1-Score: {best_result['f1']:.3f}")
    
    if best_result['roc_auc'] is not None:
        print(f"   • ROC-AUC: {best_result['roc_auc']:.3f}")
    else:
        print(f"   • ROC-AUC: не рассчитано")
    
    print(f"   • Accuracy: {best_result['accuracy']:.3f}")
    print(f"   • Precision: {best_result['precision']:.3f}")
    print(f"   • Recall: {best_result['recall']:.3f}")
else:
    print("   • Результаты моделирования: недоступны")

# 9.2 Практические рекомендации
print("\n9.2 ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ ДЛЯ HR:\n")

print("1. ПРИОРИТЕТНЫЕ ГРУППЫ РИСКА:")
print("   • Сотрудники с низкой удовлетворенностью работой и окружением")
print("   • Сотрудники, работающие сверхурочно (OverTime)")
print("   • Молодые сотрудники с небольшим стажем в компании")
print(f"   • Кластер с наибольшей текучестью (если анализ выполнен)")

print("\n2. МЕРЫ ПО УДЕРЖАНИЮ:")
print("   • Улучшение Work-Life Balance для сотрудников с высоким OverTime")
print("   • Программы наставничества для новых сотрудников")
print("   • Регулярный мониторинг удовлетворенности")
print("   • Карьерное развитие для снижения YearsSinceLastPromotion")
print("   • Анализ и коррекция факторов из топ-корреляций")

print("\n3. МОНИТОРИНГ:")
print("   • Регулярный анализ ключевых метрик из данного отчета")
print("   • Фокус на сотрудниках из 'рисковых' кластеров")
print("   • A/B тестирование мер по удержанию")
print(f"   • Мониторинг доли текучести (цель: < {attrition_yes_pct:.1f}%)")

# 9.3 Создание финального дашборда
print("\n" + "=" * 80)
print("СОЗДАНИЕ ФИНАЛЬНОГО ДАШБОРДА...")
print("=" * 80)

# Создаем финальный дашборд
fig = plt.figure(figsize=(20, 15))

# 1. Общая текучесть (левый верхний)
ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=2)
attrition_counts.plot(kind='pie', ax=ax1, autopct='%1.1f%%', 
                      colors=['#4CAF50', '#F44336'], startangle=90,
                      explode=(0, 0.1), textprops={'fontsize': 12})
ax1.set_ylabel('')
ax1.set_title('Общая текучесть кадров', fontsize=14, fontweight='bold')

# 2. Топ корреляций (правый верхний)
ax2 = plt.subplot2grid((4, 4), (0, 2), colspan=2)
top_corr_plot = attrition_corr.drop('Attrition_numeric').head(8)
colors_corr = ['red' if x < 0 else 'green' for x in top_corr_plot.values]
bars = ax2.barh(range(len(top_corr_plot)), top_corr_plot.values, color=colors_corr, alpha=0.7)
ax2.set_yticks(range(len(top_corr_plot)))
ax2.set_yticklabels(top_corr_plot.index, fontsize=10)
ax2.set_xlabel('Корреляция с Attrition', fontweight='bold')
ax2.set_title('Топ факторов влияния', fontsize=14, fontweight='bold')
ax2.axvline(x=0, color='black', linewidth=0.5)

# 3. Кластерный анализ (левый средний)
ax3 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
cluster_dist = df['Cluster'].value_counts().sort_index()
cluster_attrition_rates = []
for cluster in cluster_dist.index:
    rate = (df[df['Cluster'] == cluster]['Attrition'] == 'Yes').mean() * 100
    cluster_attrition_rates.append(rate)

x = range(len(cluster_dist))
bars_cluster = ax3.bar(x, cluster_dist.values, alpha=0.7, color='skyblue')
ax3.set_xlabel('Кластер', fontweight='bold')
ax3.set_ylabel('Количество сотрудников', fontweight='bold', color='skyblue')
ax3.set_xticks(x)
ax3.set_xticklabels([f'Кластер {i}' for i in cluster_dist.index])

# Вторая ось для процента текучести
ax3_2 = ax3.twinx()
ax3_2.plot(x, cluster_attrition_rates, 'ro-', linewidth=2, markersize=8)
ax3_2.set_ylabel('Процент текучести (%)', fontweight='bold', color='red')
ax3_2.tick_params(axis='y', labelcolor='red')
ax3.set_title('Кластерный анализ', fontsize=14, fontweight='bold')

# 4. Моделирование (правый средний)
ax4 = plt.subplot2grid((4, 4), (1, 2), colspan=2)
model_names = list(results.keys())
f1_scores = [results[m]['f1'] for m in model_names]
roc_auc_scores = [results[m]['roc_auc'] if results[m]['roc_auc'] is not None else 0 for m in model_names]

x_model = np.arange(len(model_names))
width = 0.35

bars_f1 = ax4.bar(x_model - width/2, f1_scores, width, label='F1-Score', alpha=0.7, color='orange')
bars_auc = ax4.bar(x_model + width/2, roc_auc_scores, width, label='ROC-AUC', alpha=0.7, color='green')

ax4.set_xlabel('Модель', fontweight='bold')
ax4.set_ylabel('Score', fontweight='bold')
ax4.set_xticks(x_model)
ax4.set_xticklabels(model_names, rotation=45, ha='right')
ax4.legend()
ax4.set_title('Сравнение моделей', fontsize=14, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Важные признаки (нижний левый)
ax5 = plt.subplot2grid((4, 4), (2, 0), colspan=2, rowspan=2)
if 'feature_importance' in locals():
    feature_importance = feature_importance.head(10)
    ax5.barh(range(len(feature_importance)), feature_importance['importance'].values,
            color=plt.cm.viridis(np.linspace(0, 1, len(feature_importance))))
    ax5.set_yticks(range(len(feature_importance)))
    ax5.set_yticklabels(feature_importance['feature'], fontsize=10)
else:
    # Если feature_importance не создана, используем коэффициенты из логистической регрессии
    if best_model_name == 'Logistic Regression':
        coef_abs = pd.DataFrame({
            'feature': X.columns,
            'abs_coef': np.abs(best_result['model'].coef_[0])
        }).sort_values('abs_coef', ascending=False).head(10)
        
        ax5.barh(range(len(coef_abs)), coef_abs['abs_coef'].values,
                color=plt.cm.viridis(np.linspace(0, 1, len(coef_abs))))
        ax5.set_yticks(range(len(coef_abs)))
        ax5.set_yticklabels(coef_abs['feature'], fontsize=10)
    else:
        # Альтернатива: используем корреляции
        top_corr_abs = attrition_corr.drop('Attrition_numeric').abs().sort_values(ascending=False).head(10)
        ax5.barh(range(len(top_corr_abs)), top_corr_abs.values,
                color=plt.cm.viridis(np.linspace(0, 1, len(top_corr_abs))))
        ax5.set_yticks(range(len(top_corr_abs)))
        ax5.set_yticklabels(top_corr_abs.index, fontsize=10)

ax5.set_xlabel('Важность/Влияние', fontweight='bold')
ax5.set_title('Топ-10 важных признаков', fontsize=14, fontweight='bold')
ax5.invert_yaxis()

# 6. Демографический анализ (нижний правый)
ax6 = plt.subplot2grid((4, 4), (2, 2), colspan=2, rowspan=2)

# Анализ по возрасту
age_groups = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65])
age_attrition = df.groupby(age_groups)['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)

# Анализ по стажу
tenure_groups = pd.cut(df['YearsAtCompany'], bins=[0, 2, 5, 10, 20, 50])
tenure_attrition = df.groupby(tenure_groups)['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)

x_age = range(len(age_attrition))
x_tenure = [i + 0.4 for i in x_age]  # Смещение для второго графика

bars_age = ax6.bar(x_age, age_attrition.values, width=0.4, label='По возрасту', alpha=0.7, color='blue')
bars_tenure = ax6.bar(x_tenure, tenure_attrition.values, width=0.4, label='По стажу в компании', alpha=0.7, color='red')

ax6.set_xlabel('Группы', fontweight='bold')
ax6.set_ylabel('Процент текучести (%)', fontweight='bold')
ax6.set_xticks([i + 0.2 for i in x_age])
ax6.set_xticklabels([str(age_attrition.index[i]) for i in range(len(age_attrition))], 
                   rotation=45, ha='right', fontsize=9)
ax6.legend()
ax6.set_title('Текучесть по демографическим группам', fontsize=14, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('ДАШБОРД: АНАЛИЗ ТЕКУЧЕСТИ КАДРОВ (ATTRITION ANALYSIS)', 
             fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('final_attrition_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
print("\n" + "=" * 80)
print("10. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ АНАЛИЗА")
print("=" * 80)

# Функция для расчета текучести
def get_attrition_stats(df):
    if 'Attrition' not in df.columns:
        return 0, 0, 0
    
    # Приводим к строковому формату
    attrition_series = df['Attrition'].astype(str).str.strip().str.title()
    
    # Подсчет Yes/No
    yes_count = attrition_series[attrition_series.isin(['Yes', 'YES', 'yes', '1'])].count()
    no_count = attrition_series[attrition_series.isin(['No', 'NO', 'no', '0'])].count()
    
    total = yes_count + no_count
    
    if total > 0:
        yes_pct = (yes_count / total) * 100
    else:
        yes_pct = 0
    
    return yes_count, yes_pct, total

# Получаем актуальные статистики
current_yes_count, current_yes_pct, current_total = get_attrition_stats(df)

# Сохранение обработанных данных
df.to_csv('attrition_analysis_processed.csv', index=False)
print("✓ Обработанные данные сохранены в: attrition_analysis_processed.csv")

# Сохранение ключевых метрик
summary_data = {
    'Метрика': [
        'Общая текучесть (%)',
        'Количество уволившихся',
        'Общее количество сотрудников',
        'Размер выборки (строки)',
        'Количество признаков (столбцы)',
        'Оптимальное число кластеров',
        'Лучшая модель',
        'F1-Score лучшей модели',
        'ROC-AUC лучшей модели',
        'Accuracy лучшей модели',
        'Precision лучшей модели',
        'Recall лучшей модели'
    ],
    'Значение': [
        current_yes_pct,
        int(current_yes_count),
        int(current_total),
        len(df),
        df.shape[1],
        optimal_k if 'optimal_k' in locals() else 'N/A',
        best_model_name if 'best_model_name' in locals() else 'N/A',
        best_result['f1'] if 'best_result' in locals() else 'N/A',
        best_result['roc_auc'] if 'best_result' in locals() and best_result['roc_auc'] is not None else 'N/A',
        best_result['accuracy'] if 'best_result' in locals() else 'N/A',
        best_result['precision'] if 'best_result' in locals() else 'N/A',
        best_result['recall'] if 'best_result' in locals() else 'N/A'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('attrition_analysis_summary.csv', index=False)
print("✓ Сводные метрики сохранены в: attrition_analysis_summary.csv")

# Сохранение корреляций (если они есть)
if 'attrition_corr' in locals() and attrition_corr is not None:
    try:
        attrition_corr_df = pd.DataFrame(attrition_corr).reset_index()
        attrition_corr_df.columns = ['Признак', 'Корреляция_с_Attrition']
        attrition_corr_df.to_csv('attrition_correlations.csv', index=False)
        print("✓ Корреляции сохранены в: attrition_correlations.csv")
    except Exception as e:
        print(f"✗ Ошибка при сохранении корреляций: {e}")
else:
    print("✗ Корреляции не найдены для сохранения")

# Сохранение отношений шансов (если они есть)
if 'odds_df' in locals() and not odds_df.empty and len(odds_df) > 0:
    try:
        odds_df.to_csv('attrition_odds_ratios.csv', index=False)
        print("✓ Отношения шансов сохранены в: attrition_odds_ratios.csv")
    except Exception as e:
        print(f"✗ Ошибка при сохранении отношений шансов: {e}")
else:
    print("✗ Отношения шансов не найдены для сохранения")

# Сохранение профилей кластеров (если кластеризация выполнена)
if 'Cluster' in df.columns and 'cluster_profiles' in locals():
    try:
        cluster_profiles.to_csv('attrition_cluster_profiles.csv')
        print("✓ Профили кластеров сохранены в: attrition_cluster_profiles.csv")
    except Exception as e:
        print(f"✗ Ошибка при сохранении профилей кластеров: {e}")
else:
    print("✗ Профили кластеров не найдены для сохранения")

# Дополнительно: сохранение важных признаков из лучшей модели
if 'best_model_name' in locals() and 'best_result' in locals():
    try:
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_result['model'].feature_importances_
            }).sort_values('importance', ascending=False)
            feature_importance.to_csv('attrition_feature_importance.csv', index=False)
            print("✓ Важность признаков сохранена в: attrition_feature_importance.csv")
        elif best_model_name == 'Logistic Regression':
            coef_df = pd.DataFrame({
                'feature': X.columns,
                'coefficient': best_result['model'].coef_[0],
                'abs_coefficient': np.abs(best_result['model'].coef_[0])
            }).sort_values('abs_coefficient', ascending=False)
            coef_df.to_csv('attrition_logistic_coefficients.csv', index=False)
            print("✓ Коэффициенты логистической регрессии сохранены в: attrition_logistic_coefficients.csv")
    except Exception as e:
        print(f"✗ Ошибка при сохранении важности признаков: {e}")

print("\n" + "=" * 80)
print("АНАЛИЗ УСПЕШНО ЗАВЕРШЕН!")
print("=" * 80)

print("\nСОЗДАННЫЕ ФАЙЛЫ:")
print("=" * 40)

import os
created_files = []

# Проверяем созданные файлы
file_checks = [
    ('attrition_analysis_processed.csv', 'Обработанные данные'),
    ('attrition_analysis_summary.csv', 'Сводные метрики'),
    ('attrition_correlations.csv', 'Корреляции с Attrition'),
    ('attrition_odds_ratios.csv', 'Отношения шансов'),
    ('attrition_cluster_profiles.csv', 'Профили кластеров'),
    ('attrition_feature_importance.csv', 'Важность признаков'),
    ('attrition_logistic_coefficients.csv', 'Коэффициенты регрессии'),
    # Графики
    ('class_balance_attrition.png', 'График: Баланс классов'),
    ('continuous_vars_distribution.png', 'График: Распределение числовых переменных'),
    ('ordinal_vars_distribution.png', 'График: Распределение порядковых переменных'),
    ('attrition_comparison_ttest.png', 'График: Сравнение по Attrition'),
    ('full_correlation_heatmap.png', 'График: Корреляционная матрица'),
    ('top_correlations_with_attrition.png', 'График: Топ корреляций'),
    ('odds_ratios_binary_vars.png', 'График: Odds Ratios'),
    ('optimal_clusters_determination.png', 'График: Определение кластеров'),
    ('clustering_results.png', 'График: Результаты кластеризации'),
    ('cluster_profiles_heatmap.png', 'График: Профили кластеров'),
    ('pca_variance_analysis.png', 'График: PCA анализ'),
    ('pca_loadings_analysis.png', 'График: Нагрузки PCA'),
    ('pca_biplot.png', 'График: Biplot PCA'),
    ('model_results_summary.png', 'График: Результаты моделей'),
    ('final_attrition_dashboard.png', 'Дашборд: Итоговый дашборд'),
]

for filename, description in file_checks:
    if os.path.exists(filename):
        file_size = os.path.getsize(filename) / 1024  # Размер в КБ
        created_files.append((filename, description, file_size))
        print(f"✓ {description:40} [{filename}] ({file_size:.1f} KB)")
    else:
        print(f"✗ {description:40} [{filename}] (не создан)")

print("\n" + "=" * 40)
print(f"ИТОГО: {len(created_files)} файлов создано успешно")

if created_files:
    total_size = sum(f[2] for f in created_files)
    print(f"Общий размер: {total_size:.1f} KB")

print("\nКЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
print("=" * 40)

# Выводим ключевые результаты
if current_total > 0:
    print(f"• Текучесть кадров: {current_yes_pct:.1f}% ({current_yes_count}/{current_total})")
else:
    print("• Текучесть кадров: данные недоступны")

if 'optimal_k' in locals():
    print(f"• Оптимальное число кластеров: {optimal_k}")

if 'best_model_name' in locals() and 'best_result' in locals():
    print(f"• Лучшая модель предсказания: {best_model_name}")
    print(f"• Качество модели (F1-Score): {best_result['f1']:.3f}")

print("\nРЕКОМЕНДАЦИИ:")
print("=" * 40)
print("1. Проанализируйте файл attrition_analysis_summary.csv для основных метрик")
print("2. Изучите attrition_correlations.csv для понимания факторов текучести")
print("3. Используйте attrition_cluster_profiles.csv для сегментации сотрудников")
print("4. Все графики сохранены в PNG файлах для презентаций и отчетов")
print("5. Обработанные данные в attrition_analysis_processed.csv можно использовать для дальнейшего анализа")

print("\n" + "=" * 80)
print("Анализ завершен! Результаты готовы для использования.")
print("=" * 80)