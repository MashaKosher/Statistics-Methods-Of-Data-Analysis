import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy


def _setup_plot(df: DataFrame, title: str):
    """
    Внутренняя функция для базовой настройки графика.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(df["tx"], df["ty"], color="tab:blue", label="Данные")
    plt.xlim(12, 20)
    plt.ylim(200, 300)
    plt.xlabel("tx")
    plt.ylabel("ty")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

def ScatterPlotOfTyVsTx(df: DataFrame):
    """
    Постройте точечный график данной зависимости. 
    Масштаб по оси Y установите от 200 до 300, 
    по оси X – от 12 до 20.
    """
    _setup_plot(df, "Точечный график зависимости ty от tx")
    plt.show()

def EstimateLinearRegressionParams(df: DataFrame):
    """
    По полученному графику дайте предварительную оценку 
    следующим характеристикам модели: a0 и a1
    """
    a1_hat, a0_hat = numpy.polyfit(df["tx"], df["ty"], 1)
    print(f"a0^ = {a0_hat:.2f}, a1^ = {a1_hat:.2f}")

    # Линия регрессии поверх точек
    x_line = numpy.linspace(12, 20, 100)
    y_line = a1_hat * x_line + a0_hat

    _setup_plot(df, "Линейная регрессия: ty ~ tx")
    plt.plot(x_line, y_line, color="crimson", label="Линия регрессии")
    plt.legend()  # Обновляем легенду с новой линией
    plt.show()