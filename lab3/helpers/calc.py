import pandas as pd
from typing import Optional


class DFParams:
    """Хранит серии X и Y и лениво вычисляет их выборочные средние."""

    def __init__(self) -> None:
        self._x_bar: Optional[float] = None
        self._y_bar: Optional[float] = None


    def x_bar(self, tx_column: Optional[float]) -> float:
        """Среднее значение X (x̄)."""
        if self._x_bar is None:
            if tx_column is None:
                raise ValueError("Данные для X не предоставлены")
            self._x_bar = float(tx_column.mean())
        return self._x_bar

    @property
    def y_bar(self, ty_column: Optional[float]) -> float:
        """Среднее значение Y (ȳ)."""
        if self._y_bar is None:
            if self._y is None:
                raise ValueError("Данные для Y не предоставлены")
            self._y_bar = float(ty_column.mean())
        return self._y_bar
