import pandas as pd
from typing import Optional
import numpy


class DFParams:
    """Хранит серии X и Y и лениво вычисляет их выборочные средние."""

    def __init__(self) -> None:
        self._x_bar: Optional[float] = None
        self._y_bar: Optional[float] = None
        self._Sxx: any = None
        self._Sxy: any = None 


    def x_bar(self, tx_column: Optional[float]) -> float:
        """Среднее значение X (x̄)."""
        if self._x_bar is None:
            if tx_column is None:
                raise ValueError("Данные для X не предоставлены")
            self._x_bar = float(tx_column.mean())
        return self._x_bar


    def y_bar(self, ty_column: Optional[float]) -> float:
        """Среднее значение Y (ȳ)."""
        if self._y_bar is None:
            if ty_column is None:
                raise ValueError("Данные для Y не предоставлены")
            self._y_bar = float(ty_column.mean())
        return self._y_bar

    def Sxx(self, tx_column: Optional[float]) -> float:
        """Cкорректированная сумма квадратов для признака x"""
        if self._Sxx is None:
            if tx_column is None and self.x_bar is None:
                raise ValueError("Данные для Sxx не предоставлены")
            self._Sxx =  numpy.sum((tx_column - self.x_bar) ** 2)
        return self._Sxx

    def Sxy(self, tx_column: Optional[float], ty_column: Optional[float]) -> float:
        """Cкорректированная совместная сумма:"""
        if self._Sxy is None:
            if tx_column is None and self.y_bar is None and self.y_bar is None:
                raise ValueError("Данные для Sxy не предоставлены")
            self._Sxy = numpy.sum((tx_column - self.x_bar) * (ty_column - self.y_bar))
        return self._Sxy
