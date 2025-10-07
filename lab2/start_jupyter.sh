#!/bin/bash
# start_jupyter.sh - Скрипт для запуска Jupyter Notebook

echo "Активация виртуального окружения..."
source myenv/bin/activate

echo "Запуск Jupyter Notebook..."
jupyter notebook
