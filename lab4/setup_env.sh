#!/bin/bash
# setup_env.sh - Скрипт для создания виртуального окружения и установки зависимостей

echo "Создание виртуального окружения..."
python3 -m venv myenv

echo "Активация виртуального окружения..."
source myenv/bin/activate

echo "Обновление pip..."
pip install --upgrade pip

echo "Установка зависимостей..."
pip install -r requirements.txt

echo "Виртуальное окружение создано и настроено!"
echo "Для активации используйте: source myenv/bin/activate"
echo "Для запуска Jupyter: ./start_jupyter.sh"



