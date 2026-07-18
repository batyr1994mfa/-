#!/bin/bash
set -e
cd "$(dirname "$0")"
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-web.txt
echo
echo "Готово! Запуск: ./run-web.sh"
echo
echo "Подсказка: для инструмента ocr_image (распознавание текста на изображениях)"
echo "дополнительно нужен системный пакет tesseract-ocr, например:"
echo "  sudo apt install tesseract-ocr tesseract-ocr-rus"
