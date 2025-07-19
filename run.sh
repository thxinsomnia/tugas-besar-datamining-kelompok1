#!/bin/bash

echo " Menjalankan pipeline Data Mining..."

# Optional: Cek dan install requirements
# if [ -f requirements.txt ]; then
#     echo " Menginstall dependensi dari requirements.txt..."
#     pip install -r requirements.txt
# fi
cd src
# Jalankan pipeline utama
echo " Menjalankan src/main.py..."
python main.py