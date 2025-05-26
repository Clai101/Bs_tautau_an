from pathlib import Path
import os
import uproot
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 📁 Папка, в которую будут сохраняться feather-файлы
OUTPUT_DIR = Path(".")  # можно заменить, например, на Path("feathers/")

# 🛠️ Основная функция обработки одного файла
def process_file(filename):
    try:
        with uproot.open(filename) as file:
            if "Y5S" not in file:
                raise ValueError(f"'Y5S' not found in {filename}")

            tree = file["Y5S"]
            print(f"📖 Читаем {filename} | переменные: {tree.keys()}")

            # Чтение всего дерева в pandas.DataFrame
            df = tree.arrays(library="pd")

            # Получаем имя файла без пути и расширения
            base_name = os.path.splitext(os.path.basename(filename))[0]
            feather_name = f"{base_name}.feather"

            # Полный путь к выходному файлу
            out_path = OUTPUT_DIR / feather_name

            # Сохраняем в feather
            df.to_feather(out_path)

            print(f"Сохранено: {out_path}")
            return out_path

    except Exception as e:
        print(f"Ошибка при обработке {filename}: {e}")
        return None

# 🚀 Параллельная обработка всех .root файлов с "_cut"
def main():
    from pathlib import Path

    input_dir = Path(".")  # или Path("input_root/")

    root_files = [f for f in input_dir.glob("*.root")]
    print(f"🔎 Найдено файлов: {len(root_files)}")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, str(f)): f for f in root_files}

        for future in as_completed(futures):
            result = future.result()
            if result:
                print(f"🎉 Завершено: {result}")
            else:
                print(f"⚠️ Ошибка в потоке для файла: {futures[future]}")

    # Печать итогов
    print("\n📦 Список созданных файлов:")
    for f in sorted(OUTPUT_DIR.glob("*.feather")):
        print(" -", f.name)

if __name__ == "__main__":
    main()
