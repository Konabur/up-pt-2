# TLS Point Cloud Volume Estimation

Методы оценки объёма растительного покрова по данным наземного лазерного сканирования: обзор и апробация на синтетических данных.

## Стек

- Python 3.11+
- NumPy, SciPy — вычисления
- Open3D — обработка облаков точек, фильтрация
- matplotlib, Plotly — визуализация
- alphashape — alpha-shape метод
- python-dotenv — переменные окружения

## Синтез данных

Генерируется поле пшеницы (5×8 растений):
- **Морфология**: стебель (цилиндр с изгибом), колос (эллипсоид), листья (изогнутые полоски)
- **Земля**: плоскость с микрорельефом
- **Окклюзия TLS**: удаление точек, загороженных ближними объектами
- **Шум**: гауссов шум координат, случайные выбросы (3%), фантомные отражения (1.5%)

Ground truth объём вычисляется аналитически по геометрии растений.

## Установка

### С uv (рекомендуется)

```bash
uv sync
```

### С venv + pip

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Для работы с LAS/LAZ:
```bash
pip install laspy
```

## Использование

### Базовое использование

```bash
# Генерация синтетического облака и анализ
python main.py

# Сохранить облако для повторного использования
python main.py --save-cloud cloud.npz

# Анализ готового облака
python main.py --cloud cloud.npz
```

### Анализ реальных сканов

```bash
# Базовый анализ (автоопределение единиц)
python main.py --cloud scan.las

# С указанием ground truth объёма
python main.py --cloud scan.pcd --gt-volume 0.0096

# С явным указанием единиц измерения
python main.py --cloud scan.las --units mm --gt-volume 0.0096

# Быстрый анализ (пропуск Convex Hull и Alpha Shape)
python main.py --cloud scan.pcd --skip-hull-methods
```

### Переменные окружения

Создайте `.env` файл (см. `.env.example`):

```bash
# Префикс: TPCVE (TLS Point Cloud Volume Estimation)
TPCVE_CLOUD=path/to/cloud.pcd
TPCVE_GT_VOLUME=0.0096
TPCVE_UNITS=mm
TPCVE_OUTPUT_DIR=results
TPCVE_DEFAULT_VOXEL_SIZE=0.007
TPCVE_SKIP_HULL_METHODS=false
```

Аргументы командной строки имеют приоритет над переменными окружения.

## Поддерживаемые форматы

- `.npz` — синтетические данные (с ground truth)
- `.las`, `.laz` — стандарт индустрии
- `.pcd`, `.ply`, `.xyz`, `.pts` — Open3D форматы

Для реальных облаков:
- Автоопределение единиц измерения (м/см/мм)
- Опциональное указание ground truth объёма через `--gt-volume`

## Результаты

Сохраняются в `results/`:
- `01_pipeline.png` — этапы обработки
- `02_volume_comparison.png` — сравнение методов
- `03_voxel_analysis.png` — анализ размера вокселя
- `04_hull_vs_voxel.png` — сравнение с convex hull
- `3d_raw.html`, `3d_classified.html` — 3D визуализация
- `results.json` — метрики
