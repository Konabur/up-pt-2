"""
Эксперимент: оценка объёма растительного покрова на синтетических данных
наземного лазерного сканирования (TLS).

Версия 2:
- Реалистичная модель шума (больше выбросов, мультипереотражения)
- Моделирование окклюзии (характерная проблема TLS)
- 3D интерактивная визуализация (Plotly)
- Улучшенные графики сравнения методов
"""

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull
import json
import os

np.random.seed(42)

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ОБЛАКА ТОЧЕК
# ============================================================

def generate_ground(x_range=(-2, 2), y_range=(-2, 2), density=5000, noise_std=0.008):
    """Генерация плоскости земли с микрорельефом."""
    n = density
    x = np.random.uniform(x_range[0], x_range[1], n)
    y = np.random.uniform(y_range[0], y_range[1], n)
    # микрорельеф: плавные волны + случайный шум
    z = 0.015 * np.sin(2 * x) * np.cos(3 * y) + np.random.normal(0, noise_std, n)
    return np.column_stack([x, y, z])


def generate_wheat_plant(cx, cy, height=0.6, stem_radius=0.008,
                         ear_length=0.08, ear_radius=0.012,
                         points_stem=100, points_ear=70, points_leaves=60):
    """
    Генерация одного растения пшеницы:
    - стебель: тонкий цилиндр с лёгким изгибом
    - колос: эллипсоид на вершине
    - листья: 2-4 изогнутые полоски
    """
    pts = []

    # Стебель (цилиндр с лёгким изгибом)
    t_stem = np.random.uniform(0, 1, points_stem)
    z_stem = t_stem * height
    # лёгкий изгиб стебля
    bend_dir = np.random.uniform(0, 2 * np.pi)
    bend_amount = np.random.uniform(0.005, 0.02)
    theta = np.random.uniform(0, 2 * np.pi, points_stem)
    x_stem = cx + stem_radius * np.cos(theta) + bend_amount * t_stem ** 2 * np.cos(bend_dir)
    y_stem = cy + stem_radius * np.sin(theta) + bend_amount * t_stem ** 2 * np.sin(bend_dir)
    pts.append(np.column_stack([x_stem, y_stem, z_stem]))

    # Колос (эллипсоид)
    top_x = cx + bend_amount * np.cos(bend_dir)
    top_y = cy + bend_amount * np.sin(bend_dir)
    theta_ear = np.random.uniform(0, 2 * np.pi, points_ear)
    phi_ear = np.random.uniform(0, np.pi, points_ear)
    x_ear = top_x + ear_radius * np.sin(phi_ear) * np.cos(theta_ear)
    y_ear = top_y + ear_radius * np.sin(phi_ear) * np.sin(theta_ear)
    z_ear = height + ear_length * np.cos(phi_ear)
    pts.append(np.column_stack([x_ear, y_ear, z_ear]))

    # Листья (2-4 изогнутых)
    n_leaves = np.random.randint(2, 5)
    for _ in range(n_leaves):
        leaf_angle = np.random.uniform(0, 2 * np.pi)
        leaf_start_z = np.random.uniform(0.1 * height, 0.65 * height)
        leaf_length = np.random.uniform(0.08, 0.18)
        n_lpts = points_leaves
        t = np.sort(np.random.uniform(0, 1, n_lpts))
        # изгиб листа: сначала вверх, потом вниз
        x_leaf = cx + t * leaf_length * np.cos(leaf_angle)
        y_leaf = cy + t * leaf_length * np.sin(leaf_angle)
        z_leaf = leaf_start_z + t * 0.04 - t ** 2 * 0.08
        # ширина листа меняется
        width = 0.005 * (1 - 0.5 * t)
        x_leaf += np.random.normal(0, width, n_lpts)
        y_leaf += np.random.normal(0, width, n_lpts)
        pts.append(np.column_stack([x_leaf, y_leaf, z_leaf]))

    return np.vstack(pts)


def generate_wheat_field(n_rows=5, plants_per_row=8, row_spacing=0.25,
                         plant_spacing=0.12, height_mean=0.55, height_std=0.08):
    """Генерация поля пшеницы: ряды растений."""
    all_plants = []
    plant_params = []

    for row in range(n_rows):
        cy = -((n_rows - 1) * row_spacing / 2) + row * row_spacing
        for col in range(plants_per_row):
            cx = -((plants_per_row - 1) * plant_spacing / 2) + col * plant_spacing
            cx += np.random.normal(0, 0.012)
            cy += np.random.normal(0, 0.012)
            h = max(0.3, np.random.normal(height_mean, height_std))

            plant = generate_wheat_plant(cx, cy, height=h)
            all_plants.append(plant)

            # ground truth объём
            stem_vol = np.pi * 0.008 ** 2 * h
            ear_vol = (4 / 3) * np.pi * 0.012 * 0.012 * 0.08
            leaf_vol = 3 * 0.15 * 0.008 * 0.002
            plant_params.append({
                'cx': cx, 'cy': cy, 'height': h,
                'volume': stem_vol + ear_vol + leaf_vol
            })

    return np.vstack(all_plants), plant_params


def simulate_occlusion(points, scanner_pos=np.array([-2.5, 0, 0.5]),
                       occlusion_strength=0.3):
    """
    Моделирование окклюзии TLS: точки, «загороженные» ближними объектами,
    частично удаляются. Чем дальше точка от сканера и чем больше перед ней
    других точек, тем выше вероятность удаления.
    """
    # Расстояние до сканера
    dists = np.linalg.norm(points - scanner_pos, axis=1)
    max_dist = dists.max()

    # Вероятность удаления растёт с расстоянием
    p_remove = occlusion_strength * (dists / max_dist) ** 1.5

    # Точки выше земли и дальше от сканера — больше шанс окклюзии
    above_ground = points[:, 2] > 0.03
    p_remove[above_ground] *= 1.5
    p_remove = np.clip(p_remove, 0, 0.7)

    keep_mask = np.random.random(len(points)) > p_remove
    return points[keep_mask], keep_mask


def add_realistic_noise(points, noise_std=0.008, outlier_fraction=0.06,
                        phantom_fraction=0.03):
    """
    Реалистичный шум TLS:
    - Гауссов шум на координатах (ошибки дальномера)
    - Случайные выбросы (6%)
    - Фантомные отражения (мультипереотражения) — точки между объектами
    """
    n = len(points)

    # 1. Гауссов шум
    points = points + np.random.normal(0, noise_std, points.shape)

    # 2. Случайные выбросы (далеко от объектов)
    n_outliers = int(n * outlier_fraction)
    outliers = np.random.uniform(
        [points[:, 0].min() - 1.0, points[:, 1].min() - 1.0, -0.2],
        [points[:, 0].max() + 1.0, points[:, 1].max() + 1.0, 1.2],
        (n_outliers, 3)
    )

    # 3. Фантомные отражения (точки между землёй и растительностью)
    n_phantoms = int(n * phantom_fraction)
    phantom_idx = np.random.choice(n, n_phantoms)
    phantoms = points[phantom_idx].copy()
    # смещаем случайно, имитируя мультипереотражение
    phantoms[:, 2] *= np.random.uniform(0.2, 0.8, n_phantoms)
    phantoms[:, 0] += np.random.normal(0, 0.03, n_phantoms)
    phantoms[:, 1] += np.random.normal(0, 0.03, n_phantoms)

    return np.vstack([points, outliers, phantoms])


# --- Генерация ---
print("=" * 60)
print("ГЕНЕРАЦИЯ СИНТЕТИЧЕСКОГО ОБЛАКА ТОЧЕК")
print("=" * 60)

ground_pts = generate_ground()
vegetation_pts, plant_params = generate_wheat_field()
total_gt_volume = sum(p['volume'] for p in plant_params)

all_pts_clean = np.vstack([ground_pts, vegetation_pts])

# Окклюзия
print("\nМоделирование окклюзии TLS...")
scanner_pos = np.array([-2.5, 0, 0.5])
all_pts_occluded, occ_mask = simulate_occlusion(all_pts_clean, scanner_pos, occlusion_strength=0.35)
n_occluded = len(all_pts_clean) - len(all_pts_occluded)
print(f"  Позиция сканера: {scanner_pos}")
print(f"  Удалено окклюзией: {n_occluded} ({100 * n_occluded / len(all_pts_clean):.1f}%)")

# Шум
print("\nДобавление реалистичного шума...")
all_pts_noisy = add_realistic_noise(all_pts_occluded, noise_std=0.008,
                                     outlier_fraction=0.06, phantom_fraction=0.03)

print(f"\n  Точек земли (исходно): {len(ground_pts)}")
print(f"  Точек растительности (исходно): {len(vegetation_pts)}")
print(f"  Точек после окклюзии: {len(all_pts_occluded)}")
print(f"  Точек итого (с шумом): {len(all_pts_noisy)}")
print(f"  Растений: {len(plant_params)}")
print(f"  Ground truth объём: {total_gt_volume:.6f} м³")


# ============================================================
# 2. ФИЛЬТРАЦИЯ (SOR)
# ============================================================

print("\n" + "=" * 60)
print("ФИЛЬТРАЦИЯ")
print("=" * 60)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_pts_noisy)

# SOR фильтр
pcd_sor, ind_sor = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
pts_after_sor = np.asarray(pcd_sor.points)
n_removed_sor = len(all_pts_noisy) - len(pts_after_sor)
print(f"SOR (k=20, σ=1.5): удалено {n_removed_sor} ({100 * n_removed_sor / len(all_pts_noisy):.1f}%)")

# Radius outlier removal (второй проход)
pcd_sor2 = o3d.geometry.PointCloud()
pcd_sor2.points = o3d.utility.Vector3dVector(pts_after_sor)
pcd_filtered, ind_rad = pcd_sor2.remove_radius_outlier(nb_points=3, radius=0.05)
pts_filtered = np.asarray(pcd_filtered.points)
n_removed_rad = len(pts_after_sor) - len(pts_filtered)
print(f"Radius (r=0.03, min=6): удалено {n_removed_rad} ({100 * n_removed_rad / len(pts_after_sor):.1f}%)")
print(f"Итого после фильтрации: {len(pts_filtered)}")


# ============================================================
# 3. ОТДЕЛЕНИЕ ЗЕМЛИ ОТ РАСТИТЕЛЬНОСТИ
# ============================================================

print("\n" + "=" * 60)
print("КЛАССИФИКАЦИЯ ЗЕМЛЯ / РАСТИТЕЛЬНОСТЬ")
print("=" * 60)

height_threshold = 0.04

ground_mask = pts_filtered[:, 2] < height_threshold
vegetation_mask = ~ground_mask

ground_classified = pts_filtered[ground_mask]
vegetation_classified = pts_filtered[vegetation_mask]

print(f"  Порог высоты: {height_threshold} м")
print(f"  Точек земли: {len(ground_classified)}")
print(f"  Точек растительности: {len(vegetation_classified)}")


# ============================================================
# 4. ОЦЕНКА ОБЪЁМА
# ============================================================

print("\n" + "=" * 60)
print("ОЦЕНКА ОБЪЁМА")
print("=" * 60)


def voxel_volume(points, voxel_size=0.01):
    """Воксельный метод."""
    indices = np.floor(points / voxel_size).astype(int)
    unique_voxels = set(map(tuple, indices))
    return len(unique_voxels) * (voxel_size ** 3), len(unique_voxels)


def convex_hull_volume(points):
    """Convex hull."""
    if len(points) < 4:
        return 0.0
    hull = ConvexHull(points)
    return hull.volume


def alpha_shape_volume(points, alpha_value):
    """Alpha shape."""
    try:
        import alphashape
        if len(points) > 5000:
            idx = np.random.choice(len(points), 5000, replace=False)
            points_sub = points[idx]
        else:
            points_sub = points
        alpha_sh = alphashape.alphashape(points_sub, alpha_value)
        if hasattr(alpha_sh, 'volume'):
            return alpha_sh.volume
        return 0.0
    except Exception as e:
        print(f"    Alpha shape error: {e}")
        return 0.0


# Воксельный метод — разные размеры
voxel_sizes = [0.005, 0.007, 0.01, 0.012, 0.015, 0.02, 0.03, 0.05]
voxel_results = {}
print("\nВоксельный метод:")
for vs in voxel_sizes:
    vol, n_vox = voxel_volume(vegetation_classified, voxel_size=vs)
    err = (vol - total_gt_volume) / total_gt_volume * 100
    voxel_results[vs] = {'volume': vol, 'n_voxels': n_vox, 'error_pct': err}
    print(f"  size={vs:.3f} м: V={vol:.6f} м³, вокселей={n_vox}, ошибка={err:+.1f}%")

# Convex Hull
ch_volume = convex_hull_volume(vegetation_classified)
ch_error = (ch_volume - total_gt_volume) / total_gt_volume * 100
print(f"\nConvex Hull: V={ch_volume:.6f} м³, ошибка={ch_error:+.1f}%")

# Alpha Shape
alpha_values = [1.0, 5.0, 10.0, 20.0, 50.0]
alpha_results = {}
print("\nAlpha Shape:")
for av in alpha_values:
    vol = alpha_shape_volume(vegetation_classified, av)
    if vol > 0:
        err = (vol - total_gt_volume) / total_gt_volume * 100
        alpha_results[av] = {'volume': vol, 'error_pct': err}
        print(f"  α={av}: V={vol:.6f} м³, ошибка={err:+.1f}%")
    else:
        print(f"  α={av}: не удалось вычислить")


# ============================================================
# 5. СВОДНАЯ ТАБЛИЦА
# ============================================================

print("\n" + "=" * 70)
print(f"{'МЕТОД':<30} {'ОБЪЁМ (м³)':<14} {'ОШИБКА':<14} {'ПРИМЕЧАНИЕ'}")
print("=" * 70)
print(f"{'Ground Truth':<30} {total_gt_volume:<14.6f} {'—':<14} {'эталон'}")
print("-" * 70)

all_results = []

for vs in voxel_sizes:
    r = voxel_results[vs]
    label = f"Voxel {vs*1000:.0f}мм"
    note = "✓ лучший" if abs(r['error_pct']) == min(abs(voxel_results[v]['error_pct']) for v in voxel_sizes) else ""
    print(f"{label:<30} {r['volume']:<14.6f} {r['error_pct']:+13.1f}% {note}")
    all_results.append({'method': label, 'volume': r['volume'],
                        'error_pct': r['error_pct'], 'type': 'voxel'})

print("-" * 70)
print(f"{'Convex Hull':<30} {ch_volume:<14.6f} {ch_error:+13.1f}% завышает (оболочка)")
all_results.append({'method': 'Convex Hull', 'volume': ch_volume,
                    'error_pct': ch_error, 'type': 'hull'})

for av in sorted(alpha_results.keys()):
    r = alpha_results[av]
    label = f"Alpha Shape α={av}"
    print(f"{label:<30} {r['volume']:<14.6f} {r['error_pct']:+13.1f}%")
    all_results.append({'method': label, 'volume': r['volume'],
                        'error_pct': r['error_pct'], 'type': 'alpha'})

print("=" * 70)


# ============================================================
# 6. ГРАФИКИ
# ============================================================

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white'
})

# --- График 1: Pipeline обработки (вид сбоку XZ) ---
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1a: Исходное (с шумом)
ax = fig.add_subplot(gs[0, 0])
idx = np.random.choice(len(all_pts_noisy), min(12000, len(all_pts_noisy)), replace=False)
ax.scatter(all_pts_noisy[idx, 0], all_pts_noisy[idx, 2], s=0.2, alpha=0.4, c='gray')
ax.set_title('1. Исходное облако\n(с шумом и выбросами)')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(-0.3, 1.0)

# 1b: После окклюзии (показываем до шума, но после окклюзии)
ax = fig.add_subplot(gs[0, 1])
idx = np.random.choice(len(all_pts_noisy), min(12000, len(all_pts_noisy)), replace=False)
colors_occ = np.where(all_pts_noisy[idx, 0] < -0.3, 'orange', 'steelblue')
ax.scatter(all_pts_noisy[idx, 0], all_pts_noisy[idx, 2], s=0.2, alpha=0.4, c=colors_occ)
ax.axvline(x=-0.3, color='red', linestyle=':', alpha=0.5)
ax.annotate('← сканер', xy=(-2.5, 0.5), fontsize=9, color='red')
ax.set_title('2. Эффект окклюзии\n(задние ряды разрежены)')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(-0.3, 1.0)

# 1c: После фильтрации
ax = fig.add_subplot(gs[0, 2])
idx = np.random.choice(len(pts_filtered), min(12000, len(pts_filtered)), replace=False)
ax.scatter(pts_filtered[idx, 0], pts_filtered[idx, 2], s=0.2, alpha=0.4, c='steelblue')
ax.set_title('3. После фильтрации\n(SOR + Radius)')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(-0.3, 1.0)

# 1d: Классификация
ax = fig.add_subplot(gs[1, 0])
idx_g = np.random.choice(len(ground_classified), min(5000, len(ground_classified)), replace=False)
idx_v = np.random.choice(len(vegetation_classified), min(7000, len(vegetation_classified)), replace=False)
ax.scatter(ground_classified[idx_g, 0], ground_classified[idx_g, 2],
           s=0.3, alpha=0.5, c='#8B4513', label='Земля')
ax.scatter(vegetation_classified[idx_v, 0], vegetation_classified[idx_v, 2],
           s=0.3, alpha=0.5, c='#228B22', label='Растительность')
ax.axhline(y=height_threshold, color='red', linestyle='--', alpha=0.7, label=f'Порог {height_threshold} м')
ax.set_title('4. Классификация')
ax.set_xlabel('X, м')
ax.set_ylabel('Z, м')
ax.set_ylim(-0.1, 0.9)
ax.legend(markerscale=10, fontsize=9)

# 1e: Вид сверху
ax = fig.add_subplot(gs[1, 1])
sc = ax.scatter(vegetation_classified[idx_v, 0], vegetation_classified[idx_v, 1],
                s=0.3, alpha=0.4, c=vegetation_classified[idx_v, 2], cmap='YlGn', vmin=0)
ax.set_title('5. Вид сверху\n(цвет = высота)')
ax.set_xlabel('X, м')
ax.set_ylabel('Y, м')
ax.set_aspect('equal')
plt.colorbar(sc, ax=ax, label='Z, м', shrink=0.8)

# 1f: Гистограмма высот
ax = fig.add_subplot(gs[1, 2])
ax.hist(vegetation_classified[:, 2], bins=50, color='#228B22', alpha=0.7, edgecolor='darkgreen')
mean_h = np.mean([p['height'] for p in plant_params])
ax.axvline(x=mean_h, color='red', linestyle='--', linewidth=2,
           label=f'Средняя высота GT = {mean_h:.2f} м')
ax.set_title('6. Распределение высот')
ax.set_xlabel('Высота Z, м')
ax.set_ylabel('Кол-во точек')
ax.legend(fontsize=9)

plt.savefig(f'{OUTPUT_DIR}/01_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nСохранён: 01_pipeline.png")


# --- График 2: Сравнение ВСЕХ методов (лог-шкала) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

methods_all = [r['method'] for r in all_results]
volumes_all = [r['volume'] for r in all_results]
colors_all = ['#2196F3' if r['type'] == 'voxel' else
              '#FF9800' if r['type'] == 'hull' else '#4CAF50'
              for r in all_results]

# Лог-шкала — все методы
bars = ax1.bar(range(len(methods_all)), volumes_all, color=colors_all, alpha=0.8,
               edgecolor='black', linewidth=0.5)
ax1.axhline(y=total_gt_volume, color='red', linestyle='--', linewidth=2,
            label=f'Ground Truth = {total_gt_volume:.4f} м³')
ax1.set_yscale('log')
ax1.set_xticks(range(len(methods_all)))
ax1.set_xticklabels(methods_all, rotation=55, ha='right', fontsize=8)
ax1.set_ylabel('Объём, м³ (лог. шкала)')
ax1.set_title('Все методы (логарифмическая шкала)')
ax1.legend(fontsize=10)

# Линейная шкала — только вменяемые методы (воксельные ≤0.02)
good_results = [r for r in all_results if r['type'] == 'voxel' and
                abs(r['error_pct']) < 500]
methods_good = [r['method'] for r in good_results]
volumes_good = [r['volume'] for r in good_results]
errors_good = [r['error_pct'] for r in good_results]

bars2 = ax2.bar(range(len(methods_good)), volumes_good, color='#2196F3',
                alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.axhline(y=total_gt_volume, color='red', linestyle='--', linewidth=2,
            label=f'Ground Truth')

for i, (bar, err) in enumerate(zip(bars2, errors_good)):
    color = '#228B22' if abs(err) < 30 else '#FF6600' if abs(err) < 100 else '#CC0000'
    ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.0003,
             f'{err:+.0f}%', ha='center', va='bottom', fontsize=10,
             fontweight='bold', color=color)

ax2.set_xticks(range(len(methods_good)))
ax2.set_xticklabels(methods_good, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Объём, м³')
ax2.set_title('Воксельный метод: детализация')
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_volume_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранён: 02_volume_comparison.png")


# --- График 3: Зависимость от размера вокселя ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

vs_list = sorted(voxel_results.keys())
vv_list = [voxel_results[v]['volume'] for v in vs_list]
ve_list = [abs(voxel_results[v]['error_pct']) for v in vs_list]
nv_list = [voxel_results[v]['n_voxels'] for v in vs_list]
vs_mm = [v * 1000 for v in vs_list]

# Объём
ax = axes[0]
ax.plot(vs_mm, vv_list, 'o-', color='#2196F3', linewidth=2, markersize=7)
ax.axhline(y=total_gt_volume, color='red', linestyle='--', linewidth=1.5,
           label=f'GT = {total_gt_volume:.5f}')
ax.fill_between(vs_mm, total_gt_volume * 0.8, total_gt_volume * 1.2,
                alpha=0.15, color='green', label='±20% от GT')
ax.set_xlabel('Размер вокселя, мм')
ax.set_ylabel('Объём, м³')
ax.set_title('Оценка объёма')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Ошибка
ax = axes[1]
ax.plot(vs_mm, ve_list, 's-', color='#f44336', linewidth=2, markersize=7)
ax.axhline(y=20, color='green', linestyle=':', alpha=0.7, label='20% ошибка')
ax.set_xlabel('Размер вокселя, мм')
ax.set_ylabel('|Ошибка|, %')
ax.set_title('Абсолютная ошибка')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Кол-во вокселей
ax = axes[2]
ax.plot(vs_mm, nv_list, 'D-', color='#9C27B0', linewidth=2, markersize=7)
ax.set_xlabel('Размер вокселя, мм')
ax.set_ylabel('Количество вокселей')
ax.set_title('Вычислительная нагрузка')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_voxel_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранён: 03_voxel_analysis.png")


# --- График 4: Почему Convex Hull / Alpha Shape не работают ---
fig, ax = plt.subplots(figsize=(10, 6))

method_names = []
method_volumes = []
method_colors = []

# Лучший воксельный
best_vs = min(voxel_results, key=lambda v: abs(voxel_results[v]['error_pct']))
method_names.append(f'Voxel {best_vs*1000:.0f}мм')
method_volumes.append(voxel_results[best_vs]['volume'])
method_colors.append('#2196F3')

# Convex Hull
method_names.append('Convex Hull')
method_volumes.append(ch_volume)
method_colors.append('#FF9800')

# Alpha shapes
for av in sorted(alpha_results.keys()):
    method_names.append(f'α-shape α={av}')
    method_volumes.append(alpha_results[av]['volume'])
    method_colors.append('#4CAF50')

method_names.append('Ground Truth')
method_volumes.append(total_gt_volume)
method_colors.append('#f44336')

bars = ax.barh(range(len(method_names)), method_volumes, color=method_colors,
               alpha=0.8, edgecolor='black', linewidth=0.5)
ax.axvline(x=total_gt_volume, color='red', linestyle='--', linewidth=2)

for i, (bar, vol) in enumerate(zip(bars, method_volumes)):
    if vol > 0.1:
        ratio = vol / total_gt_volume
        ax.text(vol + 0.02, bar.get_y() + bar.get_height() / 2,
                f'×{ratio:.0f}', va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(method_names)))
ax.set_yticklabels(method_names, fontsize=10)
ax.set_xlabel('Объём, м³')
ax.set_title('Convex Hull и Alpha Shape завышают объём\n'
             '(считают оболочку, а не тонкие структуры растений)')
ax.set_xscale('log')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_hull_vs_voxel.png', dpi=150, bbox_inches='tight')
plt.close()
print("Сохранён: 04_hull_vs_voxel.png")


# ============================================================
# 7. 3D ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ (Plotly)
# ============================================================

print("\nГенерация 3D визуализации...")
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Подвыборка для производительности
max_3d_pts = 8000

def subsample(pts, n):
    if len(pts) > n:
        idx = np.random.choice(len(pts), n, replace=False)
        return pts[idx]
    return pts

# --- 3D: Исходное облако с шумом ---
pts_sub = subsample(all_pts_noisy, max_3d_pts)
fig3d_raw = go.Figure(data=[go.Scatter3d(
    x=pts_sub[:, 0], y=pts_sub[:, 1], z=pts_sub[:, 2],
    mode='markers',
    marker=dict(size=1.2, color=pts_sub[:, 2], colorscale='Viridis',
                opacity=0.6, colorbar=dict(title='Z, м')),
    name='Точки'
)])
fig3d_raw.update_layout(
    title='Исходное облако точек (с шумом, выбросами и окклюзией)',
    scene=dict(
        xaxis_title='X, м', yaxis_title='Y, м', zaxis_title='Z, м',
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    width=900, height=650
)
fig3d_raw.write_html(f'{OUTPUT_DIR}/3d_raw.html')
print("Сохранён: 3d_raw.html")

# --- 3D: Классифицированное облако ---
g_sub = subsample(ground_classified, 3000)
v_sub = subsample(vegetation_classified, 5000)

fig3d_class = go.Figure()
fig3d_class.add_trace(go.Scatter3d(
    x=g_sub[:, 0], y=g_sub[:, 1], z=g_sub[:, 2],
    mode='markers',
    marker=dict(size=1.2, color='#8B4513', opacity=0.4),
    name='Земля'
))
fig3d_class.add_trace(go.Scatter3d(
    x=v_sub[:, 0], y=v_sub[:, 1], z=v_sub[:, 2],
    mode='markers',
    marker=dict(size=1.5, color=v_sub[:, 2], colorscale='YlGn',
                opacity=0.7, colorbar=dict(title='Z, м')),
    name='Растительность'
))
# Позиция сканера
fig3d_class.add_trace(go.Scatter3d(
    x=[scanner_pos[0]], y=[scanner_pos[1]], z=[scanner_pos[2]],
    mode='markers+text',
    marker=dict(size=8, color='red', symbol='diamond'),
    text=['TLS сканер'], textposition='top center',
    name='Сканер'
))
fig3d_class.update_layout(
    title='Классифицированное облако (земля / растительность)',
    scene=dict(
        xaxis_title='X, м', yaxis_title='Y, м', zaxis_title='Z, м',
        aspectmode='data',
        camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.0))
    ),
    width=900, height=650
)
fig3d_class.write_html(f'{OUTPUT_DIR}/3d_classified.html')
print("Сохранён: 3d_classified.html")


# ============================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================

results_data = {
    'ground_truth_volume_m3': total_gt_volume,
    'n_plants': len(plant_params),
    'mean_height_m': float(mean_h),
    'scanner_position': scanner_pos.tolist(),
    'points': {
        'raw_total': len(all_pts_clean),
        'after_occlusion': len(all_pts_occluded),
        'occlusion_removed_pct': round(100 * n_occluded / len(all_pts_clean), 1),
        'with_noise': len(all_pts_noisy),
        'after_sor': len(pts_after_sor),
        'after_radius': len(pts_filtered),
        'ground': len(ground_classified),
        'vegetation': len(vegetation_classified)
    },
    'voxel_results': {f'{k*1000:.0f}mm': {
        'volume': v['volume'],
        'n_voxels': v['n_voxels'],
        'error_pct': round(v['error_pct'], 1)
    } for k, v in voxel_results.items()},
    'convex_hull': {
        'volume': ch_volume,
        'error_pct': round(ch_error, 1)
    },
    'alpha_shape': {f'alpha_{k}': {
        'volume': v['volume'],
        'error_pct': round(v['error_pct'], 1)
    } for k, v in alpha_results.items()},
    'best_method': f'Voxel {best_vs*1000:.0f}mm',
    'best_error_pct': round(abs(voxel_results[best_vs]['error_pct']), 1)
}

with open(f'{OUTPUT_DIR}/results.json', 'w') as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print("\nСохранён: results.json")
print("\n" + "=" * 60)
print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
print("=" * 60)
print(f"\nЛучший метод: Voxel {best_vs*1000:.0f}мм, ошибка {abs(voxel_results[best_vs]['error_pct']):.1f}%")