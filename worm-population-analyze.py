import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import kruskal
from itertools import combinations

FRAME_RATE = 2.5
PIXEL_TO_UM = 14.376
SAVE_DIR = "path/to/output/*"

TARGET_FILES_DICT = {
    "path/to/input/*.txt": 30,
}

plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False



def fix_head_tail_swaps(df):
    df_fixed = df.copy().reset_index(drop=True)
    kpts_x = [f'x{i}' for i in range(1, 6)]
    kpts_y = [f'y{i}' for i in range(1, 6)]
    for i in range(1, len(df_fixed)):
        h_curr = np.array([df_fixed.loc[i, 'x1'], df_fixed.loc[i, 'y1']])
        t_curr = np.array([df_fixed.loc[i, 'x5'], df_fixed.loc[i, 'y5']])
        h_prev = np.array([df_fixed.loc[i - 1, 'x1'], df_fixed.loc[i - 1, 'y1']])
        t_prev = np.array([df_fixed.loc[i - 1, 'x5'], df_fixed.loc[i - 1, 'y5']])
        if (np.linalg.norm(h_curr - t_prev) + np.linalg.norm(t_curr - h_prev) <
                np.linalg.norm(h_curr - h_prev) + np.linalg.norm(t_curr - t_prev)):
            for j in range(2):
                df_fixed.loc[i, kpts_x[j]], df_fixed.loc[i, kpts_x[4 - j]] = (
                    df_fixed.loc[i, kpts_x[4 - j]], df_fixed.loc[i, kpts_x[j]])
                df_fixed.loc[i, kpts_y[j]], df_fixed.loc[i, kpts_y[4 - j]] = (
                    df_fixed.loc[i, kpts_y[4 - j]], df_fixed.loc[i, kpts_y[j]])
    return df_fixed


def fix_zero_curvature_collapse(df):
    if 'curvature' not in df.columns:
        return df
    df_fixed = df.copy()
    head_tail_dist = np.sqrt((df_fixed['x1'] - df_fixed['x5']) ** 2 + (df_fixed['y1'] - df_fixed['y5']) ** 2)
    normal_body_length = head_tail_dist.quantile(0.90)
    fake_zero_mask = (df_fixed['curvature'].abs() < 0.001) & (head_tail_dist < normal_body_length * 0.3)
    if fake_zero_mask.sum() > 0:
        cols_to_fix = ['curvature', 'max_abs_curvature', 'curvature_pt2', 'curvature_pt3', 'curvature_pt4']
        for col in cols_to_fix:
            if col in df_fixed.columns:
                df_fixed.loc[fake_zero_mask, col] = np.nan
                df_fixed[col] = df_fixed[col].interpolate(method='polynomial', order=2, limit=7,
                                                          limit_direction='both')
                df_fixed[col] = df_fixed[col].interpolate(method='linear', limit=3, limit_direction='both')
    return df_fixed


def calculate_movement_metrics(df):
    df = fix_head_tail_swaps(df)

    n = len(df)
    dt = np.gradient(df['time'].values)
    dt = np.where(dt == 0, 1e-8, dt)
    dx, dy = np.gradient(df['x'].values), np.gradient(df['y'].values)
    df['center_speed'] = np.sqrt(dx ** 2 + dy ** 2) / dt

    vec_body_x = df['x1'].values - df['x5'].values
    vec_body_y = df['y1'].values - df['y5'].values
    norm_body = np.sqrt(vec_body_x ** 2 + vec_body_y ** 2) + 1e-8

    normal_body_length = norm_body.mean() if norm_body.mean() > 0 else 1.0
    ht_ratio = norm_body / normal_body_length

    wlen = min(5, n if n % 2 == 1 else n - 1)
    if wlen < 3:
        df['directed_speed'] = ((dx * vec_body_x + dy * vec_body_y) / norm_body) / dt
    else:
        df['directed_speed'] = savgol_filter(((dx * vec_body_x + dy * vec_body_y) / norm_body) / dt, window_length=wlen,
                                             polyorder=2)

    speed_matrix = []
    for i in range(5):
        kpt_dx = np.gradient(df[f'x{i + 1}'].values)
        kpt_dy = np.gradient(df[f'y{i + 1}'].values)
        spd_raw = np.sqrt(kpt_dx ** 2 + kpt_dy ** 2) / dt
        spd = savgol_filter(spd_raw, window_length=wlen, polyorder=2) if wlen >= 3 else spd_raw
        df[f'speed_pt{i + 1}'] = spd
        speed_matrix.append(spd)
    df['avg_speed'] = np.mean(speed_matrix, axis=0)
    df['speed_head'] = df['speed_pt1']

    raw_angles = np.arctan2(vec_body_y, vec_body_x)
    df['body_angle'] = raw_angles
    valid_mask = ~np.isnan(raw_angles)
    unwrapped_angles = np.full_like(raw_angles, np.nan)
    if np.any(valid_mask):
        unwrapped_angles[valid_mask] = median_filter(np.unwrap(raw_angles[valid_mask]), size=min(15, n))
    df['unwrapped_body_angle'] = unwrapped_angles

    dir_change = np.abs(np.gradient(unwrapped_angles) / dt)
    if np.any(valid_mask) and len(dir_change[valid_mask]) > 5:
        dir_change[valid_mask] = savgol_filter(dir_change[valid_mask], window_length=min(5, len(
            dir_change[valid_mask]) if len(dir_change[valid_mask]) % 2 == 1 else len(dir_change[valid_mask]) - 1),
                                               polyorder=2)
    df['direction_change'] = np.clip(dir_change, a_min=0, a_max=None)

    curvatures_all = []
    for i in range(2, 5):
        x1, y1 = df[f'x{i - 1}'].values, df[f'y{i - 1}'].values
        x2, y2 = df[f'x{i}'].values, df[f'y{i}'].values
        x3, y3 = df[f'x{i + 1}'].values, df[f'y{i + 1}'].values
        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        area = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / 2.0
        denom = np.where(a * b * c == 0, 1e-8, a * b * c)
        k = (4.0 * area) / denom
        curvatures_all.append(k)
        df[f'curvature_pt{i}'] = k
    df['max_abs_curvature'] = np.max(np.abs(curvatures_all), axis=0)
    df['curvature'] = np.mean(curvatures_all, axis=0)

    df = fix_zero_curvature_collapse(df)

    speed_thresh = 0.15 * df['center_speed'].quantile(0.90)
    curve_thresh = df['max_abs_curvature'].quantile(0.85)
    turn_rate_thresh = df['direction_change'].quantile(0.85)

    conds = [
        ((ht_ratio < 0.35) | (df['max_abs_curvature'] > curve_thresh * 1.5)) & (
                df['center_speed'] < speed_thresh * 2.5),
        (df['direction_change'] > turn_rate_thresh) & (df['max_abs_curvature'] > curve_thresh),
        (df['directed_speed'] > speed_thresh),
        (df['directed_speed'] < -speed_thresh),
        (np.abs(df['directed_speed']) <= speed_thresh)
    ]
    raw_behaviors = np.select(conds, ['Omega', 'Turn', 'Forward', 'Backward', 'Pause'], default='Other')
    behavior_map = {'Pause': 0, 'Forward': 1, 'Backward': 2, 'Turn': 3, 'Omega': 4, 'Other': 5}
    rev_map = {v: k for k, v in behavior_map.items()}
    numeric_behaviors = np.array([behavior_map.get(b, 5) for b in raw_behaviors])

    df['behavior'] = [rev_map[val] for val in median_filter(numeric_behaviors, size=min(5, n))]

    return df


def process_population_data():
    os.makedirs(SAVE_DIR, exist_ok=True)
    all_worms_data = []

    for filepath, top_n in TARGET_FILES_DICT.items():
        if not os.path.exists(filepath):
            continue

        df_file = pd.read_csv(filepath, sep='\t')
        df_file['worm_id'] = df_file['worm_id'].replace({10: 88})
        df_file['time'] = df_file['frame'] / FRAME_RATE
        df_file = df_file[df_file['time'] <= 90.0].copy()

        id_counts = df_file['worm_id'].value_counts()
        best_ids = id_counts.head(top_n).index.tolist()

        for w_id in best_ids:
            df_worm = df_file[df_file['worm_id'] == w_id].sort_values('frame').copy()

            if len(df_worm) < 15:
                continue

            coords_cols = ['x', 'y'] + [f'x{i}' for i in range(1, 6)] + [f'y{i}' for i in range(1, 6)]
            df_worm[coords_cols] = df_worm[coords_cols].interpolate(method='linear', limit=5)

            unique_label = f"{os.path.basename(filepath)}_ID{w_id}"

            df_processed = calculate_movement_metrics(df_worm)
            df_processed['source_file'] = os.path.basename(filepath)
            df_processed['unique_worm_id'] = unique_label

            all_worms_data.append(df_processed)

    if not all_worms_data:
        raise ValueError("No valid data extracted. Please check file paths.")

    combined_df = pd.concat(all_worms_data, ignore_index=True)
    return combined_df, len(all_worms_data)


def add_significance_brackets(ax, data, x_col, y_col, order, threshold=0.05):
    from scipy.stats import mannwhitneyu
    sig_pairs = []
    for a, b in combinations(order, 2):
        g1 = data[data[x_col] == a][y_col].dropna()
        g2 = data[data[x_col] == b][y_col].dropna()
        if len(g1) < 3 or len(g2) < 3:
            continue
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        if p < threshold:
            sig_pairs.append((a, b, p))

    y_max = data[y_col].quantile(0.99)
    step = y_max * 0.08
    for i, (a, b, p) in enumerate(sig_pairs[:5]):
        x1, x2 = order.index(a), order.index(b)
        y = y_max + step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y - step * 0.3, y, y, y - step * 0.3],
                lw=1, color='black')
        stars = '***' if p < 0.001 else ('**' if p < 0.01 else '*')
        ax.text((x1 + x2) / 2, y + step * 0.05, stars,
                ha='center', va='bottom', fontsize=10)


def plot_population_results(df, n_worms):
    behavior_order = ['Forward', 'Backward', 'Pause', 'Turn', 'Omega']
    behavior_order = [b for b in behavior_order if b in df['behavior'].unique()]

    behavior_colors = {
        'Forward': '#4CAF8C',
        'Pause': '#E07B54',
        'Backward': '#7B9ED9',
        'Turn': '#C57BD6',
        'Omega': '#E8C44A',
        'Other': '#AAAAAA'
    }
    palette = [behavior_colors.get(b, '#AAAAAA') for b in behavior_order]

    plt.figure(figsize=(9, 9))
    behavior_counts = df['behavior'].value_counts()

    def autopct_filter(pct):
        return f'{pct:.1f}%' if pct >= 1.0 else ''

    wedge_colors = [behavior_colors.get(b, '#AAAAAA') for b in behavior_counts.index]
    wedges, texts, autotexts = plt.pie(
        behavior_counts,
        labels=None,
        autopct=autopct_filter,
        pctdistance=0.75,
        startangle=90,
        colors=wedge_colors,
        wedgeprops=dict(linewidth=1.5, edgecolor='white')
    )
    for at in autotexts:
        at.set_fontsize(11)
    plt.legend(
        wedges,
        [f'{label}  ({v / behavior_counts.sum() * 100:.1f}%)'
         for label, v in behavior_counts.items()],
        title='Behavior',
        loc='center left',
        bbox_to_anchor=(0, 0.05),
        fontsize=11,
        title_fontsize=12,
        frameon=True,
        edgecolor='gray'
    )
    plt.title(f'Population-Level Behavioral State Distribution\n(N = {n_worms} worms, {len(df)} frames)',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f'{SAVE_DIR}/pop_1_behavior_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    df_spd = df[df['behavior'].isin(behavior_order)].copy()
    df_spd['avg_speed'] = df_spd['avg_speed'].clip(0, df_spd['avg_speed'].quantile(0.99))

    sns.boxplot(x='behavior', y='avg_speed', data=df_spd,
                order=behavior_order, palette=palette,
                showfliers=True,
                flierprops=dict(marker='o', markersize=2, alpha=0.3),
                ax=axes[0])
    sns.stripplot(x='behavior', y='avg_speed', data=df_spd,
                  order=behavior_order, color='black',
                  alpha=0.15, size=2, jitter=True, ax=axes[0])

    groups_spd = [df_spd[df_spd['behavior'] == b]['avg_speed'].dropna()
                  for b in behavior_order if len(df_spd[df_spd['behavior'] == b]) > 2]
    if len(groups_spd) >= 2:
        stat, p_kw = kruskal(*groups_spd)
        p_text = f'p < 0.001' if p_kw < 0.001 else f'p = {p_kw:.3f}'
        axes[0].text(0.97, 0.97, f'Kruskal–Wallis\nH = {stat:.1f}, {p_text}',
                     transform=axes[0].transAxes, fontsize=10,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    axes[0].set_title('Locomotion Speed across Behavioral States', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Behavioral State', fontsize=12)
    axes[0].set_ylabel('5-Point Avg Speed (pixels/s)', fontsize=12)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    df_curv = df[df['behavior'].isin(behavior_order)].copy()
    df_curv['max_abs_curvature'] = df_curv['max_abs_curvature'].clip(
        0, df_curv['max_abs_curvature'].quantile(0.99))

    sns.boxplot(x='behavior', y='max_abs_curvature', data=df_curv,
                order=behavior_order, palette=palette,
                showfliers=True,
                flierprops=dict(marker='o', markersize=2, alpha=0.3),
                ax=axes[1])
    sns.stripplot(x='behavior', y='max_abs_curvature', data=df_curv,
                  order=behavior_order, color='black',
                  alpha=0.15, size=2, jitter=True, ax=axes[1])

    groups_curv = [df_curv[df_curv['behavior'] == b]['max_abs_curvature'].dropna()
                   for b in behavior_order if len(df_curv[df_curv['behavior'] == b]) > 2]
    if len(groups_curv) >= 2:
        stat_c, p_kw_c = kruskal(*groups_curv)
        p_text_c = f'p < 0.001' if p_kw_c < 0.001 else f'p = {p_kw_c:.3f}'
        axes[1].text(0.97, 0.97, f'Kruskal–Wallis\nH = {stat_c:.1f}, {p_text_c}',
                     transform=axes[1].transAxes, fontsize=10,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    axes[1].set_title('Body Curvature across Behavioral States', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Behavioral State', fontsize=12)
    axes[1].set_ylabel('Max Absolute Curvature (1/pixel)', fontsize=12)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle(f'Population-Level Kinematic Differences across Behavioral States\n'
                 f'(N = {n_worms} worms, {len(df)} frames)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/pop_2_behavior_kinematics.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    speed_upper = df['directed_speed'].quantile(0.99)
    active_df = df[
        (df['directed_speed'] > 0.5) &
        (df['directed_speed'] < speed_upper) &
        (df['max_abs_curvature'] < df['max_abs_curvature'].quantile(0.99))
        ].copy()

    if len(active_df) > 10:
        sns.regplot(x='max_abs_curvature', y='directed_speed', data=active_df,
                    scatter_kws={'alpha': 0.2, 'color': '#4CAF8C', 's': 15,
                                 'edgecolors': 'none'},
                    line_kws={'color': '#d62728', 'linewidth': 2.5,
                              'linestyle': '--'},
                    ci=95)
        r, p = pearsonr(active_df['max_abs_curvature'], active_df['directed_speed'])
        p_text = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
        stats_text = f"Pearson's r = {r:.3f}\n{p_text}\nN = {n_worms} worms\n{len(active_df)} forward frames"
        plt.text(0.97, 0.97, stats_text, transform=plt.gca().transAxes, fontsize=11,
                 ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f'Population-Level Inverse Relationship Between\n'
              f'Body Curvature and Forward Locomotion Speed\n'
              f'(N = {n_worms} worms, {len(active_df)} forward-locomotion frames)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Max Absolute Curvature (1/pixel)', fontsize=12)
    plt.ylabel('Directed Forward Speed (pixels/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/pop_3_curvature_speed_regression.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    df_plot = df.copy()
    df_plot['avg_speed'] = df_plot['avg_speed'].clip(
        df_plot['avg_speed'].quantile(0.01),
        df_plot['avg_speed'].quantile(0.99))
    df_plot['center_speed'] = df_plot['center_speed'].clip(
        df_plot['center_speed'].quantile(0.01),
        df_plot['center_speed'].quantile(0.99))

    speed_melt = pd.melt(df_plot, id_vars=['unique_worm_id'],
                         value_vars=['center_speed', 'avg_speed'],
                         var_name='Metric', value_name='Speed')
    speed_melt['Metric'] = speed_melt['Metric'].replace(
        {'center_speed': 'Centroid Speed', 'avg_speed': '5-Point Avg Speed'})

    sns.violinplot(x='Metric', y='Speed', data=speed_melt,
                   hue='Metric', palette='muted', inner='quartile', legend=False)
    plt.title(f'Population-Level Speed Distribution\n'
              f'(N = {n_worms} worms)',
              fontsize=13, fontweight='bold')
    plt.ylabel('Speed (pixels/s)', fontsize=12)
    plt.xlabel('')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/pop_4_speed_violin.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    df_bp = df.copy()
    df_bp['avg_speed'] = df_bp['avg_speed'].clip(
        df_bp['avg_speed'].quantile(0.005),
        df_bp['avg_speed'].quantile(0.995))
    order_spd = df_bp.groupby('unique_worm_id')['avg_speed'].median() \
        .sort_values(ascending=False).index
    short_labels_spd = [uid.replace(
        os.path.basename(list(TARGET_FILES_DICT.keys())[0]) + '_ID', 'ID')
        for uid in order_spd]

    sns.boxplot(x='unique_worm_id', y='avg_speed', data=df_bp,
                order=order_spd, palette='husl', ax=axes[0],
                showfliers=True,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    axes[0].set_xticks(range(len(order_spd)))
    axes[0].set_xticklabels(short_labels_spd, rotation=45, ha='right', fontsize=9)
    axes[0].set_title(f'Inter-Individual Variation in Locomotion Speed\n'
                      f'across {n_worms} C. elegans (sorted by median)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('5-Point Avg Speed (pixels/s)', fontsize=11)
    axes[0].set_xlabel('Worm ID', fontsize=11)
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.5)

    df_curv2 = df.copy()
    df_curv2['max_abs_curvature'] = df_curv2['max_abs_curvature'].clip(
        0, df_curv2['max_abs_curvature'].quantile(0.99))
    order_curv = df_curv2.groupby('unique_worm_id')['max_abs_curvature'].median() \
        .sort_values(ascending=False).index
    short_labels_curv = [uid.replace(
        os.path.basename(list(TARGET_FILES_DICT.keys())[0]) + '_ID', 'ID')
        for uid in order_curv]

    sns.boxplot(x='unique_worm_id', y='max_abs_curvature', data=df_curv2,
                order=order_curv, palette='YlGnBu', ax=axes[1],
                showfliers=True,
                flierprops=dict(marker='o', markersize=2, alpha=0.3))
    axes[1].set_xticks(range(len(order_curv)))
    axes[1].set_xticklabels(short_labels_curv, rotation=45, ha='right', fontsize=9)
    axes[1].set_title(f'Inter-Individual Variation in Body Curvature\n'
                      f'across {n_worms} C. elegans (sorted by median)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Max Absolute Curvature (1/pixel)', fontsize=11)
    axes[1].set_xlabel('Worm ID', fontsize=11)
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.5)

    plt.suptitle('Population-Level Inter-Individual Kinematic Variation',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/pop_5_individual_variation.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    try:
        combined_df, total_worms = process_population_data()
        combined_df.to_csv(f'{SAVE_DIR}/population_data.csv', index=False)
        plot_population_results(combined_df, total_worms)
    except Exception as e:
        pass