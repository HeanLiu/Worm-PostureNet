import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import argparse
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.stats import circstd
from scipy.ndimage import median_filter
import seaborn as sns

FRAME_RATE = 2.5
TARGET_WORM_ID = 10
SAVE_DIR = "***"

plt.rcParams['font.sans-serif'] = ['Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def fix_head_tail_swaps(df):
    df_fixed = df.copy().reset_index(drop=True)
    kpts_x = [f'x{i}' for i in range(1, 6)]
    kpts_y = [f'y{i}' for i in range(1, 6)]
    swap_count = 0

    for i in range(1, len(df_fixed)):
        head_curr = np.array([df_fixed.loc[i, 'x1'], df_fixed.loc[i, 'y1']])
        tail_curr = np.array([df_fixed.loc[i, 'x5'], df_fixed.loc[i, 'y5']])
        head_prev = np.array([df_fixed.loc[i - 1, 'x1'], df_fixed.loc[i - 1, 'y1']])
        tail_prev = np.array([df_fixed.loc[i - 1, 'x5'], df_fixed.loc[i - 1, 'y5']])

        cost_normal = np.linalg.norm(head_curr - head_prev) + np.linalg.norm(tail_curr - tail_prev)
        cost_swap = np.linalg.norm(head_curr - tail_prev) - np.linalg.norm(tail_curr - head_prev)

        if cost_swap < cost_normal:
            swap_count += 1
            for j in range(2):
                temp_x = df_fixed.loc[i, kpts_x[j]]
                df_fixed.loc[i, kpts_x[j]] = df_fixed.loc[i, kpts_x[4 - j]]
                df_fixed.loc[i, kpts_x[4 - j]] = temp_x
                temp_y = df_fixed.loc[i, kpts_y[j]]
                df_fixed.loc[i, kpts_y[j]] = df_fixed.loc[i, kpts_y[4 - j]]
                df_fixed.loc[i, kpts_y[4 - j]] = temp_y

    return df_fixed

def fix_zero_curvature_collapse(df):
    if 'curvature' not in df.columns:
        return df

    df_fixed = df.copy()
    head_tail_dist = np.sqrt((df_fixed['x1'] - df_fixed['x5']) ** 2 + (df_fixed['y1'] - df_fixed['y5']) ** 2)
    normal_body_length = head_tail_dist.quantile(0.90)

    fake_zero_mask = (df_fixed['curvature'].abs() < 0.001) & (head_tail_dist < normal_body_length * 0.3)
    collapse_count = fake_zero_mask.sum()

    if collapse_count > 0:
        cols_to_fix = ['curvature', 'max_abs_curvature', 'curvature_pt2', 'curvature_pt3', 'curvature_pt4']
        for col in cols_to_fix:
            if col in df_fixed.columns:
                df_fixed.loc[fake_zero_mask, col] = np.nan
                df_fixed[col] = df_fixed[col].interpolate(method='polynomial', order=2, limit=7, limit_direction='both')
                df_fixed[col] = df_fixed[col].interpolate(method='linear', limit=3, limit_direction='both')

    return df_fixed

def load_data(worm_id):
    df = pd.read_csv('***', sep='\t')
    df = df[df['worm_id'] == worm_id].sort_values('frame')

    if df.empty:
        raise ValueError(f"No data found for worm ID {worm_id}")

    coords_cols = ['x', 'y'] + [f'x{i}' for i in range(1, 6)] + [f'y{i}' for i in range(1, 6)]
    for col in coords_cols:
        if col in df.columns:
            pass

    df[coords_cols] = df[coords_cols].interpolate(method='linear', limit=5)
    df['time'] = df['frame'] / FRAME_RATE
    df = df[df['time'] <= 90.0].copy()
    return df

def calculate_movement_metrics(df):
    df = fix_head_tail_swaps(df)
    df = calculate_speeds(df)
    df = calculate_directions(df)
    df = calculate_curvature(df)
    df = classify_behaviors(df)
    return df

def calculate_speeds(df):
    dt = np.gradient(df['time'].values)
    dt = np.where(dt == 0, 1e-8, dt)

    dx = np.gradient(df['x'].values)
    dy = np.gradient(df['y'].values)
    df['center_speed'] = np.sqrt(dx ** 2 + dy ** 2) / dt

    vec_body_x = df['x1'].values - df['x5'].values
    vec_body_y = df['y1'].values - df['y5'].values
    norm_body = np.sqrt(vec_body_x ** 2 + vec_body_y ** 2) + 1e-8

    df['directed_speed'] = ((dx * vec_body_x + dy * vec_body_y) / norm_body) / dt
    df['directed_speed'] = savgol_filter(df['directed_speed'], window_length=5, polyorder=5)

    kpts = ['head', 'body1', 'body2', 'body3', 'tail']
    speed_matrix = []
    for i, kpt in enumerate(kpts):
        kpt_dx = np.gradient(df[f'x{i + 1}'].values)
        kpt_dy = np.gradient(df[f'y{i + 1}'].values)
        spd = np.sqrt(kpt_dx ** 2 + kpt_dy ** 2) / dt
        df[f'speed_{kpt}'] = savgol_filter(spd, window_length=5, polyorder=2)
        speed_matrix.append(df[f'speed_{kpt}'].values)

    df['avg_speed'] = np.mean(speed_matrix, axis=0)
    return df

def calculate_directions(df):
    dx_body = df['x1'] - df['x5']
    dy_body = df['y1'] - df['y5']
    raw_angles = np.arctan2(dx_body, dy_body)
    df['body_angle'] = raw_angles

    valid_mask = ~np.isnan(raw_angles)
    unwrapped_angles = np.full_like(raw_angles, np.nan)

    if np.any(valid_mask):
        unwrapped = np.unwrap(raw_angles[valid_mask])
        smoothed_angles = median_filter(unwrapped, size=15)
        unwrapped_angles[valid_mask] = smoothed_angles

    df['unwrapped_body_angle'] = unwrapped_angles

    dt = np.gradient(df['time'].values)
    dt = np.where(dt == 0, 1e-8, dt)
    dir_change = np.abs(np.gradient(unwrapped_angles) / dt)

    if np.any(valid_mask) and len(dir_change[valid_mask]) > 5:
        dir_change[valid_mask] = savgol_filter(dir_change[valid_mask], window_length=5, polyorder=2)

    df['direction_change'] = np.clip(dir_change, a_min=0, a_max=None)
    return df

def calculate_curvature(df):
    curvatures_all = []

    for i in range(2, 5):
        x1, y1 = df[f'x{i - 1}'].values, df[f'y{i - 1}'].values
        x2, y2 = df[f'x{i}'].values, df[f'y{i}'].values
        x3, y3 = df[f'x{i + 1}'].values, df[f'y{i + 1}'].values

        a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

        cross_product = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        area = cross_product / 2.0

        denom = a * b * c
        denom = np.where(denom == 0, 1e-8, denom)
        k = (4.0 * area) / denom

        curvatures_all.append(k)
        df[f'curvature_pt{i}'] = k

    df['max_abs_curvature'] = np.max(np.abs(curvatures_all), axis=0)
    df['curvature'] = np.mean(curvatures_all, axis=0)

    df = fix_zero_curvature_collapse(df)

    valid_speed = df['speed_head'].dropna().values
    if len(valid_speed) > 10:
        freqs, psd = welch(valid_speed, fs=FRAME_RATE, nperseg=min(64, len(valid_speed) // 2))
        df['dominant_freq'] = freqs[np.argmax(psd)]
        df['power_ratio'] = np.max(psd) / np.sum(psd)
    else:
        df['dominant_freq'], df['power_ratio'] = np.nan, np.nan

    return df

def classify_behaviors(df):
    speed_thresh = 0.15 * df['center_speed'].quantile(0.90)
    curve_thresh = df['max_abs_curvature'].quantile(0.85)
    turn_rate_thresh = df['direction_change'].quantile(0.85)

    conditions = [
        (df['max_abs_curvature'] > curve_thresh * 2.0) & (df['center_speed'] < speed_thresh * 2),
        (df['direction_change'] > turn_rate_thresh) & (df['max_abs_curvature'] > curve_thresh),
        (df['directed_speed'] > speed_thresh),
        (df['directed_speed'] < -speed_thresh),
        (np.abs(df['directed_speed']) <= speed_thresh)
    ]
    choices = ['Omega', 'Turn', 'Forward', 'Backward', 'Pause']
    raw_behaviors = np.select(conditions, choices, default='Other')

    behavior_map = {'Pause': 0, 'Forward': 1, 'Backward': 2, 'Turn': 3, 'Omega': 4, 'Other': 5}
    reverse_map = {v: k for k, v in behavior_map.items()}
    numeric_behaviors = np.array([behavior_map.get(b, 5) for b in raw_behaviors])

    smoothed_numeric = median_filter(numeric_behaviors, size=15)
    df['behavior'] = [reverse_map[val] for val in smoothed_numeric]

    return df

def plot_all_results(df, worm_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plot_trajectory(df, worm_id, save_dir)
    plot_speed_analysis(df, worm_id, save_dir)
    plot_direction_analysis(df, worm_id, save_dir)
    plot_curvature_analysis(df, worm_id, save_dir)
    plot_behavior_stats(df, worm_id, save_dir)
    plot_undulation_analysis(df, worm_id, save_dir)

def plot_trajectory(df, worm_id, save_dir):
    plt.figure(figsize=(10, 8))

    points = np.array([df['x'], df['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(df)))
    lc.set_array(np.arange(len(df)))
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label='Frame number')

    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, label='Start')
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, label='End')

    for i in range(0, len(df), 10):
        dx = np.cos(df['body_angle'].iloc[i]) * (50 / 1.0)
        dy = np.sin(df['body_angle'].iloc[i]) * (50 / 1.0)
        plt.arrow(df['x1'].iloc[i], df['y1'].iloc[i], dx, dy,
                  head_width=(15 / 1.0), fc='black')

    plt.title(f'Worm ID {worm_id} Movement Trajectory')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.savefig(f'{save_dir}/worm_{worm_id}_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_analysis(df, worm_id, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['center_speed'], 'b-', label='Center speed')
    plt.plot(df['time'], df['avg_speed'], 'r-', label='5-point average speed')
    plt.title(f'Worm ID {worm_id} Speed Analysis')
    plt.xlabel('Time (sec)')
    plt.ylabel('Speed (pixels/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_speed_1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, kpt in enumerate(['head', 'body1', 'body2', 'body3', 'tail']):
        plt.plot(df['time'], df[f'speed_{kpt}'], color=colors[i],
                 label=f'{kpt} speed', alpha=0.7)
    plt.title('Body Part Speeds')
    plt.xlabel('Time (sec)')
    plt.ylabel('Speed (pixels/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_speed_2_bodyparts.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    speed_data = df[['speed_head', 'speed_body1', 'speed_body2',
                     'speed_body3', 'speed_tail']].T
    plt.imshow(speed_data, aspect='auto', cmap='hot',
               extent=[df['time'].min(), df['time'].max(), 0, 5])
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5],
               ['Head', 'Body1', 'Body2', 'Body3', 'Tail'])
    plt.title('Speed Heatmap')
    plt.xlabel('Time (sec)')
    plt.ylabel('Body part')
    plt.colorbar(label='Speed (pixels/s)')
    plt.savefig(f'{save_dir}/worm_{worm_id}_speed_3_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_direction_analysis(df, worm_id, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], np.degrees(df['unwrapped_body_angle']), 'b-', linewidth=1.5)
    plt.title('Body direction angle changes')
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_dir_1_angle_changes.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], np.degrees(df['direction_change']), 'r-', linewidth=1.5)
    plt.title('Direction change rate')
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle change (deg/sec)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_dir_2_change_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    angles = np.degrees(df['body_angle'])
    plt.hist(angles % 360, bins=36, range=(0, 360), color='green')
    plt.title('Body direction distribution')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_dir_3_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    theta = df['body_angle'].values % (2 * np.pi)
    r = np.ones_like(theta)
    scatter = ax.scatter(theta, r, c=df['time'], cmap='viridis', alpha=0.5)
    plt.title('Body direction polar distribution', pad=20)
    plt.colorbar(scatter, label='Time (sec)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_dir_4_polar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_curvature_analysis(df, worm_id, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(df['time'], df['curvature'], 'b-')
    plt.title('Body curvature changes')
    plt.xlabel('Time (sec)')
    plt.ylabel('Curvature (1/pixel)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_curve_1_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    speed_limit_pixel = 500 / 1.0
    active_df = df[(df['directed_speed'] > 0) & (df['directed_speed'] < speed_limit_pixel)].copy()

    if not active_df.empty:
        sns.scatterplot(
            x='max_abs_curvature',
            y='directed_speed',
            hue='behavior',
            palette='Set1',
            data=active_df,
            alpha=0.7,
            s=40
        )
        sns.regplot(
            x='max_abs_curvature',
            y='directed_speed',
            data=active_df,
            scatter=False,
            line_kws={'color': 'black', 'linewidth': 2.5, 'linestyle': '--'}
        )
        plt.legend(title='Behavior', loc='upper right')

    plt.title('Max Absolute Curvature vs Directed Speed')
    plt.xlabel('Max Absolute Curvature (1/pixel)')
    plt.ylabel('Directed Forward Speed (pixels/s)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_curve_2_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df['max_abs_curvature'], kde=True, color='purple')
    plt.title('Max Absolute Curvature Distribution')
    plt.xlabel('Absolute Curvature (1/pixel)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_curve_3_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    segments = []
    for i in range(4):
        seg_curve = np.abs(df[f'x{i + 2}'] - df[f'x{i + 1}']) + np.abs(df[f'y{i + 2}'] - df[f'y{i + 1}'])
        segments.append(seg_curve)
    plt.imshow(pd.DataFrame(segments), aspect='auto', cmap='hot',
               extent=[df['time'].min(), df['time'].max(), 0, 4])
    plt.yticks([0.5, 1.5, 2.5, 3.5], ['Head-Body1', 'Body1-Body2', 'Body2-Body3', 'Body3-Tail'])
    plt.xlabel('Time (sec)')
    plt.ylabel('Body segment')
    plt.colorbar(label='Curvature intensity')
    plt.title('Segmental curvature heatmap')
    plt.savefig(f'{save_dir}/worm_{worm_id}_curve_4_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_behavior_stats(df, worm_id, save_dir):
    plt.figure(figsize=(8, 8))
    behavior_counts = df['behavior'].value_counts()
    plt.pie(behavior_counts, labels=behavior_counts.index,
            autopct='%1.1f%%', startangle=90)
    plt.title('Behavior time distribution')
    plt.savefig(f'{save_dir}/worm_{worm_id}_behav_1_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    for behavior in df['behavior'].unique():
        mask = df['behavior'] == behavior
        plt.scatter(df['time'][mask], [behavior] * sum(mask), label=behavior, alpha=0.6)
    plt.title('Behavior temporal distribution')
    plt.xlabel('Time (sec)')
    plt.ylabel('Behavior type')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_behav_2_temporal.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='behavior', y='directed_speed', data=df)
    plt.title('Directed Speed distribution by behavior')
    plt.xlabel('Behavior type')
    plt.ylabel('Directed Speed (pixels/s)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_behav_3_vs_speed.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(x='behavior', y='max_abs_curvature', data=df)
    plt.title('Curvature distribution by behavior')
    plt.xlabel('Behavior type')
    plt.ylabel('Max Absolute Curvature (1/pixel)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_behav_4_vs_curvature.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_undulation_analysis(df, worm_id, save_dir):
    plt.figure(figsize=(10, 5))
    freqs, psd = welch(df['speed_head'], fs=FRAME_RATE, nperseg=min(64, len(df) // 2))
    plt.plot(freqs, psd, 'b-')
    plt.title('Head speed power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_undul_1_spectrum.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    for kpt, color in zip(['head', 'body1', 'body2', 'body3', 'tail'],
                          ['red', 'blue', 'green', 'purple', 'orange']):
        plt.plot(df['time'], df[f'speed_{kpt}'], color=color, label=kpt)
    plt.title('Undulation wave propagation')
    plt.xlabel('Time (sec)')
    plt.ylabel('Speed (pixels/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_undul_2_propagation.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    delays = []
    for kpt in ['body1', 'body2', 'body3', 'tail']:
        corr = np.correlate(df['speed_head'], df[f'speed_{kpt}'], mode='full')
        delay = (np.argmax(corr) - len(df) + 1) / FRAME_RATE
        delays.append(delay)
    plt.plot(range(1, 5), delays, 'bo-')
    plt.title('Phase delay relative to head')
    plt.xlabel('Body part (1=head, 4=tail)')
    plt.ylabel('Delay time (sec)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_undul_3_delay.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    sample_time = df['time'].iloc[len(df) // 2]
    sample_frame = df[df['time'] >= sample_time].iloc[0]
    plt.plot([1, 2, 3, 4, 5],
             [sample_frame[f'speed_{kpt}'] for kpt in ['head', 'body1', 'body2', 'body3', 'tail']],
             'ro-')
    plt.title(f'Wave pattern at time={sample_time:.1f}s')
    plt.xlabel('Body part (1=head, 5=tail)')
    plt.ylabel('Speed (pixels/s)')
    plt.grid(True)
    plt.savefig(f'{save_dir}/worm_{worm_id}_undul_4_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='***.pt')
    parser.add_argument('--conf_thres', type=float, default=0.25)
    opt = parser.parse_args()

    try:
        if getattr(opt, 'conf_thresh', 0.5) > 0.3:
            TARGET_WORM_ID = None
    except AttributeError:
        pass

    df = load_data(TARGET_WORM_ID)
    df = calculate_movement_metrics(df)
    df.to_csv(f'{SAVE_DIR}/worm_{TARGET_WORM_ID}_***.csv', index=False)
    plot_all_results(df, TARGET_WORM_ID, SAVE_DIR)