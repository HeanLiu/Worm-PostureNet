import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from scipy.signal import savgol_filter, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.stats import circstd
import seaborn as sns


FRAME_RATE = 2  
TARGET_WORM_ID = 10  
SAVE_DIR = "outputs/" 


plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def load_data(worm_id):

    df = pd.read_csv('outputs/worm_tracking_data_all.txt', sep='\t')
    df = df[df['worm_id'] == worm_id].sort_values('frame')

    if df.empty:
        raise ValueError(f"No data found for worm ID {worm_id}")

    # Calculate time column
    df['time'] = df['frame'] / FRAME_RATE
    return df


def calculate_movement_metrics(df):

    # 1. Basic speed calculations
    df = calculate_speeds(df)

    # 2. Movement direction analysis
    df = calculate_directions(df)

    # 3. Body curvature and undulation
    df = calculate_curvature(df)

    # 4. Behavior classification
    df = classify_behaviors(df)

    return df


def calculate_speeds(df):

    dt = np.gradient(df['time'].values)

    # Center point speed
    dx = np.gradient(df['x'].values)
    dy = np.gradient(df['y'].values)
    df['center_speed'] = np.sqrt(dx ** 2 + dy ** 2) / dt


    kpts = ['head', 'body1', 'body2', 'body3', 'tail']
    for i, kpt in enumerate(kpts):
        x, y = df[f'x{i + 1}'].values, df[f'y{i + 1}'].values
        dx = np.gradient(x)
        dy = np.gradient(y)
        df[f'speed_{kpt}'] = savgol_filter(np.sqrt(dx ** 2 + dy ** 2) / dt, 11, 3)


    df['avg_speed'] = df[[f'speed_{kpt}' for kpt in kpts]].mean(axis=1)

    return df


def calculate_directions(df):

    dx = np.gradient(df['x'].values)
    dy = np.gradient(df['y'].values)
    df['movement_angle'] = np.arctan2(dy, dx)  # radians

    dx_body = df['x1'] - df['x5']
    dy_body = df['y1'] - df['y5']
    df['body_angle'] = np.arctan2(dy_body, dx_body)

    df['direction_change'] = np.abs(np.diff(df['body_angle'], prepend=np.nan))

    return df


def calculate_curvature(df):

    # Body segment vectors
    segments = []
    for i in range(4):
        seg_dx = df[f'x{i + 2}'] - df[f'x{i + 1}']
        seg_dy = df[f'y{i + 2}'] - df[f'y{i + 1}']
        segments.append(np.arctan2(seg_dy, seg_dx))

    curvatures = []
    for i in range(3):
        curvatures.append(np.abs(segments[i + 1] - segments[i]))
    df['curvature'] = pd.DataFrame(curvatures).mean(axis=0)

    # Undulation frequency analysis (using head speed)
    freqs, psd = welch(df['speed_head'], fs=FRAME_RATE, nperseg=min(64, len(df) // 2))
    df['dominant_freq'] = freqs[np.argmax(psd)]
    df['power_ratio'] = np.max(psd) / np.sum(psd)

    return df


def classify_behaviors(df):

    speed_thresh = 0.1 * df['avg_speed'].quantile(0.95)
    curve_thresh = df['curvature'].quantile(0.75)

    conditions = [
        (df['avg_speed'] > speed_thresh) & (df['curvature'] < curve_thresh / 2),
        (df['avg_speed'] < -speed_thresh) & (df['curvature'] < curve_thresh / 2),
        (df['curvature'] > curve_thresh * 1.5),
        (df['curvature'] > curve_thresh * 3),
        (df['avg_speed'].abs() < speed_thresh / 2)
    ]
    choices = ['forward', 'backward', 'turn', 'omega', 'pause']
    df['behavior'] = np.select(conditions, choices, default='other')

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
    """Plot movement trajectory"""
    plt.figure(figsize=(12, 10))

    # Trajectory line (color indicates time)
    points = np.array([df['x'], df['y']]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(df)))
    lc.set_array(np.arange(len(df)))
    plt.gca().add_collection(lc)
    plt.colorbar(lc, label='Frame number')

    # Mark key points
    plt.scatter(df['x'].iloc[0], df['y'].iloc[0], c='green', s=100, label='Start')
    plt.scatter(df['x'].iloc[-1], df['y'].iloc[-1], c='red', s=100, label='End')

    # Mark head direction every 10 frames
    for i in range(0, len(df), 10):
        dx = np.cos(df['body_angle'].iloc[i]) * 20
        dy = np.sin(df['body_angle'].iloc[i]) * 20
        plt.arrow(df['x1'].iloc[i], df['y1'].iloc[i],
                  dx, dy, head_width=5, fc='black')

    plt.title(f'Worm ID {worm_id} Movement Trajectory\n(Arrows indicate head direction)')
    plt.xlabel('X coordinate (pixels)')
    plt.ylabel('Y coordinate (pixels)')
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.savefig(f'{save_dir}/worm_{worm_id}_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_speed_analysis(df, worm_id, save_dir):
    """Plot speed analysis"""
    plt.figure(figsize=(15, 12))

    # 1. Speed comparison
    plt.subplot(3, 1, 1)
    plt.plot(df['time'], df['center_speed'], 'b-', label='Center speed')
    plt.plot(df['time'], df['avg_speed'], 'r-', label='5-point average speed')
    plt.title(f'Worm ID {worm_id} Speed Analysis')
    plt.ylabel('Speed (pixels/sec)')
    plt.legend()
    plt.grid(True)

    # 2. Body part speeds
    plt.subplot(3, 1, 2)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, kpt in enumerate(['head', 'body1', 'body2', 'body3', 'tail']):
        plt.plot(df['time'], df[f'speed_{kpt}'], color=colors[i],
                 label=f'{kpt} speed', alpha=0.7)
    plt.ylabel('Speed (pixels/sec)')
    plt.legend()
    plt.grid(True)


    plt.subplot(3, 1, 3)
    speed_data = df[['speed_head', 'speed_body1', 'speed_body2',
                     'speed_body3', 'speed_tail']].T
    plt.imshow(speed_data, aspect='auto', cmap='hot',
               extent=[df['time'].min(), df['time'].max(), 0, 5])
    plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5],
               ['Head', 'Body1', 'Body2', 'Body3', 'Tail'])
    plt.xlabel('Time (sec)')
    plt.ylabel('Body part')
    plt.colorbar(label='Speed (pixels/sec)')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/worm_{worm_id}_speed_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_direction_analysis(df, worm_id, save_dir):
    """Plot direction analysis"""
    plt.figure(figsize=(15, 8))

    # 1. Direction changes
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], np.degrees(df['body_angle']), 'b-')
    plt.title('Body direction angle changes')
    plt.ylabel('Angle (degrees)')
    plt.grid(True)

    # 2. Direction change rate
    plt.subplot(2, 2, 2)
    plt.plot(df['time'], np.degrees(df['direction_change']), 'r-')
    plt.title('Direction change rate')
    plt.ylabel('Angle change (deg/sec)')
    plt.grid(True)

    # 3. Direction histogram
    plt.subplot(2, 2, 3)
    angles = np.degrees(df['body_angle'])
    plt.hist(angles % 360, bins=36, range=(0, 360), color='green')
    plt.title('Body direction distribution')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Frequency')
    plt.grid(True)

    # 4. Polar plot
    plt.subplot(2, 2, 4, polar=True)
    theta = df['body_angle'].values % (2 * np.pi)
    r = np.ones_like(theta)
    plt.scatter(theta, r, c=df['time'], cmap='viridis', alpha=0.5)
    plt.title('Body direction polar distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/worm_{worm_id}_direction_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_curvature_analysis(df, worm_id, save_dir):
    """Plot curvature analysis"""
    plt.figure(figsize=(15, 10))

    # 1. Curvature over time
    plt.subplot(2, 2, 1)
    plt.plot(df['time'], df['curvature'], 'b-')
    plt.title('Body curvature changes')
    plt.ylabel('Curvature (radians)')
    plt.grid(True)

    # 2. Curvature vs speed
    plt.subplot(2, 2, 2)
    plt.scatter(df['curvature'], df['avg_speed'], c=df['time'], cmap='viridis')
    plt.title('Curvature vs speed relationship')
    plt.xlabel('Curvature (radians)')
    plt.ylabel('Average speed')
    plt.colorbar(label='Time (sec)')
    plt.grid(True)

    # 3. Curvature distribution
    plt.subplot(2, 2, 3)
    sns.histplot(df['curvature'], kde=True, color='purple')
    plt.title('Curvature distribution')
    plt.xlabel('Curvature (radians)')
    plt.grid(True)

    # 4. Curvature heatmap
    plt.subplot(2, 2, 4)
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

    plt.tight_layout()
    plt.savefig(f'{save_dir}/worm_{worm_id}_curvature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_behavior_stats(df, worm_id, save_dir):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    behavior_counts = df['behavior'].value_counts()
    plt.pie(behavior_counts, labels=behavior_counts.index,
            autopct='%1.1f%%', startangle=90)
    plt.title('Behavior time distribution')

     plt.subplot(2, 2, 2)
    for behavior in df['behavior'].unique():
        mask = df['behavior'] == behavior
        plt.scatter(df['time'][mask], [behavior] * sum(mask), label=behavior)
    plt.title('Behavior temporal distribution')
    plt.xlabel('Time (sec)')
    plt.ylabel('Behavior type')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    sns.boxplot(x='behavior', y='avg_speed', data=df)
    plt.title('Speed distribution by behavior')
    plt.xlabel('Behavior type')
    plt.ylabel('Average speed')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    sns.violinplot(x='behavior', y='curvature', data=df)
    plt.title('Curvature distribution by behavior')
    plt.xlabel('Behavior type')
    plt.ylabel('Curvature')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/worm_{worm_id}_behavior_stats.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_undulation_analysis(df, worm_id, save_dir):
     plt.figure(figsize=(15, 10))

    # 1. Undulation frequency
    plt.subplot(2, 2, 1)
    freqs, psd = welch(df['speed_head'], fs=FRAME_RATE, nperseg=min(64, len(df) // 2))
    plt.plot(freqs, psd, 'b-')
    plt.title('Head speed power spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    for kpt, color in zip(['head', 'body1', 'body2', 'body3', 'tail'],
                          ['red', 'blue', 'green', 'purple', 'orange']):
        plt.plot(df['time'], df[f'speed_{kpt}'], color=color, label=kpt)
    plt.title('Undulation wave propagation')
    plt.xlabel('Time (sec)')
    plt.ylabel('Speed')
    plt.legend()
    plt.grid(True)

    # 3. Phase delay
    plt.subplot(2, 2, 3)
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

    # 4. Wave pattern example
    plt.subplot(2, 2, 4)
    sample_time = df['time'].iloc[len(df) // 2]
    sample_frame = df[df['time'] >= sample_time].iloc[0]
    plt.plot([1, 2, 3, 4, 5],
             [sample_frame[f'speed_{kpt}'] for kpt in ['head', 'body1', 'body2', 'body3', 'tail']],
             'ro-')
    plt.title(f'Wave pattern at time={sample_time:.1f}s')
    plt.xlabel('Body part (1=head, 5=tail)')
    plt.ylabel('Speed')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/worm_{worm_id}_undulation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    print(f"Starting analysis for worm ID {TARGET_WORM_ID}...")

    try:
          df = load_data(TARGET_WORM_ID)

        df = calculate_movement_metrics(df)

        df.to_csv(f'{SAVE_DIR}/worm_{TARGET_WORM_ID}_analysis_data.csv', index=False)

        plot_all_results(df, TARGET_WORM_ID, SAVE_DIR)

        print(f"Analysis complete! Results saved to '{SAVE_DIR}' directory")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")