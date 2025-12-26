import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
import os
import cv2
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

=
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class WormBehaviorAnalyzer:
    def __init__(self, data_path, fps=2):
        
        self.data_path = data_path
        self.fps = fps
        self.dt = 1.0 / fps
        self.data = None
        self.results = {}
        self.visualization_data = {}

  
        self.SPEED_THRESHOLD = 0.05  
        self.PAUSE_THRESHOLD = 0.01 
        self.OMEGA_ANGLE_THRESHOLD = 120 
        self.COILING_CURVATURE_THRESHOLD = 2.0  
        self.PIROUETTE_ANGULAR_VELOCITY = 30  

      
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):

        self.data = pd.read_csv(self.data_path, sep='\t')

        self.data.sort_values(['frame', 'worm_id'], inplace=True)

        worm_ids = self.data['worm_id'].unique()

        for worm_id in worm_ids:
            worm_data = self.data[self.data['worm_id'] == worm_id].copy()


            head_x = worm_data['x1'].values
            head_y = worm_data['y1'].values

            tail_x = worm_data['x2'].values
            tail_y = worm_data['y2'].values

            body_x = (worm_data['x3'] + worm_data['x4'] + worm_data['x5']) / 3
            body_y = (worm_data['y3'] + worm_data['y4'] + worm_data['y5']) / 3

            dx = np.diff(head_x, prepend=head_x[0])
            dy = np.diff(head_y, prepend=head_y[0])
            distance = np.sqrt(dx**2 + dy**2)
            speed = distance / self.dt  

            speed_smoothed = gaussian_filter1d(speed, sigma=2)

            body_length = np.sqrt((head_x - tail_x)**2 + (head_y - tail_y)**2)

            body_angle = np.arctan2(tail_y - head_y, tail_x - head_x) * 180 / np.pi

            motion_angle = np.arctan2(np.diff(head_y, prepend=head_y[0]),
                                     np.diff(head_x, prepend=head_x[0])) * 180 / np.pi

            curvature = self.calculate_curvature(worm_data)

            w = worm_data['w'].values
            h = worm_data['h'].values
            contraction = (w - np.mean(w)) / np.mean(w)

            turning = self.calculate_turning(body_angle)

            self.results[worm_id] = {
                'frame': worm_data['frame'].values,
                'head_x': head_x,
                'head_y': head_y,
                'tail_x': tail_x,
                'tail_y': tail_y,
                'body_x': body_x.values,
                'body_y': body_y.values,
                'speed': speed,
                'speed_smoothed': speed_smoothed,
                'body_length': body_length,
                'body_angle': body_angle,
                'motion_angle': motion_angle,
                'curvature': curvature,
                'contraction': contraction,
                'turning': turning,
                'w': w,
                'h': h
            }

    def calculate_curvature(self, worm_data):
        kpts = []
        for i in range(1, 6):
            kpts.append(np.column_stack((worm_data[f'x{i}'].values, worm_data[f'y{i}'].values)))

        curvatures = []
        for i in range(len(worm_data)):
            vectors = []
            for j in range(4):
                vec = kpts[j + 1][i] - kpts[j][i]
                vectors.append(vec)

            angles = []
            for j in range(3):
                v1 = vectors[j]
                v2 = vectors[j + 1]
                dot = np.dot(v1, v2)
                det = np.linalg.det(np.vstack([v1, v2]))
                angle = np.arctan2(det, dot) * 180 / np.pi
                angles.append(abs(angle))

            curvature = np.mean(angles)
            curvatures.append(curvature)

        return np.array(curvatures)

    def calculate_turning(self, body_angle):
        angle_diff = np.diff(body_angle, prepend=body_angle[0])
        angle_diff = (angle_diff + 180) % 360 - 180
        angular_velocity = angle_diff / self.dt
        return angular_velocity

    def analyze_behavior(self):
        for worm_id, data in self.results.items():
            speed = data['speed_smoothed']

            angle_diff = np.abs(np.diff(data['body_angle']))
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            flips = np.sum(angle_diff > 90)
            flip_frequency = flips / (len(data['frame']) * self.dt)

            pause_frames = np.sum(speed < self.PAUSE_THRESHOLD)
            pause_ratio = pause_frames / len(data['frame'])

            omega_turns = np.sum(data['curvature'] > self.OMEGA_ANGLE_THRESHOLD)

            angle_diff = np.abs(data['motion_angle'] - data['body_angle'])
            angle_diff = np.minimum(angle_diff, 360 - angle_diff)
            forward = np.sum(angle_diff < 90) / len(data['frame'])
            backward = 1 - forward

            coiling = np.sum(data['curvature'] > self.COILING_CURVATURE_THRESHOLD)
            coiling_ratio = coiling / len(data['frame'])

            pirouette = np.sum(np.abs(data['turning']) > self.PIROUETTE_ANGULAR_VELOCITY)
            pirouette_ratio = pirouette / len(data['frame'])

            curvature = data['curvature']
            peaks, _ = find_peaks(curvature, height=np.mean(curvature), distance=int(0.5 * self.fps))
            wave_frequency = len(peaks) / (len(data['frame']) * self.dt)

            self.results[worm_id]['analysis'] = {
                'mean_speed': np.mean(speed),
                'max_speed': np.max(speed),
                'flip_frequency': flip_frequency,
                'pause_ratio': pause_ratio,
                'omega_turns': omega_turns,
                'forward_ratio': forward,
                'backward_ratio': backward,
                'coiling_ratio': coiling_ratio,
                'pirouette_ratio': pirouette_ratio,
                'wave_frequency': wave_frequency
            }

              self.prepare_visualization_data(worm_id)

    def prepare_visualization_data(self, worm_id):
        data = self.results[worm_id]
  
        time = data['frame'] * self.dt

        trajectory = np.column_stack((data['head_x'], data['head_y']))

        speed = data['speed_smoothed']

        curvature = data['curvature']

        body_angle = data['body_angle']

        contraction = data['contraction']

        self.visualization_data[worm_id] = {
            'time': time,
            'trajectory': trajectory,
            'speed': speed,
            'curvature': curvature,
            'body_angle': body_angle,
            'contraction': contraction
        }

    def visualize_population_behavior(self, output_dir='outputs/population_analysis'):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        all_times = []
        all_trajectories = []
        all_speeds = []
        all_curvatures = []
        all_body_angles = []
        all_contractions = []
        all_analyses = []

        for worm_id, vis_data in self.visualization_data.items():
            all_times.append(vis_data['time'])
            all_trajectories.append(vis_data['trajectory'])
            all_speeds.append(vis_data['speed'])
            all_curvatures.append(vis_data['curvature'])
            all_body_angles.append(vis_data['body_angle'])
            all_contractions.append(vis_data['contraction'])
            all_analyses.append(self.results[worm_id]['analysis'])

        min_length = min(len(t) for t in all_times)
        aligned_times = np.array([t[:min_length] for t in all_times])
        aligned_speeds = np.array([s[:min_length] for s in all_speeds])
        aligned_curvatures = np.array([c[:min_length] for c in all_curvatures])
        aligned_body_angles = np.array([b[:min_length] for b in all_body_angles])
        aligned_contractions = np.array([c[:min_length] for c in all_contractions])

        mean_time = np.mean(aligned_times, axis=0)
        mean_speed = np.mean(aligned_speeds, axis=0)
        std_speed = np.std(aligned_speeds, axis=0)
        mean_curvature = np.mean(aligned_curvatures, axis=0)
        std_curvature = np.std(aligned_curvatures, axis=0)

        plt.figure(figsize=(18, 12))
        plt.suptitle('multi-worms', fontsize=16)

        plt.subplot(3, 3, 1)
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(all_trajectories))]

        for i, traj in enumerate(all_trajectories):
            plt.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], alpha=0.5, linewidth=1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{len(all_trajectories)}')
        plt.grid(True)

        plt.subplot(3, 3, 2)
        plt.fill_between(mean_time, mean_speed - std_speed, mean_speed + std_speed,
                        alpha=0.2, color='blue')
        plt.plot(mean_time, mean_speed, 'b-', linewidth=2, label='群体平均')

        for i, speed in enumerate(aligned_speeds):
            plt.plot(mean_time, speed, '-', color=colors[i], alpha=0.1)

        plt.xlabel('time')
        plt.ylabel('speed')
        plt.title('spped-v')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 3, 3)
        plt.fill_between(mean_time, mean_curvature - std_curvature, mean_curvature + std_curvature,
                        alpha=0.2, color='green')
        plt.plot(mean_time, mean_curvature, 'g-', linewidth=2, label='群体平均')

        for i, curv in enumerate(aligned_curvatures):
            plt.plot(mean_time, curv, '-', color=colors[i], alpha=0.1)

        plt.xlabel('s')
        plt.ylabel('cur')
        plt.title('cur-v')
        plt.grid(True)
        plt.legend()


        plt.subplot(3, 3, 4)
        for i, angle in enumerate(aligned_body_angles):
            plt.plot(mean_time, angle, '-', color=colors[i], alpha=0.3)

        plt.xlabel('time')
        plt.ylabel('degree')
        plt.title('direction')
        plt.grid(True)


        plt.subplot(3, 3, 5)
        mean_contraction = np.mean(aligned_contractions, axis=0)
        std_contraction = np.std(aligned_contractions, axis=0)

        plt.fill_between(mean_time, mean_contraction - std_contraction, mean_contraction + std_contraction,
                        alpha=0.2, color='purple')
        plt.plot(mean_time, mean_contraction, 'm-', linewidth=2, label='av')

        plt.xlabel('time')
        plt.ylabel('radio')
        plt.title('radio-v')
        plt.grid(True)
        plt.legend()

 
        plt.subplot(3, 3, 6)
        behavior_metrics = {
            'spped': [a['mean_speed'] for a in all_analyses],
            'freq': [a['flip_frequency'] for a in all_analyses],
            'pause': [a['pause_ratio'] for a in all_analyses],
            'forward': [a['forward_ratio'] for a in all_analyses],
            'cur': [a['coiling_ratio'] for a in all_analyses]
        }

        plt.boxplot(behavior_metrics.values(), labels=behavior_metrics.keys())
        plt.xticks(rotation=45)
        plt.title('behacior')
        plt.ylabel('value')
        plt.grid(True)

  
        plt.subplot(3, 3, 7)
        for i in range(len(aligned_speeds)):
            plt.scatter(aligned_speeds[i], aligned_curvatures[i], color=colors[i], alpha=0.3, s=10)

        plt.xlabel('speed')
        plt.ylabel('cur')
        plt.title('spped-curv')
        plt.grid(True)

  
        plt.subplot(3, 3, 8)
        all_points = np.vstack(all_trajectories)
        plt.hist2d(all_points[:, 0], all_points[:, 1], bins=50, cmap='hot')
        plt.colorbar(label='freq')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('heatmap')

   
        plt.subplot(3, 3, 9)
        behavior_time = {
            'forward': np.mean([a['forward_ratio'] for a in all_analyses]),
            'back': np.mean([a['backward_ratio'] for a in all_analyses]),
            'pause': np.mean([a['pause_ratio'] for a in all_analyses]),
            'cur': np.mean([a['coiling_ratio'] for a in all_analyses]),
            'xz': np.mean([a['pirouette_ratio'] for a in all_analyses])
        }

        plt.pie(behavior_time.values(), labels=behavior_time.keys(), autopct='%1.1f%%')
        plt.title('time-radio')

        plt.tight_layout()

     
        output_path = os.path.join(output_dir, 'population_behavior_analysis.png')
        plt.savefig(output_path, dpi=150)
        plt.close()

   

    def create_population_animation(self, output_dir='outputs/population_analysis'):
       
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig = plt.figure(figsize=(15, 8))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)  
        ax2 = plt.subplot2grid((2, 2), (1, 0)) 
        ax3 = plt.subplot2grid((2, 2), (1, 1))  

  
        all_trajectories = [v['trajectory'] for v in self.visualization_data.values()]
        all_x = np.concatenate([t[:, 0] for t in all_trajectories])
        all_y = np.concatenate([t[:, 1] for t in all_trajectories])

        ax1.set_xlim(np.min(all_x) - 50, np.max(all_x) + 50)
        ax1.set_ylim(np.min(all_y) - 50, np.max(all_y) + 50)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('tail')

     
        max_frame = max(len(v['time']) for v in self.visualization_data.values())
        time_axis = np.arange(max_frame) * self.dt

        ax2.set_xlim(0, max_frame * self.dt)
        ax2.set_ylim(0, max([np.max(v['speed']) for v in self.visualization_data.values()]) * 1.1)
        ax2.set_xlabel('s')
        ax2.set_ylabel('v')
        ax2.set_title('spped')

        ax3.set_xlim(0, max_frame * self.dt)
        ax3.set_ylim(0, max([np.max(v['curvature']) for v in self.visualization_data.values()]) * 1.1)
        ax3.set_xlabel('t')
        ax3.set_ylabel('c')
        ax3.set_title('c-radio')

    
        cmap = plt.get_cmap('viridis')
        colors = [cmap(i) for i in np.linspace(0, 1, len(self.visualization_data))]

        traj_lines = []
        head_points = []
        speed_lines = []
        curv_lines = []

        for i, (worm_id, data) in enumerate(self.visualization_data.items()):
         
            traj_line, = ax1.plot([], [], '-', color=colors[i], alpha=0.7, linewidth=1)
            head_point, = ax1.plot([], [], 'o', color=colors[i], markersize=6)
            traj_lines.append(traj_line)
            head_points.append(head_point)

          
            speed_line, = ax2.plot([], [], '-', color=colors[i], alpha=0.7)
            speed_lines.append(speed_line)

         
            curv_line, = ax3.plot([], [], '-', color=colors[i], alpha=0.7)
            curv_lines.append(curv_line)

       
        def update(frame):
            for i, (worm_id, data) in enumerate(self.visualization_data.items()):
                if frame < len(data['time']):
                   
                    traj_lines[i].set_data(data['trajectory'][:frame + 1, 0], data['trajectory'][:frame + 1, 1])
                    head_points[i].set_data(data['trajectory'][frame, 0], data['trajectory'][frame, 1])

                  
                    speed_lines[i].set_data(data['time'][:frame + 1], data['speed'][:frame + 1])

                
                    curv_lines[i].set_data(data['time'][:frame + 1], data['curvature'][:frame + 1])

            return traj_lines + head_points + speed_lines + curv_lines

    
        ani = animation.FuncAnimation(
            fig, update, frames=max_frame,
            interval=1000 / self.fps, blit=True
        )

      
        output_path = os.path.join(output_dir, 'population_behavior_animation.mp4')
        ani.save(output_path, writer='ffmpeg', fps=self.fps)
        plt.close()

  

 

# 使用示例
if __name__ == "__main__":
    
    analyzer = WormBehaviorAnalyzer('outputs/', fps=2)




