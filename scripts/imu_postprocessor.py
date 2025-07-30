#!/usr/bin/env python3
"""
高精度IMU加速度后处理算法

基于Isaac Lab IMU传感器数据，使用Savitzky-Golay滤波器提高加速度计算精度。
该脚本从CSV文件读取线速度数据，通过高阶数值微分计算高精度加速度，
并可配置地降采样到指定频率。
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import warnings


class IMUAccelerationPostProcessor:
    """
    IMU加速度后处理器
    
    使用Savitzky-Golay滤波器对线速度数据进行平滑和微分，
    生成高精度的加速度数据，并可降采样到指定频率。
    """
    
    def __init__(self, 
                 sg_window: int = 51, 
                 sg_polyorder: int = 3, 
                 cutoff_freq_ratio: float = 0.4,
                 butter_order: int = 4):
        """
        初始化后处理器
        
        Args:
            sg_window: Savitzky-Golay滤波器窗口大小（必须为奇数）
            sg_polyorder: Savitzky-Golay滤波器多项式阶数
            cutoff_freq_ratio: 抗混叠滤波器截止频率相对于目标奈奎斯特频率的比率
            butter_order: Butterworth滤波器阶数
        """
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.cutoff_freq_ratio = cutoff_freq_ratio
        self.butter_order = butter_order
        
        # 验证参数
        if sg_window % 2 == 0:
            raise ValueError("Savitzky-Golay窗口大小必须为奇数")
        if sg_polyorder >= sg_window:
            raise ValueError("多项式阶数必须小于窗口大小")
            
    def load_data(self, csv_path: str) -> tuple:
        """
        从CSV文件加载IMU数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            tuple: (timestamps, lin_vel_data, quat_data, sampling_frequency, dt, df, velocity_cols, quat_cols)
        """
        print(f"正在加载数据: {csv_path}")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"无法读取CSV文件: {e}")
            
        # 检查必需的列
        required_cols = ['timestamp']
        velocity_cols = []
        
        # 查找速度列（支持多种命名方式）
        possible_vel_names = [
            ['lin_vel_0', 'lin_vel_1', 'lin_vel_2'],
            ['vx', 'vy', 'vz'],
            ['vel_x', 'vel_y', 'vel_z'],
            ['lin_vel_b_x', 'lin_vel_b_y', 'lin_vel_b_z']
        ]
        
        for vel_set in possible_vel_names:
            if all(col in df.columns for col in vel_set):
                velocity_cols = vel_set
                break
                
        if not velocity_cols:
            available_cols = ', '.join(df.columns)
            raise ValueError(f"未找到速度数据列。可用列: {available_cols}")
            
        print(f"使用速度列: {velocity_cols}")
        
        # 查找四元数列（用于重力补偿）
        possible_quat_names = [
            ['quat_0', 'quat_1', 'quat_2', 'quat_3'],  # w, x, y, z
            ['qw', 'qx', 'qy', 'qz'],
            ['quat_w', 'quat_x', 'quat_y', 'quat_z']
        ]
        
        quat_cols = []
        for quat_set in possible_quat_names:
            if all(col in df.columns for col in quat_set):
                quat_cols = quat_set
                break
        
        if not quat_cols:
            print("警告: 未找到四元数数据，将跳过重力补偿")
            quat_data = None
        else:
            print(f"使用四元数列: {quat_cols}")
            quat_data = df[quat_cols].values
        
        # 提取数据
        timestamps = df['timestamp'].values
        lin_vel_data = df[velocity_cols].values
        
        # 计算采样频率
        dt_values = np.diff(timestamps)
        dt_mean = np.mean(dt_values)
        dt_std = np.std(dt_values)
        sampling_freq = 1.0 / dt_mean
        
        print(f"数据点数: {len(timestamps)}")
        print(f"时间跨度: {timestamps[-1] - timestamps[0]:.3f} 秒")
        print(f"平均时间步长: {dt_mean:.6f} 秒")
        print(f"时间步长标准差: {dt_std:.6f} 秒")
        print(f"计算出的采样频率: {sampling_freq:.1f} Hz")
        
        # 检查时间步长的一致性
        if dt_std / dt_mean > 0.01:  # 1%的变化容忍度
            warnings.warn(f"时间步长变化较大 (std/mean = {dt_std/dt_mean:.3f}), 可能影响处理质量")
            
        return timestamps, lin_vel_data, quat_data, sampling_freq, dt_mean, df, velocity_cols, quat_cols
    
    def apply_savgol_filter(self, lin_vel_data: np.ndarray, dt: float) -> np.ndarray:
        """
        应用Savitzky-Golay滤波器计算高精度加速度
        
        Args:
            lin_vel_data: 线速度数据 (N, 3)
            dt: 时间步长
            
        Returns:
            np.ndarray: 高精度加速度数据 (N, 3)
        """
        print(f"应用Savitzky-Golay滤波器 (窗口={self.sg_window}, 多项式阶数={self.sg_polyorder})")
        
        # 检查数据长度是否足够
        if len(lin_vel_data) < self.sg_window:
            raise ValueError(f"数据长度 ({len(lin_vel_data)}) 小于滤波器窗口大小 ({self.sg_window})")
        
        # 对每个轴独立应用滤波器
        lin_acc_filtered = np.zeros_like(lin_vel_data)
        
        for axis in range(3):
            lin_acc_filtered[:, axis] = savgol_filter(
                lin_vel_data[:, axis], 
                self.sg_window, 
                self.sg_polyorder, 
                deriv=1,  # 计算一阶导数
                delta=dt  # 时间步长
            )
            
        return lin_acc_filtered
    
    def add_gravity_compensation(self, lin_acc_data: np.ndarray, quat_data: np.ndarray = None) -> np.ndarray:
        """
        为计算出的加速度添加重力补偿，使IMU在静止时显示[0,0,9.81]
        
        Args:
            lin_acc_data: 运动加速度数据 (N, 3)
            quat_data: 四元数数据 (N, 4)，格式为[w,x,y,z]，如果为None则假设无旋转
            
        Returns:
            np.ndarray: 包含重力补偿的IMU加速度数据 (N, 3)
        """
        print("添加重力补偿...")
        
        # 世界坐标系下的重力向量 [0, 0, -9.81] (向下)
        gravity_world = np.array([0.0, 0.0, -9.81])
        
        if quat_data is None:
            # 如果没有四元数数据，假设IMU始终水平放置（无旋转）
            # 在这种情况下，重力在IMU坐标系中就是 [0, 0, 9.81]（向上，因为IMU测量的是对重力的反作用力）
            print("未提供四元数数据，假设IMU无旋转")
            gravity_imu = np.array([0.0, 0.0, 9.81])  # IMU测量的是支撑力，与重力方向相反
            gravity_compensation = np.tile(gravity_imu, (len(lin_acc_data), 1))
        else:
            # 使用四元数将世界坐标系的重力转换到IMU坐标系
            print("使用四元数进行重力坐标系转换")
            
            # 检查四元数格式并转换为scipy格式 [x,y,z,w]
            if quat_data.shape[1] == 4:
                # 假设输入格式为 [w,x,y,z]，转换为 [x,y,z,w]
                quat_scipy = quat_data[:, [1, 2, 3, 0]]
            else:
                raise ValueError("四元数数据格式错误，应为 (N, 4)")
            
            # 创建旋转对象
            rotations = R.from_quat(quat_scipy)
            
            # 将世界坐标系的重力向量转换到IMU坐标系
            # 使用逆旋转，因为我们要从世界坐标系转到IMU坐标系
            gravity_imu_per_sample = rotations.apply(gravity_world, inverse=True)
            
            # IMU测量的是对重力的反作用力，所以要取负号
            gravity_compensation = -gravity_imu_per_sample
        
        # 将运动加速度与重力补偿相加
        imu_acceleration = lin_acc_data + gravity_compensation
        
        return imu_acceleration
    
    def decimate_data(self, data: np.ndarray, timestamps: np.ndarray,
                     original_freq: float, target_freq: float) -> tuple:
        """
        将数据降采样到目标频率
        
        Args:
            data: 要降采样的数据 (N, 3)
            timestamps: 时间戳 (N,)
            original_freq: 原始采样频率
            target_freq: 目标采样频率
            
        Returns:
            tuple: (降采样后的数据, 降采样后的时间戳)
        """
        if target_freq >= original_freq:
            print(f"目标频率 ({target_freq} Hz) 大于等于原始频率 ({original_freq:.1f} Hz)，不进行降采样")
            return data, timestamps
            
        decimation_factor = int(round(original_freq / target_freq))
        actual_target_freq = original_freq / decimation_factor
        
        print(f"降采样: {original_freq:.1f} Hz -> {actual_target_freq:.1f} Hz (因子: {decimation_factor})")
        
        # 设计抗混叠滤波器
        nyquist_freq = target_freq / 2
        cutoff_freq = nyquist_freq * self.cutoff_freq_ratio
        
        print(f"抗混叠滤波器: 截止频率 = {cutoff_freq:.1f} Hz")
        
        # 设计Butterworth低通滤波器
        b, a = butter(self.butter_order, cutoff_freq, fs=original_freq, btype='low')
        
        # 应用零相位滤波器
        data_filtered = np.zeros_like(data)
        for axis in range(3):
            data_filtered[:, axis] = filtfilt(b, a, data[:, axis])
        
        # 执行降采样
        decimated_data = data_filtered[::decimation_factor]
        decimated_timestamps = timestamps[::decimation_factor]
        
        print(f"降采样后数据点数: {len(decimated_data)}")
        
        return decimated_data, decimated_timestamps
    
    def process(self, csv_path: str, target_freq: float = None, add_gravity: bool = True) -> dict:
        """
        完整的处理流程
        
        Args:
            csv_path: 输入CSV文件路径
            target_freq: 目标输出频率 (Hz)，如果为None则不降采样
            add_gravity: 是否添加重力补偿
            
        Returns:
            dict: 包含处理结果的字典
        """
        # 1. 数据加载与频率计算
        timestamps, lin_vel_data, quat_data, original_freq, dt, original_df, velocity_cols, quat_cols = self.load_data(csv_path)
        
        # 2. 高精度加速度计算 (Savitzky-Golay)
        lin_acc_filtered = self.apply_savgol_filter(lin_vel_data, dt)
        
        # 3. 添加重力补偿（如果需要）
        if add_gravity:
            lin_acc_final = self.add_gravity_compensation(lin_acc_filtered, quat_data)
        else:
            lin_acc_final = lin_acc_filtered
        
        # 4. 降采样（如果需要）
        if target_freq is not None:
            final_acc, final_timestamps = self.decimate_data(
                lin_acc_final, timestamps, original_freq, target_freq)
            # 降采样四元数数据（如果存在）
            if quat_data is not None:
                quat_decimated, _ = self.decimate_data(
                    quat_data, timestamps, original_freq, target_freq)
            else:
                quat_decimated = None
        else:
            final_acc = lin_acc_final
            final_timestamps = timestamps
            quat_decimated = quat_data
        
        return {
            'final_timestamps': final_timestamps,
            'final_acceleration': final_acc,
            'final_quaternion': quat_decimated,
            'original_freq': original_freq,
            'target_freq': target_freq or original_freq,
            'gravity_compensated': add_gravity,
            'original_df': original_df,
            'velocity_cols': velocity_cols,
            'quat_cols': quat_cols
        }
    
    def save_results(self, results: dict, output_path: str, original_df: pd.DataFrame,
                    velocity_cols: list, quat_cols: list = None):
        """
        保存处理结果到CSV文件，保留原始数据的所有列
        
        Args:
            results: 处理结果字典
            output_path: 输出文件路径
            original_df: 原始数据DataFrame
            velocity_cols: 速度列名列表
            quat_cols: 四元数列名列表（可选）
        """
        # 获取降采样的索引
        original_freq = results['original_freq']
        target_freq = results['target_freq']
        
        if target_freq >= original_freq:
            # 没有降采样，使用所有数据
            decimated_indices = slice(None)
        else:
            # 计算降采样因子和索引
            decimation_factor = int(round(original_freq / target_freq))
            decimated_indices = slice(None, None, decimation_factor)
        
        # 从原始DataFrame中提取降采样后的数据
        output_df = original_df.iloc[decimated_indices].copy()
        
        # 更新时间戳
        output_df['timestamp'] = results['final_timestamps']
        
        # 更新加速度列
        # 查找原始数据中的加速度列
        possible_acc_names = [
            ['lin_acc_0', 'lin_acc_1', 'lin_acc_2'],
            ['ax', 'ay', 'az'],
            ['acc_x', 'acc_y', 'acc_z'],
            ['lin_acc_b_x', 'lin_acc_b_y', 'lin_acc_b_z']
        ]
        
        acc_cols = []
        for acc_set in possible_acc_names:
            if all(col in output_df.columns for col in acc_set):
                acc_cols = acc_set
                break
        
        if acc_cols:
            # 如果原始数据中有加速度列，就更新它们
            print(f"更新原始加速度列: {acc_cols}")
            output_df[acc_cols[0]] = results['final_acceleration'][:, 0]
            output_df[acc_cols[1]] = results['final_acceleration'][:, 1]
            output_df[acc_cols[2]] = results['final_acceleration'][:, 2]
        else:
            # 如果没有加速度列，就添加新的加速度列
            print("添加新的加速度列")
            output_df['lin_acc_0'] = results['final_acceleration'][:, 0]
            output_df['lin_acc_1'] = results['final_acceleration'][:, 1]
            output_df['lin_acc_2'] = results['final_acceleration'][:, 2]
        
        # 更新四元数列（如果存在）
        if quat_cols and results['final_quaternion'] is not None:
            print(f"更新四元数列: {quat_cols}")
            output_df[quat_cols[0]] = results['final_quaternion'][:, 0]
            output_df[quat_cols[1]] = results['final_quaternion'][:, 1]
            output_df[quat_cols[2]] = results['final_quaternion'][:, 2]
            output_df[quat_cols[3]] = results['final_quaternion'][:, 3]
        
        # 保存到文件
        output_df.to_csv(output_path, index=False)
        
        print(f"结果已保存到: {output_path}")
        print(f"输出数据频率: {results['target_freq']:.1f} Hz")
        print(f"输出数据点数: {len(output_df)}")
        print(f"保留的数据列数: {len(output_df.columns)}")
        if results['gravity_compensated']:
            print("已包含重力补偿")
        else:
            print("未包含重力补偿")
    
    def plot_acceleration(self, results: dict, save_path: str = None):
        """
        绘制处理后的加速度数据
        
        Args:
            results: 处理结果字典
            save_path: 图片保存路径（可选）
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # 时间轴（只显示前5秒的数据以便清晰观察）
        max_time = 5.0
        final_mask = results['final_timestamps'] <= (results['final_timestamps'][0] + max_time)
        final_t = results['final_timestamps'][final_mask] - results['final_timestamps'][0]
        
        axis_names = ['X', 'Y', 'Z']
        colors = ['red', 'blue', 'green']
        
        title_suffix = '(含重力补偿)' if results['gravity_compensated'] else '(纯运动加速度)'
        
        for i in range(3):
            ax = axes[i]
            
            # 处理后的加速度
            ax.plot(final_t, results['final_acceleration'][final_mask, i],
                   color=colors[i], linewidth=2,
                   label=f'Savitzky-Golay滤波后加速度')
            
            ax.set_ylabel(f'加速度 {axis_names[i]} (m/s²)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_title(f'{axis_names[i]}轴加速度 {title_suffix}')
        
        axes[-1].set_xlabel('时间 (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"加速度图已保存到: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='IMU加速度后处理器 - 使用Savitzky-Golay滤波器提高加速度精度',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('input_csv', help='输入CSV文件路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径，默认为输入文件名加"_processed"后缀')
    parser.add_argument('--freq', type=float, default=200,
                       help='目标输出频率 (Hz)')
    parser.add_argument('--sg_window', type=int, default=51,
                       help='Savitzky-Golay滤波器窗口大小（必须为奇数）')
    parser.add_argument('--sg_polyorder', type=int, default=3,
                       help='Savitzky-Golay滤波器多项式阶数')
    parser.add_argument('--cutoff_ratio', type=float, default=0.4,
                       help='抗混叠滤波器截止频率比率')
    parser.add_argument('--butter_order', type=int, default=4,
                       help='Butterworth滤波器阶数')
    parser.add_argument('--plot', action='store_true',
                       help='显示加速度图')
    parser.add_argument('--plot_save', type=str,
                       help='保存加速度图的文件路径')
    parser.add_argument('--no_gravity', action='store_true',
                       help='不添加重力补偿（默认会添加重力补偿）')
    
    args = parser.parse_args()
    
    # 验证输入文件
    if not Path(args.input_csv).exists():
        print(f"错误: 输入文件不存在: {args.input_csv}")
        return 1
    
    if args.output is None:
        # 默认输出路径为输入文件名加"_processed"后缀
        output_path = Path(args.input_csv).with_name(
            Path(args.input_csv).stem + "_processed.csv"
        )
    else:
        output_path = args.output
    
    try:
        # 创建处理器
        processor = IMUAccelerationPostProcessor(
            sg_window=args.sg_window,
            sg_polyorder=args.sg_polyorder,
            cutoff_freq_ratio=args.cutoff_ratio,
            butter_order=args.butter_order
        )
        
        # 执行处理
        print("=" * 60)
        print("IMU加速度后处理开始")
        print("=" * 60)
        
        results = processor.process(args.input_csv, args.freq, add_gravity=not args.no_gravity)
        
        # 保存结果
        processor.save_results(results, output_path, results['original_df'],
                             results['velocity_cols'], results['quat_cols'])
        
        # 绘制加速度图
        if args.plot or args.plot_save:
            processor.plot_acceleration(results, args.plot_save)
        
        print("=" * 60)
        print("处理完成!")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    exit(main())