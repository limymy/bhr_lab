# IMU加速度后处理器使用指南

## 概述

`imu_postprocessor.py` 是一个专门设计用于提高Isaac Lab IMU传感器加速度精度的后处理工具。该工具解决了Isaac Lab中通过简单后向差分计算加速度时精度不足的问题，通过使用Savitzky-Golay滤波器实现高精度的数值微分，并且支持重力补偿功能，使IMU在静止时显示真实的重力加速度值[0,0,9.81]。

## 背景

Isaac Lab中的IMU传感器使用以下方式计算线加速度：
```python
lin_acc_w = (lin_vel_w - self._prev_lin_vel_w) / dt
```

这种简单的后向差分方法在离散时间步长下会引入显著的数值噪声，特别是在高频采样时。本工具通过以下方法改善精度：

1. **Savitzky-Golay滤波器**: 在局部窗口内拟合多项式并计算其导数
2. **重力补偿**: 添加重力加速度影响，使IMU在静止时显示[0,0,9.81]
3. **抗混叠滤波**: 在降采样前应用低通滤波器
4. **可配置参数**: 允许根据不同应用场景调整处理参数

## 安装依赖

```bash
pip install numpy pandas scipy matplotlib
```

## 基本使用

### 1. 最简单的使用方式

```bash
python scripts/imu_postprocessor.py input_data.csv --output output_data.csv
```

这将使用默认参数处理数据，输出频率为200Hz，并包含重力补偿。

### 2. 指定目标频率

```bash
python scripts/imu_postprocessor.py input_data.csv --target_freq 100
```

### 3. 不使用重力补偿

```bash
python scripts/imu_postprocessor.py input_data.csv --no_gravity
```

### 4. 显示加速度图

```bash
python scripts/imu_postprocessor.py input_data.csv --plot
```

### 5. 保存加速度图

```bash
python scripts/imu_postprocessor.py input_data.csv --plot_save acceleration.png
```

## 输入数据格式

CSV文件必须包含以下列：

### 必需列
- `timestamp`: 时间戳（秒）

### 速度列（以下任意一组）
- `lin_vel_0`, `lin_vel_1`, `lin_vel_2`
- `vx`, `vy`, `vz`
- `vel_x`, `vel_y`, `vel_z`
- `lin_vel_b_x`, `lin_vel_b_y`, `lin_vel_b_z`

### 四元数列（可选，用于重力补偿）
- `quat_0`, `quat_1`, `quat_2`, `quat_3` (w, x, y, z)
- `qw`, `qx`, `qy`, `qz`
- `quat_w`, `quat_x`, `quat_y`, `quat_z`

### 示例CSV格式
```csv
timestamp,lin_vel_0,lin_vel_1,lin_vel_2,quat_0,quat_1,quat_2,quat_3
0.000000,-0.001234,0.005678,-0.000234,1.0,0.0,0.0,0.0
0.001000,-0.001456,0.005432,-0.000267,0.999,0.001,0.002,0.001
0.002000,-0.001678,0.005187,-0.000301,0.998,0.002,0.003,0.002
...
```

## 输出数据格式

输出CSV文件**保留输入文件的所有列**，并进行以下修改：
- `timestamp`: 根据目标频率进行降采样
- 加速度列: 如果输入中存在加速度列，将被高精度的Savitzky-Golay滤波结果替换
- 如果输入中没有加速度列，将添加新的 `lin_acc_0`, `lin_acc_1`, `lin_acc_2` 列
- 四元数列: 如果存在，将同步进行降采样
- **所有其他数据列**: 完全保留并同步降采样

**重要特性**: 输出文件保持与输入文件相同的数据结构，只修正加速度值和调整采样频率，其他所有传感器数据和元信息都会被保留。

## 命令行参数详解

### 必需参数
- `input_csv`: 输入CSV文件路径

### 可选参数

#### 输出控制
- `--output PATH`: 输出CSV文件路径（默认为输入文件名加"_processed"后缀）

#### 频率控制
- `--target_freq FREQ` (默认: 200): 目标输出频率 (Hz)

#### 重力补偿
- `--no_gravity`: 不添加重力补偿（默认会添加重力补偿）

#### Savitzky-Golay滤波器参数
- `--sg_window WINDOW` (默认: 51): 滤波器窗口大小（必须为奇数）
- `--sg_polyorder ORDER` (默认: 3): 多项式阶数

#### 降采样参数
- `--cutoff_ratio RATIO` (默认: 0.4): 抗混叠滤波器截止频率比率
- `--butter_order ORDER` (默认: 4): Butterworth滤波器阶数

#### 可视化参数
- `--plot`: 显示加速度图
- `--plot_save PATH`: 保存加速度图到指定路径

## 参数调优指南

### Savitzky-Golay窗口大小 (sg_window)

**作用**: 控制平滑程度和响应性的平衡
- **较小值** (15-31): 保留更多高频细节，但噪声抑制较弱
- **中等值** (31-71): 平衡的选择，适合大多数情况
- **较大值** (71-101): 更强的平滑效果，但可能过度平滑真实信号

**推荐设置**:
- 高噪声数据: 71-101
- 一般用途: 51 (默认)
- 保留细节: 31-51

### 多项式阶数 (sg_polyorder)

**作用**: 控制局部拟合的精度
- **2**: 二次多项式，适合平滑运动
- **3**: 三次多项式，适合一般情况 (默认)
- **4-5**: 高阶多项式，保留更多细节但对噪声敏感

### 截止频率比率 (cutoff_ratio)

**作用**: 控制抗混叠滤波器的截止频率
- **0.3-0.4**: 保守设置，强抗混叠 (推荐)
- **0.4-0.5**: 平衡设置
- **0.5+**: 激进设置，可能引入混叠

## 使用示例

### 示例1: 标准处理流程（含重力补偿）
```bash
# 处理1kHz数据，输出200Hz，包含重力补偿
python scripts/imu_postprocessor.py \
    data/imu_1khz.csv \
    --output results/imu_200hz_processed.csv \
    --target_freq 200 \
    --plot_save results/acceleration.png
```

### 示例2: 纯运动加速度处理
```bash
# 处理数据但不包含重力补偿（纯运动加速度）
python scripts/imu_postprocessor.py \
    data/imu_1khz.csv \
    --output results/motion_only.csv \
    --no_gravity \
    --target_freq 200
```

### 示例3: 高精度处理
```bash
# 使用较大窗口以获得更平滑的结果
python scripts/imu_postprocessor.py \
    data/noisy_imu.csv \
    --output results/smooth_imu.csv \
    --sg_window 71 \
    --sg_polyorder 3 \
    --target_freq 100
```

### 示例4: 保留细节的处理
```bash
# 使用较小窗口保留更多动态响应
python scripts/imu_postprocessor.py \
    data/dynamic_motion.csv \
    --output results/detailed_motion.csv \
    --sg_window 31 \
    --sg_polyorder 4 \
    --target_freq 500
```

## 输出解释

### 控制台输出示例
```
============================================================
IMU加速度后处理开始
============================================================
正在加载数据: data/imu_1khz.csv
使用速度列: ['lin_vel_x', 'lin_vel_y', 'lin_vel_z']
数据点数: 10000
时间跨度: 10.000 秒
平均时间步长: 0.001000 秒
计算出的采样频率: 1000.0 Hz
应用Savitzky-Golay滤波器 (窗口=51, 多项式阶数=3)
降采样: 1000.0 Hz -> 200.0 Hz (因子: 5)
抗混叠滤波器: 截止频率 = 40.0 Hz
降采样后数据点数: 2000
结果已保存到: results/imu_200hz_processed.csv
输出数据频率: 200.0 Hz
输出数据点数: 2000
已包含重力补偿
============================================================
处理完成!
============================================================
```

### 对比图说明

生成的对比图显示：
- **红线**: 原始后向差分加速度
- **橙线**: Savitzky-Golay滤波后的加速度

预期看到：
- 橙线明显比红线更平滑
- 高频噪声被显著抑制
- 主要运动趋势得到保留
- 如果包含重力补偿，Z轴在静止时应显示约9.81的值

## 质量评估

### 成功处理的指标
1. **噪声降低**: 高频噪声明显减少
2. **信号保真度**: 主要运动特征保持不变
3. **物理合理性**: 加速度值在合理范围内
4. **频率响应**: 没有引入不自然的振荡

### 常见问题及解决方案

#### 问题1: 过度平滑
**现象**: 加速度信号过于平滑，丢失重要动态
**解决**: 减小`sg_window`或增加`sg_polyorder`

#### 问题2: 噪声仍然明显
**现象**: 处理后仍有明显高频噪声
**解决**: 增大`sg_window`或减小`cutoff_ratio`

#### 问题3: 引入伪影
**现象**: 出现不自然的振荡或尖峰
**解决**: 调整窗口大小，确保为奇数且适合数据特性

## 性能考虑

- **内存使用**: 与数据长度线性相关
- **计算时间**: 主要由Savitzky-Golay滤波器决定，O(N*window)
- **建议数据长度**: < 1M点 for 实时处理

## 技术细节

### Savitzky-Golay滤波器原理
该滤波器在每个数据点周围的窗口内拟合一个多项式，然后使用该多项式的导数作为该点的导数估计。这等效于应用一个特定的FIR滤波器。

### 重力补偿原理
该工具通过以下步骤实现重力补偿：
1. 定义世界坐标系下的重力向量：[0, 0, -9.81] m/s²
2. 如果提供了四元数数据，使用四元数将重力向量转换到IMU坐标系
3. 如果没有四元数数据，假设IMU水平放置，重力在IMU坐标系中为[0, 0, 9.81]
4. 将运动加速度与重力补偿相加，得到IMU应该测量的总加速度

### 抗混叠滤波设计
使用Butterworth低通滤波器，其截止频率设置为：
```
f_cutoff = (target_freq / 2) * cutoff_ratio
```

这确保了在降采样时不会发生频率混叠。

## 依赖库要求

确保安装以下Python库：
```bash
pip install numpy pandas scipy matplotlib
```

## 许可证

此工具遵循Isaac Lab项目的BSD-3-Clause许可证。