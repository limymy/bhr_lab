import torch
from collections.abc import Sequence

from isaaclab.managers.recorder_manager import (
    RecorderTerm,               # 基类
    RecorderTermCfg,            # 单个 Recorder term 的配置
)
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.sensors import Camera, ContactSensor, Imu

from isaaclab.envs import ManagerBasedEnv

class MyRecorderTerm(RecorderTerm):
    """
    基类，所有自定义的 Recorder Term 都应该继承这个类。
    这里可以添加一些通用的初始化逻辑或方法。
    """
    def __init__(self, cfg: RecorderTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._dt = env.physics_dt
        self._timestamp = torch.zeros(self._env.num_envs, device=self._env.device)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        """
        在每次环境重置后调用，重置时间戳。
        """
        if env_ids is not None:
            self._timestamp[env_ids] = 0.0
        else:
            self._timestamp = torch.zeros(self._env.num_envs, device=self._env.device)
        return None, None
    
    def update_timestamp(self):
        """
        更新时间戳，通常在每个物理步骤后调用。
        """
        self._timestamp += self._dt

#  关节状态 Recorder
class JointStateRecorder(MyRecorderTerm):
    """
    一个自定义的Recorder Term，用于记录机器人关节的位置和速度。支持自定义记录周期。
    """
    def __init__(self, cfg: "JointStateRecorderCfg", env):
        super().__init__(cfg, env)
        self._robot = env.scene["robot"]
        self._record_period = cfg.record_period if cfg.record_period is not None else self._dt
        self._next_update_time = torch.full_like(self._timestamp, self._record_period)

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, torch.Tensor | dict | None]:
        super().record_post_reset(env_ids)
        if env_ids is not None:
            self._next_update_time[env_ids] = self._record_period
        else:
            self._next_update_time = torch.full_like(self._timestamp, self._record_period)
        return None, None

    def record_post_physics_step(self) -> tuple[str | None, dict[str, torch.Tensor] | None]:
        self.update_timestamp()
        
        if self._timestamp[0] < self._next_update_time[0]:
            return None, None
        
        while self._timestamp[0] >= self._next_update_time[0]:
            self._next_update_time += self._record_period
        
        data_dict = {
            "timestamp": self._timestamp.clone().cpu(),
            "joint_pos": self._robot.data.joint_pos.clone().cpu(),
            "joint_vel": self._robot.data.joint_vel.clone().cpu(),
        }
        return "joint_state", data_dict

@configclass
class JointStateRecorderCfg(RecorderTermCfg):
    class_type: type = JointStateRecorder
    record_period: float | None = None
    """记录周期，单位为秒。如果为 None，则使用环境的物理步长。默认值为 None。"""

# IMU Recorder
class ImuRecorder(MyRecorderTerm):
    """
    每次 env.step() 内部的每个 physics 子步结束后，检测imu是否更新，并将新值记录。
    """
    def __init__(self, cfg: "ImuRecorderCfg", env):
        super().__init__(cfg, env)
        if cfg.imu_name is None:
            raise ValueError("The 'imu_name' must be provided for the ImuRecorder.")
        self._imu_name = cfg.imu_name
        self._imu: Imu = env.scene.sensors[cfg.imu_name]
        self._imu_dt = self._imu.cfg.update_period
        self._next_update_time = torch.full_like(self._timestamp, self._imu_dt)

    def record_post_reset(self, env_ids):
        super().record_post_reset(env_ids)
        if env_ids is not None:
            self._next_update_time[env_ids] = self._imu_dt
        else:
            self._next_update_time = torch.full_like(self._timestamp, self._imu_dt)
        return None, None

    def record_post_physics_step(self):
        self.update_timestamp()

        if self._timestamp[0] < self._next_update_time[0]:
            # 未到imu更新周期
            return None, None
        
        while self._timestamp[0] >= self._next_update_time[0]:
            self._next_update_time += self._imu_dt
        
        data_dict = {
            "timestamp": self._timestamp.clone().cpu(),
            "pos": self._imu.data.pos_w.clone().cpu(),
            "quat": self._imu.data.quat_w.clone().cpu(),
            "ang_vel": self._imu.data.ang_vel_b.clone().cpu(),
            "lin_vel": self._imu.data.lin_vel_b.clone().cpu(),
            "lin_acc": self._imu.data.lin_acc_b.clone().cpu(),
        }
        return self._imu_name, data_dict

@configclass
class ImuRecorderCfg(RecorderTermCfg):
    class_type: type = ImuRecorder
    imu_name: str | None = None


# Contact Recorder
class ContactRecorder(MyRecorderTerm):
    """
    每次 env.step() 内部的每个 physics 子步结束后，检测接触力传感器是否更新，并将新值记录。
    """
    def __init__(self, cfg: "ContactRecorderCfg", env):
        super().__init__(cfg, env)
        if cfg.contact_sensor_name is None:
            raise ValueError("The 'contact_sensor_name' must be provided for the ContactRecorder.")
        self._contact_sensor_name: str = cfg.contact_sensor_name
        self._contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
        if not self._contact_sensor.cfg.track_pose:
            raise ValueError("ContactRecorder requires the contact sensor to track pose (track_pose=True).")
        self._contact_sensor_dt = self._contact_sensor.cfg.update_period
        # Initialize next update time. The first update will be at or after this time.
        self._next_update_time = torch.full_like(self._timestamp, self._contact_sensor_dt)

    def record_post_reset(self, env_ids: Sequence[int] | None):
        """在每次环境重置后调用，重置下次更新时间。"""
        # First, call the parent's reset method to handle timestamp reset
        super().record_post_reset(env_ids)
        # Now, reset the next update time for the specified environments
        if env_ids is not None:
            # Note: self._timestamp is already reset to 0 in the parent call
            self._next_update_time[env_ids] = self._contact_sensor_dt
        else:
            self._next_update_time = torch.full_like(self._timestamp, self._contact_sensor_dt)
        return None, None

    def record_post_physics_step(self):
        self.update_timestamp()

        # Assuming all envs are synchronized, check the first one
        if self._timestamp[0] < self._next_update_time[0]:
            # 未到接触传感器更新周期
            return None, None

        # It's time to record, so update the next trigger time for all envs.
        # This prevents drift by adding the fixed period, rather than basing it on the current time.
        # A while loop handles cases where the simulation might lag and skip multiple recording intervals.
        while self._timestamp[0] >= self._next_update_time[0]:
            self._next_update_time += self._contact_sensor_dt

        quat_w = self._contact_sensor.data.quat_w.clone()
        net_forces_w = self._contact_sensor.data.net_forces_w.clone()
        forces_body = quat_apply_inverse(quat_w, net_forces_w)
        data_dict = {
            "timestamp": self._timestamp.clone().cpu(),
            "forces": forces_body.cpu(),
        }
        return self._contact_sensor_name, data_dict
    
@configclass
class ContactRecorderCfg(RecorderTermCfg):
    class_type: type = ContactRecorder
    contact_sensor_name: str | None = None

    
# 相机 Recorder
class CameraRecorder(MyRecorderTerm):
    """
    每次 env.step() 内部的每个 physics 子步结束后，检测是否有新帧产生并记录
    """
    def __init__(self, cfg: "CameraRecorderCfg", env):
        super().__init__(cfg, env)
        if cfg.cam_name is None:
            raise ValueError("The 'cam_name' must be provided for the CameraRecorder.")
        self._cam: Camera = env.scene.sensors[cfg.cam_name]
        self._cam_name = cfg.cam_name
        self._frame_id = torch.zeros(self._env.num_envs, device=self._env.device)

    def record_post_reset(self, env_ids):
        if env_ids is not None:
            self._frame_id[env_ids] = 0
        else:
            self._frame_id = torch.zeros(self._env.num_envs, device=self._env.device)
        return super().record_post_reset(env_ids)

    def record_post_physics_step(self):
        self.update_timestamp()

        data = self._cam.data
        frame_id = self._cam.frame.clone()
        if frame_id == self._frame_id:
            # 如果帧 ID 没有变化，说明没有新帧可用
            return None, None
        
        self._frame_id = frame_id
        data_dict = {
            "timestamp": self._timestamp.clone().cpu(),
            "frame_id": self._frame_id.clone().cpu(),
            "rgb": data.output["rgb"].clone().cpu(),
        }
        return self._cam_name, data_dict

@configclass
class CameraRecorderCfg(RecorderTermCfg):
    class_type: type = CameraRecorder
    cam_name: str | None = None
