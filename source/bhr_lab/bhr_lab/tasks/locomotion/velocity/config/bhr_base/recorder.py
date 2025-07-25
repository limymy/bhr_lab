import torch
from pathlib import Path

from isaaclab.managers.recorder_manager import (
    RecorderTerm,               # 基类
    RecorderTermCfg,            # 单个 Recorder term 的配置
)
from isaaclab.utils import configclass
from isaaclab.sensors import Camera

class RGBFrameRecorder(RecorderTerm):
    """
    每次 env.step() 内部的每个 physics 子步结束后，
    相机已更新若干帧；在 record_post_step 里把最新帧写入 episode buffer。
    """
    def __init__(self, cfg: "RGBFrameRecorderCfg", env):
        super().__init__(cfg, env)
        # 拿到你在 scene 里放好的相机（假设叫 front_cam）
        self.cam: Camera = env.scene[self.cfg.cam_name]

    def record_post_step(self):
        """把最新一帧 RGB 拿出来，存到 episode['camera']['rgb']"""
        rgb = self.cam.data.output["rgb"]
        print(f"Recording RGB frame: {self.cam.frame} {self.cam.image_shape}")
        return "camera/rgb", rgb                          # (key, value)

@configclass
class RGBFrameRecorderCfg(RecorderTermCfg):
    """把 term 注册到 Hydra 时用的 cfg"""

    class_type: type = RGBFrameRecorder
    cam_name: str = "cam_left"
