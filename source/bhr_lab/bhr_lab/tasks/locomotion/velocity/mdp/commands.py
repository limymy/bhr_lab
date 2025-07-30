from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.assets import Articulation
from isaaclab.utils import configclass
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg


class DelayedUniformVelocityCommand(UniformVelocityCommand):
    """
    A velocity command generator that enforces zero velocity for a specified duration
    at the beginning of each episode before switching to a uniform random velocity.
    """
    cfg: DelayedUniformVelocityCommandCfg

    def __init__(self, cfg: DelayedUniformVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)
        # buffer to track the time since the episode started for each env
        self.episode_time = torch.zeros(self.num_envs, device=self.device)
        self.zero_command = torch.zeros_like(self.vel_command_b)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset the command for the selected environment indices."""
        # reset the episode timer for the given environments
        self.episode_time[env_ids] = 0.0
        # call the parent reset to handle command resampling timers
        super()._reset_idx(env_ids)

    def _update_command(self):
        """
        Updates the command buffer.
        
        This method is called at every simulation step. It increments the episode timer
        and calls the parent's _update_command to apply normal processing like
        heading control.
        """
        # increment the episode timer
        self.episode_time += self._env.step_dt
        # call parent update for heading control and other logic
        super()._update_command()

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        # get the command from the parent class
        parent_command = super().command
        # determine which environments are in the delay phase
        in_delay_phase = self.episode_time < self.cfg.delay_time
        # return zero command for envs in delay phase, otherwise return parent command
        return torch.where(in_delay_phase.unsqueeze(-1), self.zero_command, parent_command)

            
@configclass
class DelayedUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    """
    Configuration for a uniform velocity command generator that enforces zero velocity
    for a specified duration at the beginning of each episode.
    """
    class_type: type = DelayedUniformVelocityCommand
    
    delay_time: float = 1.0
    """The duration (in seconds) to enforce zero velocity at the start of an episode."""
