
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower_dt")
@dataclass
class SO101FollowerDTConfig(RobotConfig):
    # Path to the URDF file for the digital twin
    urdf_path: str
        

    # cameras (optional for simulation)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)