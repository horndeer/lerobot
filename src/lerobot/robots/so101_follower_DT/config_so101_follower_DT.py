
from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower_dt")
@dataclass
class SO101FollowerDTConfig(RobotConfig):
    # Path to the URDF file for the digital twin
    urdf_path: str
    
    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = True

    # cameras (optional for simulation)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)