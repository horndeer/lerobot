#!/usr/bin/env python

import logging
import time
from functools import cached_property
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import rerun as rr

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.cameras import CameraConfig
from lerobot.motors import MotorCalibration
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .config_so101_follower_DT import SO101FollowerDTConfig

logger = logging.getLogger(__name__)




class SO101FollowerDT(Robot):
    """
    SO-101 Follower Arm Digital Twin using URDF for digital twin.
    """

    config_class = SO101FollowerDTConfig
    name = "so101_follower_dt"

    def __init__(self, config: SO101FollowerDTConfig):
        super().__init__(config)
        self.config = config
        
        self.motor_steps = 4096

        # Initialize joint positions (in degrees if use_degrees=True, otherwise in normalized range)
        # Default to middle positions
        self.joint_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,  # Always in 0-100 range
        }
        
        self.joint_path = {"shoulder_pan": "base_link/shoulder_pan"}
        self.joint_path["shoulder_lift"] = (
            self.joint_path["shoulder_pan"] + "/shoulder_link/shoulder_lift"
        )
        self.joint_path["elbow_flex"] = (
            self.joint_path["shoulder_lift"] + "/upper_arm_link/elbow_flex"
        )
        self.joint_path["wrist_flex"] = (
            self.joint_path["elbow_flex"] + "/lower_arm_link/wrist_flex"
        )
        self.joint_path["wrist_roll"] = self.joint_path["wrist_flex"] + "/wrist_link/wrist_roll"
        self.joint_path["gripper"] = self.joint_path["wrist_roll"] + "/gripper_link/gripper"

        self.joint_axis = {
            "shoulder_pan": [0, 0, 1],
            "shoulder_lift": [0, 1, 0],
            "elbow_flex": [0, 0, 1],
            "wrist_flex": [0, 0, 1],
            "wrist_roll": [0, 1, 0],
            "gripper": [0, 1, 0],  # Gripper axis
        }

        # Log joint transforms to animate the URDF
        self.base_path = "so101/so101_new_calib"

        # Initialize cameras if configured
        # self.cameras = make_cameras_from_configs(config.cameras)
        self.cameras = {}

        # Track connection state
        self._connected = False

        # Validate URDF path
        self.urdf_path = Path(config.urdf_path)
        if not self.urdf_path.exists():
            logger.warning(f"URDF file not found at: {self.urdf_path}")

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.joint_positions}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._connected and all(
            cam.is_connected for cam in self.cameras.values()
        )

    def connect(self, calibrate: bool = True) -> None:
        """Connect the digital twin simulation and display URDF."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Load and display the URDF robot model
        if self.urdf_path.exists():
            print(f"Loading URDF from: {self.urdf_path}")
            try:
                rr.log_file_from_path(self.urdf_path, entity_path_prefix="so101")
                print("URDF loaded successfully!")
            except Exception as e:
                print(f"Error loading URDF: {e}")
                print("Make sure you have the latest version of Rerun installed:")
                print("pip install --upgrade rerun-sdk")
        else:
            print(f"URDF file not found at: {self.urdf_path}")
            print("please provide a valid URDF file")
            exit()

        # Mark as connected
        self._connected = True

        # Connect cameras if any
        # for cam in self.cameras.values():
        #     cam.connect()

        # For digital twin, we don't need real calibration
        if calibrate and not self.is_calibrated:
            logger.info("Digital twin mode: skipping calibration")

        self.configure()
        logger.info(f"{self} digital twin connected.")

    @property
    def is_calibrated(self) -> bool:
        """For digital twin, we're always 'calibrated'."""
        return True

    def calibrate(self) -> None:
        """No-op for digital twin."""
        logger.info("Simulation mode: calibration not needed")

    def configure(self) -> None:
        """Configure the digital twin parameters."""
        logger.info(f"{self} digital twin configured with URDF: {self.urdf_path}")

    def get_observation(self) -> dict[str, Any]:
        """Get current simulation state (joint positions and camera images)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Return current joint positions
        obs_dict = {f"{motor}.pos": val for motor, val in self.joint_positions.items()}

        # Capture images from cameras if any
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Update joint positions in the digital twin and control 3D transforms.

        Args:
            action: Dictionary with joint position commands in format {"joint_name.pos": value}

        Returns:
            The action actually applied (no safety limits - behaves like original SO101Follower)
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Extract goal positions from action
        goal_pos = {
            key.removesuffix(".pos"): val
            for key, val in action.items()
            if key.endswith(".pos")
        }

        # Update joint positions (immediate in simulation - no safety limits)
        for joint_name, target_pos in goal_pos.items():
            if joint_name in self.joint_positions:
                if self.calibration and joint_name in self.calibration:
                    target_pos = np.clip(target_pos, self.calibration[joint_name].range_min, self.calibration[joint_name].range_max)
                else:
                    self.joint_positions[joint_name] = target_pos

        # Control 3D transforms in Rerun (same pattern as sim_test.py)
        self._log_joint_transforms()

        # Return the action actually sent (no clipping)
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    def _log_joint_transforms(self) -> None:
        """Log joint transforms to Rerun for URDF animation (same pattern as sim_test.py)."""
        # Convert degrees to radians for URDF joints if using degrees
        joint_positions = {
            joint: ( pos - self.motor_steps / 2) / (self.motor_steps) * 2 * np.pi for joint, pos in self.joint_positions.items()
        }


        # Joint paths and axes (from sim_test.py)
        for joint_name, angle in joint_positions.items():
            if joint_name in self.joint_path:
                # Log the joint transform using the correct joint path from URDF
                rr.log(
                    f"{self.base_path}/{self.joint_path[joint_name]}",
                    rr.Transform3D(
                        rotation=rr.RotationAxisAngle(
                            axis=self.joint_axis[joint_name], radians=angle
                        )
                    ),
                )

    def disconnect(self) -> None:
        """Disconnect the digital twin simulation."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self._connected = False

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} digital twin disconnected.")

    def get_joint_positions(self) -> dict[str, float]:
        """Get current joint positions (convenience method for visualization)."""
        return self.joint_positions.copy()

    def set_joint_positions(self, positions: dict[str, float]) -> None:
        """Set joint positions directly (convenience method for testing)."""
        for joint, pos in positions.items():
            if joint in self.joint_positions:
                self.joint_positions[joint] = pos
