# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import e
import os
from typing import Any
from dataclasses import dataclass
import numpy as np
import rerun as rr


@dataclass
class RerunConfig:
    session_name: str = "lerobot_control_loop"
    port: int | None = None
    address: str | None = None
    blueprint_path: str | None = None


def _init_rerun(cfg: RerunConfig) -> None:
    """Initializes the Rerun SDK for visualizing the control loop.
    If address and port are provided, it connects to the remote rerun server with gRPC.
    Otherwise, it spawns a local viewer.
    """
    session_name = cfg.session_name
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    if cfg.address is not None and cfg.port is not None:
        rr.connect_grpc(f"rerun+http://{cfg.address}:{cfg.port}/proxy")
    else:
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)



def log_rerun_data(observation: dict[str | Any], action: dict[str | Any]):
    for obs, val in observation.items():
        if isinstance(val, float):
            rr.log(f"observation.{obs}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                for i, v in enumerate(val):
                    rr.log(f"observation.{obs}_{i}", rr.Scalars(float(v)))
            else:
                rr.log(f"observation.{obs}", rr.Image(val), static=True)
    for act, val in action.items():
        if isinstance(val, float):
            rr.log(f"action.{act}", rr.Scalars(val))
        elif isinstance(val, np.ndarray):
            for i, v in enumerate(val):
                rr.log(f"action.{act}_{i}", rr.Scalars(float(v)))
