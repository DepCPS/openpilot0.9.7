import math
from multiprocessing import Queue

from metadrive.component.sensors.base_camera import _cuda_enable
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.component.map.base_map import BaseMap
from metadrive.component.sensors.lidar import Lidar
import logging

from openpilot.tools.sim.bridge.common import SimulatorBridge
from openpilot.tools.sim.bridge.metadrive.metadrive_common import RGBCameraRoad, RGBCameraWide
from openpilot.tools.sim.bridge.metadrive.metadrive_world import MetaDriveWorld
from openpilot.tools.sim.lib.camerad import W, H


def straight_block(length):
  return {
    "id": "S",
    "pre_block_socket_index": 0,
    "length": length
  }

def curve_block(length, angle=45, direction=0):
  return {
    "id": "C",
    "pre_block_socket_index": 0,
    "length": length,
    "radius": length,
    "angle": angle,
    "dir": direction
  }

def create_map(track_size=60):
  curve_len = track_size * 2
  return dict(
    type=MapGenerateMethod.PG_MAP_FILE,
    lane_num=3,
    lane_width=4.5,
    config=[
      None,
      straight_block(120),
      curve_block(240, 90),
      straight_block(120),
      curve_block(240, 90),
      straight_block(120),
      curve_block(240, 90),
      straight_block(120),
      curve_block(240, 90),
    ]
  )
'''
      None,
      straight_block(120),
      straight_block(120),
      straight_block(120),
      curve_block(240, 90),
      straight_block(120),
      straight_block(120),
      curve_block(240, 90),
      straight_block(120),
      straight_block(120),
      straight_block(120),
      straight_block(120),
      straight_block(120),
      straight_block(120),
'''


class MetaDriveBridge(SimulatorBridge):
  TICKS_PER_FRAME = 5

  def __init__(self, dual_camera, high_quality, test_duration=110, test_run=True): # test_duration=math.inf, test_run=False
    super().__init__(dual_camera, high_quality)

    self.should_render = False
    self.test_run = test_run
    self.test_duration = test_duration if self.test_run else math.inf

  def spawn_world(self, queue: Queue):
    sensors = {
      "rgb_road": (RGBCameraRoad, W, H, )
    }

    if self.dual_camera:
      sensors["rgb_wide"] = (RGBCameraWide, W, H)

    config = dict(
      use_render=self.should_render,
      vehicle_config=dict(
        enable_reverse=False,
        image_source="rgb_road",
        spawn_longitude=-50,
        #spawn_longitude=-10,
        #spawn_lateral=5.0,
        # wheel friction can change in metadrive/component/vehicle/base_vehicle.py
      ),
      sensors=sensors,
      image_on_cuda=_cuda_enable,
      image_observation=True,
      interface_panel=[],
      out_of_route_done=False,
      on_continuous_line_done=True,
      crash_vehicle_done=True,
      crash_object_done=False,
      arrive_dest_done=False,
      decision_repeat=1,
      physics_world_step_size=self.TICKS_PER_FRAME/100,
      preload_models=False,

      #render_pipeline=False,
      #daytime='22:00',
      #shadow_range=50,

      # map and traffic
      traffic_density=0.0, # traffic is incredibly expensive
      map_region_size=2048,
      map_config=create_map(),
      log_level=logging.WARNING,

    )

    return MetaDriveWorld(queue, config, self.test_duration, self.test_run, self.dual_camera)
