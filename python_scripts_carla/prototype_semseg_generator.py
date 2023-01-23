#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import queue
import glob
import os
import sys
import time
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, args):
        self.world = carla_world
        self.sync = args.sync
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.__player = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._cam_width = args.width
        self._cam_height = args.height
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        
        if not self.map.get_spawn_points():
                    print('There are no spawn points available in your map/town.')
                    print('Please add some Vehicle Spawn Point to your UE4 scene.')
                    sys.exit(1)

        self.__player_blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))
        self.__player_available_waypoints = self.map.generate_waypoints(1.0)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.__player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.world.unload_map_layer(selected)
        else:
            self.world.load_map_layer(selected)

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def __spawn_player(self):
        self.__player_blueprint.set_attribute('role_name', self.actor_role_name)
        if self.__player_blueprint.has_attribute('color'):
            color = random.choice(self.__player_blueprint.get_attribute('color').recommended_values)
            self.__player_blueprint.set_attribute('color', color)
        if self.__player_blueprint.has_attribute('driver_id'):
            driver_id = random.choice(self.__player_blueprint.get_attribute('driver_id').recommended_values)
            self.__player_blueprint.set_attribute('driver_id', driver_id)
        if self.__player_blueprint.has_attribute('is_invincible'):
            self.__player_blueprint.set_attribute('is_invincible', 'true')

        spawn_point = random.choice(self.__player_available_waypoints).transform if self.__player_available_waypoints else carla.Transform()
        while self.__player is None:
            spawn_point.location.z += 0.1
            self.__player = self.world.try_spawn_actor(self.__player_blueprint, spawn_point)
            self.modify_vehicle_physics(self.__player)
        self.world.tick()

    def __spawn_sensors(self):
        if self.camera_manager is None: 
            if self.__player is not None:
                self.camera_manager = SensorsManager(self.world, self._cam_width, self._cam_height)
            else:
                print('Ego vehicle spawn --> Sensors spawn.')
        self.camera_manager.transform_index = 0
        self.camera_manager.spawn_sensors(self.__player)

    def __despawn_player(self):
        if self.__player is not None:
            print(self.__player.destroy())
            self.__player = None
        self.world.tick()

    def __despawn_sensors(self):
        self.camera_manager.despawn_sensors()

    def apply_settings(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        # TODO: Fix hardcoded delta sec. value
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def reset_settings(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
            
    def spawn_actors(self):
        self.__spawn_player()
        self.__spawn_sensors()

    def despawn_actors(self):
        self.__despawn_sensors()
        self.__despawn_player()

class SensorsManager(object):
    RESOLUTION_MULTIPLIER = 2.25
    SEM_CURRENT_FRAME = 0
    RGB_CURRENT_FRAME = 0

    def __init__(self, world, width, height):
        self.world = world
        self.queue_dict = {}
        self.width = SensorsManager.RESOLUTION_MULTIPLIER * width
        self.height = SensorsManager.RESOLUTION_MULTIPLIER * height
        self.transform_index = 0
        self.bp_library = None
        self.sensor_types = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
        ]
        
        self.index = None

    def despawn_sensors(self):
        if self.sensor_rgb is not None:
            self.sensor_rgb.stop()
            self.sensor_rgb.destroy()
            self.sensor_rgb = None
        if self.sensor_semseg is not None:
            self.sensor_semseg.stop()    
            self.sensor_semseg.destroy()
            self.sensor_semseg = None

    def spawn_sensors(self, parent_actor):
        self.__parent = parent_actor

        if self.bp_library is None:
            self.bp_library = self.__parent.get_world().get_blueprint_library()
            # TODO: Solve hardcoded values

            # Epsilon value is subtracted to ensure that there is no sensor capture miss because of world tick rate
            # RGB sensor blueprint fetch
            self.bp_sensor_rgb = self.bp_library.find('sensor.camera.rgb')
            self.bp_sensor_rgb.set_attribute('image_size_x', str(self.width))
            self.bp_sensor_rgb.set_attribute('image_size_y', str(self.height))
            self.bp_sensor_rgb.set_attribute('sensor_tick', str(0.5 - sys.float_info.epsilon))
            # Semantic segmentation blueprint fetch
            self.bp_sensor_semseg = self.bp_library.find('sensor.camera.semantic_segmentation')
            self.bp_sensor_semseg.set_attribute('image_size_x', str(self.width))
            self.bp_sensor_semseg.set_attribute('image_size_y', str(self.height))
            self.bp_sensor_semseg.set_attribute('sensor_tick', str(0.5 - sys.float_info.epsilon))
            
        vehicle_bound_x = 0.5 + self.__parent.bounding_box.extent.x
        vehicle_bound_y = 0.5 + self.__parent.bounding_box.extent.y
        vehicle_bound_z = 0.5 + self.__parent.bounding_box.extent.z
        camera_transform = carla.Transform(carla.Location(x=+0.8*vehicle_bound_x, y=+0.0*vehicle_bound_y, z=1.3*vehicle_bound_z))

        self.sensor_rgb = self.__parent.get_world().spawn_actor(
                self.bp_sensor_rgb,
                camera_transform,
                attach_to=self.__parent,
                attachment_type=carla.AttachmentType.Rigid)
        self.sensor_semseg = self.__parent.get_world().spawn_actor(
                self.bp_sensor_semseg,
                camera_transform,
                attach_to=self.__parent,
                attachment_type=carla.AttachmentType.Rigid)

        self.world.tick()
        self.__listen_sensors()
         
    def __listen_sensors(self):
        self.queue_dict['rgb'] = queue.Queue()
        self.queue_dict['semantic_segmentation'] = queue.Queue()
        self.sensor_rgb.listen(self.queue_dict['rgb'].put)
        self.sensor_semseg.listen(self.queue_dict['semantic_segmentation'].put)

    @staticmethod
    def parse_image(data):
        # TODO: Parametrize hardcoded path strings
        if data["name"] == 'semantic_segmentation':
            array = np.frombuffer(data["image"].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data["image"].height, data["image"].width, 4))
            array = array[:, :, 2]
            array_resized = cv2.resize(array, (int(data["image"].width / SensorsManager.RESOLUTION_MULTIPLIER), int(data["image"].height / SensorsManager.RESOLUTION_MULTIPLIER)))
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\semantic_segmentation\\synthetic_{data["name"]}_{SensorsManager.SEM_CURRENT_FRAME}.png', array_resized)
            SensorsManager.SEM_CURRENT_FRAME += 1
        elif data["name"] == 'rgb':
            array = np.frombuffer(data["image"].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data["image"].height, data["image"].width, 4))
            array = array[:, :, :3]
            array_resized = cv2.resize(array, (int(data["image"].width / SensorsManager.RESOLUTION_MULTIPLIER), int(data["image"].height / SensorsManager.RESOLUTION_MULTIPLIER)))
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\rgb\\synthetic_{data["name"]}_{SensorsManager.RGB_CURRENT_FRAME}.png', array_resized)
            SensorsManager.RGB_CURRENT_FRAME += 1

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    
    world = None
    number_of_images = 10

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)
        sim_world = client.get_world()

        if args.autopilot and not sim_world.get_settings().synchronous_mode:
            print("WARNING: You are currently in asynchronous mode and could "
                "experience some issues with the traffic simulation")   

        world = World(sim_world, args)
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

        world.apply_settings()
        for index in range(number_of_images):
            world.spawn_actors()
            world.world.tick()
            print(len(world.camera_manager.queue_dict.items()))
            for queue_key, queue in world.camera_manager.queue_dict.items():
                while not queue.empty():
                    SensorsManager.parse_image({'image': queue.get(), 'name': queue_key})
            print(index)
            world.despawn_actors()

    finally:
        world.reset_settings()  
        pass

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='2048x1024',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.tesla.model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='ego',
        help='actor role name (default: "ego")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        default='True',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
