#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

from __future__ import print_function
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

import carla

from carla import ColorConverter as cc

import argparse
import logging
import random
import re

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

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

    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class World(object):

    def __init__(self, carla_client, carla_world, args):
        self.client = carla_client
        self.world = carla_world
        self.sync = args.sync
        self._ego = None
        self.camera_manager = None
        self.vehicles_manager = None
        self.maps = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"] # list(map(lambda x: x.split('/')[-1], self.client.get_available_maps()))
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._cam_width = args.width
        self._cam_height = args.height
        self._ego_blueprint = random.choice(get_actor_blueprints(self.world, self._actor_filter, self._actor_generation))

    def set_weather(self, index):
        mod_index = index % len(self._weather_presets)
        preset = self._weather_presets[mod_index]
        self._ego.get_world().set_weather(preset[0])

    def set_map(self, index):
        self.client.load_world(self.maps[index], reset_settings=False)
        self.world.tick()
        time.sleep(1)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        try:
            self.map_available_waypoints = self.map.generate_waypoints(3.0)
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)

    def modify_vehicle_physics(self, actor):
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def _spawn_ego(self):
        if self._ego_blueprint.has_attribute('color'):
            color = random.choice(self._ego_blueprint.get_attribute('color').recommended_values)
            self._ego_blueprint.set_attribute('color', color)
        if self._ego_blueprint.has_attribute('driver_id'):
            driver_id = random.choice(self._ego_blueprint.get_attribute('driver_id').recommended_values)
            self._ego_blueprint.set_attribute('driver_id', driver_id)
        if self._ego_blueprint.has_attribute('is_invincible'):
            self._ego_blueprint.set_attribute('is_invincible', 'true')

        spawn_point = carla.Transform()
        if self.map_available_waypoints:
            spawn_point = random.choice(self.map_available_waypoints)        
            self.map_available_waypoints.remove(spawn_point)
            spawn_point = spawn_point.transform

        while self._ego is None:
            spawn_point.location.z += 0.1
            self._ego = self.world.try_spawn_actor(self._ego_blueprint, spawn_point)
            self.modify_vehicle_physics(self._ego)
        spectator_transform = spawn_point
        spectator_transform.location.z += 2.0
        self.world.get_spectator().set_transform(spectator_transform)
        self.world.tick()

    def _spawn_sensors(self):
        if self.camera_manager is None: 
            if self._ego is not None:
                self.camera_manager = SensorsManager(self.world, self._cam_width, self._cam_height)
            else:
                print('Ego vehicle spawn --> Sensors spawn.')
                exit(-1)
        self.camera_manager.transform_index = 0
        self.camera_manager.spawn_sensors(self._ego)

    def _spawn_vehicles(self):
        if self.vehicles_manager is None:
            if self._ego is not None:
            # TODO: Parametrize vehicle count
                self.vehicles_manager = VehiclesManager(self.world, vehicle_count=3)
            else:
                print('Ego vehicle spawn --> Sensors spawn.')
                exit(-1)
        self.vehicles_manager.spawn_vehicles(spawn_points=self.map_available_waypoints)

    def _despawn_ego(self):
        if self._ego is not None:
            print(self._ego.destroy())
            self._ego = None
        self.world.tick()

    def _despawn_sensors(self):
        self.camera_manager.despawn_sensors()

    def _despawn_vehicles(self):
        self.vehicles_manager.despawn_vehicles()

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
        self._spawn_ego()
        self._spawn_vehicles()
        self._spawn_sensors()

    def despawn_actors(self):
        self._despawn_sensors()
        self._despawn_vehicles()
        self._despawn_ego()

# TODO: Implement random vehicle traffic
class VehiclesManager(object):
    def __init__(self, world, vehicle_count) -> None:
        self.world = world
        self.vehicle_count = vehicle_count
        self.spawned_vehicles = []

    def spawn_vehicles(self, spawn_points):
        shuffled_spawn_points = spawn_points.copy()
        random.shuffle(shuffled_spawn_points)
        
        # Choose //vehicle_count// spawn points, iterate over and pop one at a time
        if self.vehicle_count > len(shuffled_spawn_points):
            print('More vehicles than available spawn points!')
            exit(-1)

        vehicle_blueprints = get_actor_blueprints(self.world, "vehicle.*", "2")
        for _ in range(self.vehicle_count):   
            spawn_point = random.choice(shuffled_spawn_points)
            shuffled_spawn_points.remove(spawn_point)
            spawn_point = spawn_point.transform
            vehicle_blueprint = random.choice(vehicle_blueprints)

            if vehicle_blueprint.has_attribute('color'):
                color = random.choice(vehicle_blueprint.get_attribute('color').recommended_values)
                vehicle_blueprint.set_attribute('color', color)
            if vehicle_blueprint.has_attribute('driver_id'):
                driver_id = random.choice(vehicle_blueprint.get_attribute('driver_id').recommended_values)
                vehicle_blueprint.set_attribute('driver_id', driver_id)
            if vehicle_blueprint.has_attribute('is_invincible'):
                vehicle_blueprint.set_attribute('is_invincible', 'true')

            vehicle = None
            while vehicle is None:
                spawn_point.location.z += 0.1
                vehicle = self.world.try_spawn_actor(vehicle_blueprint, spawn_point)
            self.spawned_vehicles += [vehicle]
            
            spectator_transform = spawn_point
            spectator_transform.location.z += 2.0
            self.world.get_spectator().set_transform(spectator_transform)
            self.world.tick()
            time.sleep(1.0)

    def despawn_vehicles(self):
        for spawned_vehicle in self.spawned_vehicles:
            if spawned_vehicle is not None:
                spawned_vehicle.destroy()
                spawned_vehicle = None
            self.world.tick()


class SensorsManager(object):
    RESOLUTION_MULTIPLIER = 2.25
    SEM_CURRENT_FRAME = 0
    RGB_CURRENT_FRAME = 0

    def __init__(self, world, width, height):
        self._parent = None
        self.world = world
        self.queue_dict = {}
        self.width = SensorsManager.RESOLUTION_MULTIPLIER * width
        self.height = SensorsManager.RESOLUTION_MULTIPLIER * height
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
        self._parent = parent_actor

        if self.bp_library is None:
            self.bp_library = self._parent.get_world().get_blueprint_library()
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
            
        vehicle_bound_x = 0.5 + self._parent.bounding_box.extent.x
        vehicle_bound_y = 0.5 + self._parent.bounding_box.extent.y
        vehicle_bound_z = 0.5 + self._parent.bounding_box.extent.z
        camera_transform = carla.Transform(carla.Location(x=+0.8*vehicle_bound_x, y=+0.0*vehicle_bound_y, z=1.3*vehicle_bound_z))

        self.sensor_rgb = self._parent.get_world().spawn_actor(
                self.bp_sensor_rgb,
                camera_transform,
                attach_to=self._parent,
                attachment_type=carla.AttachmentType.Rigid)
        self.sensor_semseg = self._parent.get_world().spawn_actor(
                self.bp_sensor_semseg,
                camera_transform,
                attach_to=self._parent,
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
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\{data["name"]}\\synthetic_{data["name"]}_{SensorsManager.SEM_CURRENT_FRAME}.png', array_resized)
            SensorsManager.SEM_CURRENT_FRAME += 1
        elif data["name"] == 'rgb':
            array = np.frombuffer(data["image"].raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data["image"].height, data["image"].width, 4))
            array = array[:, :, :3]
            array_resized = cv2.resize(array, (int(data["image"].width / SensorsManager.RESOLUTION_MULTIPLIER), int(data["image"].height / SensorsManager.RESOLUTION_MULTIPLIER)))
            cv2.imwrite(f'C:\\Users\\Manuel\\Projects\\GitHub_Repositories\\master_thesis\\datasets\\synthetic\\{data["name"]}\\synthetic_{data["name"]}_{SensorsManager.RGB_CURRENT_FRAME}.png', array_resized)
            SensorsManager.RGB_CURRENT_FRAME += 1

def simulation_loop(args):
    
    world = None
    number_of_images = 24

    client = carla.Client(args.host, args.port)
    client.set_timeout(200.0)
    sim_world = client.get_world()

    world = World(client, sim_world, args)
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    map_index = -1
    for image_index in range(number_of_images):
        world.apply_settings()

        new_map_index = int(image_index / (number_of_images / len(world.maps)))
        if new_map_index != map_index:
            # TODO: Fix sleeping with a wait on the currently active actors (especially cameras)
            time.sleep(2)
            map_index = new_map_index
            world.set_map(map_index)
            print('------------------------- New map -------------------------')

        world.spawn_actors()
        world.set_weather(image_index)
        world.world.tick()

        print('Map index:\t', map_index)
        print('Image index:\t', image_index)
        print('Cameras:\t', len(world.camera_manager.queue_dict.items()))
        
        for queue_key, queue in world.camera_manager.queue_dict.items():
            print(f'\t{queue_key} received images', queue.qsize())
            while not queue.empty():
                SensorsManager.parse_image({'image': queue.get(), 'name': queue_key})

        world.despawn_actors()
    world.reset_settings()


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
        simulation_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()
