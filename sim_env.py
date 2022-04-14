
import os
import pickle
import queue

import numpy as np
import pyflex
import igl

from flex_utils import (PickerPickPlace, get_camera_matrix,
                        get_current_cover_area, get_default_camera_config,
                        get_default_scene_config, load_cloth,
                        set_random_cloth_color, wait_until_stable)


class SimEnv:
    def __init__(self,
        gui=False,
        grasp_height=0.02,
        large_grasp=False,
        dump_visualizations=False,
        grasp_policy='heuristic',
        blow_policy='blow',
        wind_life_time=50,
        position_speed=0.005, # unit: meter
        orientation_speed=0.02,   # unit: degree
        blow_z_rotation=-105
    ):
        # set display
        gpu = os.environ['CUDA_VISIBLE_DEVICES']
        if not gui:
            os.environ['DISPLAY'] = f':0.{gpu}'
        
        self.gui = gui
        self.dump_visualizations = dump_visualizations
        self.env_video_frames = list()
        self.grasp_height = grasp_height
        self.ray_handle = None
        self.wind_life_time = wind_life_time
        self.position_speed = position_speed
        self.orientation_speed = orientation_speed
        self.blow_z_rotation = blow_z_rotation
        
        self.scene_config = get_default_scene_config()
        self.camera_config = get_default_camera_config()
        self.image_size = self.camera_config['cam_size']
        pyflex.init(not self.gui, True, self.image_size[0], self.image_size[1], self.scene_config['msaaSamples'])
        self.grasp_policy = grasp_policy
        self.blow_policy = blow_policy
        self.num_picker = 1 if grasp_policy == 'pick_and_place' else 2
        self.action_tool = PickerPickPlace(num_picker=self.num_picker, particle_radius=self.scene_config['radius'], picker_radius=self.grasp_height, large_grasp=large_grasp)
        self.grasp_states = [False] * self.num_picker

        self.particle_view_camera_config = self.camera_config.copy()
        self.particle_view_camera_config['render_type'] = ['points']

        self.camera_config['render_type'] = ['cloth']
        pyflex.set_camera_params(self.camera_config)


    def reset(self, init_state_path=None):
        set_random_cloth_color()

        pyflex.set_scene(self.scene_config['scene_id'], self.scene_config)
        pyflex.set_camera_params(self.camera_config)
        self.cam_intr, self.cam_pose = get_camera_matrix(*pyflex.get_camera_params())

        self.wind_base_index_queue = queue.Queue(maxsize=self.wind_life_time)

        if init_state_path is None:
            init_state = {
                'cloth_stiffness': [0.8, 1, 0.9],
                'cloth_mass': 1,
                'cloth_type': 'square',
                'cloth_size': [70, 70],
                'max_cover_area': 0.7 * 0.7,
                'init_cover_area': 0.7 * 0.7
            }
        else:
            with open(init_state_path, 'rb') as f:
                init_state = pickle.load(f)

        stiffness = init_state['cloth_stiffness']
        cloth_mass = init_state['cloth_mass']
        self.cloth_type = init_state['cloth_type']
        if self.cloth_type == 'square':
            self.cloth_size = init_state['cloth_size']
            baseIndex, num_verts = pyflex.add_cloth_square([0, 0.1, 0], [0, 0, 0], self.cloth_size, stiffness, cloth_mass, True)
            self.stretch_method = 'option1'
        else:
            cloth_category = init_state['cloth_category']
            self.mesh_scaling = init_state['scaling']
            self.tri_v, self.tri_f = igl.read_triangle_mesh(init_state['mesh'])
            mesh_verts, mesh_faces, stretch_edges, bend_edges, shear_edges = load_cloth(init_state['mesh'])
            mesh_verts = mesh_verts * self.mesh_scaling
            baseIndex, num_verts = pyflex.add_cloth_mesh([0, 0.1, 0], mesh_verts.reshape(-1), mesh_faces.reshape(-1), stretch_edges.reshape(-1), bend_edges.reshape(-1), shear_edges.reshape(-1), stiffness, cloth_mass)
            self.stretch_method = 'option1'

        self.cloth_particle_num = num_verts
        wait_until_stable()
        self.max_cover_area = init_state['max_cover_area']

        # load init position
        if 'particle_pos' in init_state:
            positions = pyflex.get_positions()
            velcities = pyflex.get_velocities()
            positions[:init_state['particle_pos'].shape[0]] = init_state['particle_pos']
            velcities[:init_state['particle_vel'].shape[0]] = init_state['particle_vel']
            pyflex.set_positions(positions)
            pyflex.set_velocities(velcities)

        self.set_grasp(False)
        self.action_tool.reset(self.cloth_particle_num)

        init_pose = [[0.8, 0.5, 0], [-0.8, 0.5, 0]][:self.num_picker]
        self.movep(init_pose, speed=5e-2)

        self.blow_action = None
        init_cover_area = init_state['init_cover_area']

        return self.max_cover_area, init_cover_area, self.get_observation()


    def get_camera_matrix(self):
        return get_camera_matrix(*pyflex.get_camera_params())


    def step_simulation(self):
        pyflex.step()


    def stretch_cloth(self, grasp_dist: float, lifting_height: float = 0.5, max_grasp_dist: float = 1.0, increment_step=0.02):
        if self.stretch_method == 'option1':
            # Option1: get GT init position
            picked_particles = self.action_tool.picked_particles
            if self.cloth_type == 'square':
                pl = np.array([picked_particles[0][0] // self.cloth_size[0], picked_particles[0][0] % self.cloth_size[0]]) * self.scene_config['radius']
                pr = np.array([picked_particles[1][0] // self.cloth_size[0], picked_particles[1][0] % self.cloth_size[0]]) * self.scene_config['radius']
                grasp_dist = np.linalg.norm(pl - pr)
                grasp_dist_scaling = 1 + grasp_dist / 3 # TODO: hacky scaling factor
                grasp_dist *= grasp_dist_scaling
            else:
                grasp_dist = igl.exact_geodesic(v=self.tri_v, f=self.tri_f, vs=np.array([picked_particles[0]]), vt=np.array([picked_particles[1]]))
                grasp_dist_scaling = 1.15 # TODO: hacky scaling factor
                grasp_dist *= grasp_dist_scaling * self.mesh_scaling
            
            max_grasp_dist = 0.7 if self.blow_policy == 'fling' else 1
            grasp_dist = min(grasp_dist, max_grasp_dist)

            left, right = self.action_tool._get_picker_pos()
            left[1] = lifting_height
            right[1] = lifting_height
            midpoint = (left + right) / 2
            direction = left - right
            direction = direction/np.linalg.norm(direction)
            left = midpoint + direction * grasp_dist/2
            right = midpoint - direction * grasp_dist/2
            self.movep([left, right], speed=2e-3)
            return grasp_dist

        elif self.stretch_method == 'option2':
            # Option2: move until stable
            # keep stretching until cloth is tight
            left, right = self.action_tool._get_picker_pos()
            left[1] = lifting_height
            right[1] = lifting_height
            midpoint = (left + right)/2
            direction = left - right
            direction = direction/np.linalg.norm(direction)
            
            num_midpoints = 1
            cloth_midpoints = np.zeros([num_midpoints, 3])

            self.movep([left, right], speed=2e-3, min_steps=20)
            stable_steps = 0
            while True:
                # left, right = self.action_tool._get_picker_pos()
                midpoints = [left * i / (num_midpoints + 1) + right * (num_midpoints + 1 - i) / (num_midpoints + 1) for i in range(1, num_midpoints + 1)]

                positions = pyflex.get_positions().reshape((-1, 4))[:self.cloth_particle_num, :3]
                # get midpoints
                high_positions = positions[positions[:, 1] > lifting_height-0.1, ...]
                if (high_positions[:, 0] < 0).all() or (high_positions[:, 0] > 0).all():
                    # single grasp
                    return grasp_dist

                new_cloth_midpoints = list()
                for m in midpoints:
                    dist_key = [np.linalg.norm(pos[[0, 2]]-m[[0, 2]]) for pos in positions]
                    min_idx = np.argmin(dist_key)
                    new_cloth_midpoints.append(positions[min_idx])
                new_cloth_midpoints = np.array(new_cloth_midpoints)
                stable = np.max(np.linalg.norm(new_cloth_midpoints - cloth_midpoints, axis=1)) < 2e-2 and new_cloth_midpoints[0][1] > lifting_height * 0.5
                if stable:
                    stable_steps += 1
                else:
                    stable_steps = 0
                stretched = stable_steps > 1
                if stretched:
                    return grasp_dist
                cloth_midpoints = new_cloth_midpoints
                grasp_dist += increment_step
                left = midpoint + direction*grasp_dist/2
                right = midpoint - direction*grasp_dist/2
                self.movep([left, right], speed=2e-3)
                pyflex.step()
                pyflex.step()
                if grasp_dist > max_grasp_dist:
                    return max_grasp_dist


    def lift_and_stretch_primitive(self, p1, p2, lifting_height):
        self.blow_action = None
        left_grasp_pos, right_grasp_pos = p1.copy(), p2.copy()

        # premove
        left_grasp_pos[1] = 0.1
        right_grasp_pos[1] = 0.1
        self.action_tool.add_pickers([left_grasp_pos, right_grasp_pos])
        self.movep([left_grasp_pos, right_grasp_pos], speed=1e-1)

        # move to target position
        left_grasp_pos[1] = self.grasp_height
        right_grasp_pos[1] = self.grasp_height
        self.movep([left_grasp_pos, right_grasp_pos], speed=1e-2)
        lift_observation = self.get_observation()

        # grasp distance
        dist = np.linalg.norm(np.array(left_grasp_pos) - np.array(right_grasp_pos))

        self.set_grasp(True)
        pre_lift_height = 0.5
        self.movep([[dist/2, pre_lift_height, 0.3], [-dist/2, pre_lift_height, 0.3]], speed=5e-3)
        if not True in [x is None for x in self.action_tool.picked_particles]:
            dist = self.stretch_cloth(grasp_dist=dist, lifting_height=pre_lift_height)
            self.movep([[dist/2, lifting_height, 0.3], [-dist/2, lifting_height, 0.3]], speed=5e-3)
        stretch_observation = self.get_observation()
        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])
        return lift_observation, stretch_observation, cover_area

    def pick_and_place(self, p1, p2, lifting_height):
        # p1: picker
        # p2: place

        # premove
        pick_pos, place_pos = p1.copy(), p2.copy()
        pick_pos[1] = 0.1
        self.action_tool.add_pickers([pick_pos])
        self.movep([pick_pos], speed=1e-1)

        # move to target position
        pick_pos[1] = self.grasp_height
        self.movep([pick_pos], speed=1e-2)
        lift_observation = self.get_observation()

        self.set_grasp(True)

        pick_pos[1] = self.grasp_height
        self.movep([pick_pos], speed=1e-2)
        lift_observation = self.get_observation()

        pick_pos[1] = lifting_height
        self.movep([pick_pos], speed=1e-2)

        place_pos[1]= lifting_height
        self.movep([place_pos], speed=4e-3)

        stretch_observation = self.get_observation()
        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])
        return lift_observation, stretch_observation, cover_area


    def place_cloth(self):
        self.blow_action = None
        positions = pyflex.get_positions().reshape((-1, 4))
        positions[self.cloth_particle_num:, 0] = 100
        pyflex.set_positions(positions)
            
        if self.num_picker == 2:
            left_grasp_pos, right_grasp_pos = self.action_tool._get_picker_pos()
            left_grasp_pos[1] = self.grasp_height
            right_grasp_pos[1] = self.grasp_height
            left_grasp_pos[2] += 0.05
            right_grasp_pos[2] += 0.05
            self.movep([left_grasp_pos, right_grasp_pos], speed=1e-3)
        else:
            grasp_pos = self.action_tool._get_picker_pos()[0]
            grasp_pos[1] = self.grasp_height
            grasp_pos[2] += 0.05
            self.movep([grasp_pos], speed=1e-3)

        self.set_grasp(False)
        self.action_tool.remove_pickers()
        wait_until_stable(gui=self.gui)
        observation = self.get_observation()
        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])

        return cover_area, observation


    def blow(self, position, orientation, num_layer, alpha, velocity, mass, step_num):
        target_position = np.array(position)
        target_orientation = np.array(orientation)
        target_orientation[2] = self.blow_z_rotation / 180 * np.pi

        current_position = self.blow_action[:3] if self.blow_action is not None else target_position
        current_orientation = self.blow_action[3:] if self.blow_action is not None else target_orientation
        self.blow_action = np.concatenate([target_position, target_orientation])

        observation = self.get_observation()
        if not True in [x is None for x in self.action_tool.picked_particles]:
            for step in range(step_num):
                current_position = current_position + np.clip(target_position - current_position, -self.position_speed, self.position_speed)
                current_orientation = current_orientation + np.clip(target_orientation - current_orientation, -self.orientation_speed, self.orientation_speed)

                base_index = self.wind_base_index_queue.get() if self.wind_base_index_queue.full() else -1
                base_index, particle_num = pyflex.emit_particles_cone(current_position, current_orientation / np.pi * 180, num_layer, alpha, velocity, mass, base_index)
                # base_index, particle_num = pyflex.emit_particles_box(current_position, current_orientation / np.pi * 180, [10, 4], [0, velocity, 0], mass, self.scene_config['radius'] * 1.5, base_index, True)
                self.wind_base_index_queue.put(base_index)
                self.step_simulation()

        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])
        observation = self.get_observation(particle_view=True)

        return cover_area, observation


    def blow_box(self, position, orientation, velocity, mass, step_num):
        target_position = np.asarray(position)
        target_orientation = np.asarray(orientation)
        current_position = self.blow_action[:3] if self.blow_action is not None else target_position
        current_orientation = self.blow_action[3:] if self.blow_action is not None else target_orientation
        self.blow_action = np.concatenate([target_position, target_orientation])

        observation = self.get_observation()

        if not True in [x is None for x in self.action_tool.picked_particles]:
            for step in range(step_num):
                current_position = current_position + np.clip(target_position - current_position, -self.position_speed, self.position_speed)
                current_orientation = current_orientation + np.clip(target_orientation - current_orientation, -self.orientation_speed, self.orientation_speed)

                base_index = self.wind_base_index_queue.get() if self.wind_base_index_queue.full() else -1
                base_index, particle_num = pyflex.emit_particles_box(current_position, current_orientation / np.pi * 180, [10, 4], [0, velocity, 0], mass, self.scene_config['radius'] * 2, base_index, True)
                self.wind_base_index_queue.put(base_index)
                self.step_simulation()

        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])
        observation = self.get_observation(particle_view=True)

        return cover_area, observation
    

    def lift_cloth(self, grasp_dist: float, fling_height: float = 0.7, increment_step: float = 0.05, max_height=0.7):
        while True:
            positions = pyflex.get_positions().reshape((-1, 4))[:self.cloth_particle_num, :3]
            heights = positions[:, 1]
            if heights.min() > 0.02:
                return fling_height
            fling_height += increment_step
            self.movep([[grasp_dist/2, fling_height, 0.3],
                        [-grasp_dist/2, fling_height, 0.3]], speed=5e-3)
            if fling_height >= max_height:
                return fling_height


    def fling_cloth(self, fling_height=0.7, fling_speed=6e-3):
        left_grasp_pos, right_grasp_pos = self.action_tool._get_picker_pos()
        dist = np.linalg.norm(np.array(left_grasp_pos) - np.array(right_grasp_pos))

        fling_height = left_grasp_pos[1]
        fling_height = self.lift_cloth(dist, fling_height)

        # fling
        self.movep([[dist/2, fling_height, 0.2],
                    [-dist/2, fling_height, 0.2]], speed=fling_speed)
        self.movep([[dist/2, fling_height, -0.2],
                    [-dist/2, fling_height, -0.2]], speed=fling_speed)
        self.movep([[dist/2, fling_height, -0.2],
                    [-dist/2, fling_height, -0.2]], speed=1e-2, min_steps=4)
        # lower
        self.movep([[dist/2, self.grasp_height*2, 0.2],
                    [-dist/2, self.grasp_height*2, 0.2]], speed=1e-2)
        self.movep([[dist/2, self.grasp_height*2, 0.25],
                    [-dist/2, self.grasp_height*2, 0.25]], speed=5e-3)
                
        cover_area = get_current_cover_area(self.cloth_particle_num, self.scene_config['radius'])
        observation = self.get_observation(particle_view=True)

        return cover_area, observation



    def movep(self, pos, speed=None, limit=1000, min_steps=None, eps=1e-4):
        if speed is None:
            if self.dump_visualizations:
                speed = self.default_speed
            else:
                speed = 0.1
        target_pos = np.array(pos)
        for step in range(limit):
            curr_pos = self.action_tool._get_picker_pos()
            deltas = [(targ - curr) for targ, curr in zip(target_pos, curr_pos)]
            dists = [np.linalg.norm(delta) for delta in deltas]
            if all([dist < eps for dist in dists]) and (min_steps is None or step > min_steps):
                return
            action = []
            for targ, curr, delta, dist, gs in zip(target_pos, curr_pos, deltas, dists, self.grasp_states):
                if dist < speed:
                    action.extend([*targ, float(gs)])
                else:
                    delta = delta/dist
                    action.extend([*(curr + delta * speed), float(gs)])
            action = np.array(action)
            self.action_tool.step(action, step_sim_fn=self.step_simulation)
            if step % 4 == 0 and self.dump_visualizations:
                self.env_video_frames.append(self.get_observation())
        raise Exception()


    def set_grasp(self, grasp):
        if type(grasp) == bool:
            self.grasp_states = [grasp] * len(self.grasp_states)
        elif len(grasp) == len(self.grasp_states):
            self.grasp_states = grasp
        else:
            raise Exception()


    def setup_ray(self, id):
        self.ray_handle = {"val": id}


    def get_observation(self, particle_view=False):
        color_img, depth_img, _ = pyflex.render(uv=False)
        color_img = np.flip(color_img.reshape(self.camera_config['cam_size'] + [4]), 0)[:, :, :3]
        depth_img = np.flip(depth_img.reshape(self.camera_config['cam_size']), 0)
        observation = {
            'color_img': color_img,
            'depth_img': depth_img
        }
        if particle_view:
            pyflex.set_camera_params(self.particle_view_camera_config)
            color_img, depth_img, _ = pyflex.render(uv=False)
            color_img = np.flip(color_img.reshape(self.camera_config['cam_size'] + [4]), 0)[:, :, :3]
            depth_img = np.flip(depth_img.reshape(self.camera_config['cam_size']), 0)
            pyflex.set_camera_params(self.camera_config)
            
            observation['particle_view_color_img'] = color_img
            observation['particle_view_depth_img'] = depth_img
            
        return observation


    def get_random_grasping(self, num_pair=100):
        positions = pyflex.get_positions().reshape(-1, 4)[:self.cloth_particle_num, :3]
        idx = np.random.choice(self.cloth_particle_num, 2 * num_pair)
        select_positions = positions[idx, :].reshape(2, num_pair, 3)
        distance = np.linalg.norm(select_positions[0] - select_positions[1], axis=1)
        pair_idx = np.argmax(distance)
        p1 = select_positions[0, pair_idx]
        p2 = select_positions[1, pair_idx]
        if p1[0] < p2[0]:
            p1, p2 = p2, p1
        return p1, p2

    def get_random_pick_and_place(self, num_pair=100):
        positions = pyflex.get_positions().reshape(-1, 4)[:self.cloth_particle_num, :3]
        idx = np.random.choice(self.cloth_particle_num)
        p1 = positions[idx]
        direction = np.random.rand() * 2 * np.pi
        distance = np.random.rand() * 0.1 + 0.1
        p2 = p1 + distance * np.array([np.cos(direction), 0, np.sin(direction)])
        return p1, p2
