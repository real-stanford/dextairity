import numpy as np
from sim_env import SimEnv


def test_env():
    seed = 11
    np.random.seed(seed)
    env = SimEnv(gui=False, wind_life_time=60)
    # env.camera_config['render_type'] = ['points']
    env.reset()
    p1, p2 = env.get_random_grasping()
    env.lift_and_stretch_primitive(p1, p2, lifting_height=0.12)

    position = [0, 0.03, 0.45]
    for rx in [-30, 0, 30]:
        orientation = [rx / 180 * np.pi, 0, -95 / 180 * np.pi]
        cover_area, observation = env.blow(position, orientation, num_layer=2, alpha=5.0, velocity=5, mass=0.1, step_num=150)

    cover_area, observation = env.place_cloth()
    print('coverage = ', cover_area / env.max_cover_area)


if __name__=='__main__':
    test_env()