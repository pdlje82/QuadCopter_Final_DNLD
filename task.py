import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,
                 init_pose=None,
                 init_velocities=None,
                 init_angle_velocities=None,
                 runtime=5.,
                 target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        print('')
        print('init_pose: ', init_pose)
        print('init_velocities: ', init_velocities)
        print('target_pos: ', target_pos)
        print('')
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 130             # originally 0
        self.action_high = 730          # originally 900
        self.action_size = 1            # 4 rotor speeds
        self.alpha = 1e-4               # learning rate for actor / policy update
        self.beta = 1e-3                # learning rate for critic / value update

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, done):
        # reward function for the takeoff-task
        #rewards depending on position
        if (self.sim.pose[2] < self.target_pos[2] + 1) and (self.sim.v[2] < 0):
            #print('test')
            reward = -.5
        elif (self.sim.pose[2] > self.target_pos[2] - 1) and (self.sim.v[2] > 0):
            reward = -.5
        else:
            reward = 1. - .125 * (abs(self.target_pos[2] - self.sim.pose[2]))
            reward = np.clip(reward, -1, 1)
        # penalize crash
        if done and self.sim.time < self.sim.runtime:
            reward += -1
        #print(reward)
        return reward


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):

            # use * np.ones(4) only if working with 1 rotor speed for all 4 rotors
            done = self.sim.next_timestep(rotor_speeds * np.ones(4))  # update the sim pose and velocities

            reward = self.get_reward(done)
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state