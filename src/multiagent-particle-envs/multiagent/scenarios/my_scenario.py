import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

import ipdb


from scipy.special import huber

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15 # 0.15
            agent.max_speed = 0.2   # temp
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])


        # set random initial states without collisions
        has_collision = True
        while has_collision:
            for agent in world.agents:
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)

            # Check if initial position violates constraints
            has_collision = False
            for i in range(len(world.agents)):
                for j in range(i + 1, len(world.agents), 1):
                    if self.is_reasonable(world.agents[i],world.agents[j]):
                        has_collision = True


        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # Do the same for landmarks
        has_collision = True
        while has_collision:
            for i, landmark in enumerate(world.landmarks):
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

            # Check if initial position violates constraints
            has_collision = False
            for i in range(len(world.landmarks)):
                for j in range(i + 1, len(world.landmarks), 1):
                    if self.is_reasonable(world.landmarks[i],world.landmarks[j]):
                        has_collision = True

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_reasonable(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size + 0.21
        return True if dist < dist_min else False

    def reward(self, agent, world, action_n):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0

        # Get Agent index
        idx = int(agent.name.split(' ')[1])

        target_pos = world.landmarks[idx].state.p_pos
        agent_pos  = agent.state.p_pos

        dist   = np.linalg.norm(target_pos - agent_pos,1)
        action_norm = np.linalg.norm(action_n[idx])

        rew = -dist #- 0.2*action_norm

        if (dist < 0.02):
            rew += 5

        '''
        for i, l in enumerate(world.landmarks):
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]

            dists = np.linalg.norm(l.state.p_pos - world.agents[i].state.p_pos)

            #rew -= min(dists)
            # add a penalty on actions for stability
            rew  -= dists[i]
            rew  -= 0.25*np.linalg.norm(action_n[i], 2)
        '''

        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 0.2
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)
        #return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos)

    def constraints(self,agent, world):
        # Constraint Type 1: Collisions with other robots
        other_agents = [a for a in world.agents if a is not agent]

        collision_signals = np.zeros(len(world.agents) - 1)
        for i, other in enumerate(other_agents):
            collision_signals[i] = np.linalg.norm(other.state.p_pos - agent.state.p_pos)

        # Constraint Type 2: Obstacles
        # TODO
        return collision_signals



