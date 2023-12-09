# Import nessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path


# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")

class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardmode = is_hardmode

        # Generate random cliff positions
        if self.is_hardmode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                    (state not in self.cliff_positions)
                    and (state != self.start_state)
                    and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, -100, False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        return [(1 / 3, new_state, -1, is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}   must be in [0, 1, 2, 3]")

        if self.is_hardmode:
            match action:
                case 0:
                    action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 1:
                    action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                case 2:
                    action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 3:
                    action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking - Edited by Audrina & Kian")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


# Create an environment
env = CliffWalking(render_mode="human")
observation, info = env.reset(seed=30)


class MDP:
    def __init__(self, max_iterations, discount_factor, cliff_positions):
        self.V = np.zeros((4, 12))
        self.Q = np.zeros((4, 12, 4))
        self.pi = np.zeros((4, 12), dtype=int)
        self.max_iter = max_iterations
        self.gama = discount_factor
        self.cliffs = cliff_positions
        self.max_converged = 100

        self.action_reward = -1
        self.goal_reward = 4000
        self.cliff_reward = -100


    def update_QV(self):
        di = [-1, 0, 1, 0]
        dj = [0, 1, 0, -1]
        # update Q
        for i in range(4):
            for j in range(12):
                for k in range(4):

                    num = 0
                    for action in [UP, RIGHT, DOWN, LEFT]:
                        if k != action and action%2 == k%2:
                            continue
                        next_state = (i+di[action], j+dj[action])
                        if next_state[0] >= 4 or next_state[1] >= 12 or next_state[0] < 0 or next_state[1] < 0:
                            num += 1
                    prob = 1/3
                    if (num == 2):
                        prob = 1
                    elif (num == 1):
                        prob = 1/2

                    self.Q[i, j, k] = 0
                    for action in [UP, RIGHT, DOWN, LEFT]:
                        if k != action and action%2 == k%2:
                            continue
                        next_state = (i+di[action], j+dj[action])
                        if next_state[0] >= 4 or next_state[1] >= 12 or next_state[0] < 0 or next_state[1] < 0:
                            continue
                        self.Q[i, j, k] += prob * (self.action_reward + self.gama * self.V[next_state[0], next_state[1]])

        # update V
        for i in range(4):
            for j in range(12):
                if i == 3 and j == 11:
                    continue
                if (i, j) in self.cliffs:
                    continue
                self.V[i, j] = max(self.Q[i, j])


    def update_pi(self):
        for i in range(4):
            for j in range(12):
                for action in [UP, RIGHT, DOWN, LEFT]:
                    if self.Q[i, j, action] == self.V[i, j]:
                        self.pi[i, j] = action


    def run(self):
        self.V[3, 11] = self.goal_reward
        for ci, cj in self.cliffs:
            self.V[ci, cj] = self.cliff_reward

        iter_cnt = 0
        converged = 0
        while iter_cnt < self.max_iter or converged < self.max_converged:
            tmp_V = np.array(self.V, copy = True)
            self.update_QV()
            if self.V.all() == tmp_V.all():
                converged += 1
            else:
                converged = 0
            iter_cnt += 1

        print(iter_cnt)
        self.update_pi()


    def policy(self, state):
        return self.pi[state[0], state[1]]


    def print_pi(self):
        direction = {0: '^', 1: '>', 2: 'V', 3: '<'}
        for i in range(4):
            for j in range(12):
                print(direction[self.
                      pi[i, j]], end=" ")
            print()
        print()


    def print_v(self):
        for i in range(4):
            for j in range(12):
                print(round(self.V[i, j], 2), end="\t")
            print()
        print()


# Define the maximum number of iterations
max_iter_number = 1000
iter_of_mdp = 1000
gama = 0.99
current_state = (3, 0)
wins = 0

mdp = MDP(iter_of_mdp, gama, env.cliff_positions)
mdp.run()
mdp.print_pi()
mdp.print_v()

for __ in range(max_iter_number):
    try:
        # Choose an action
        action = mdp.policy(current_state)

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated, info = env.step(action)
        current_state = (next_state//12, next_state%12)

        if done or truncated:
            observatistopon, info = env.reset()
            current_state = (3, 0)
        wins += done

    except KeyboardInterrupt:
        break

print("Wins =", wins)
env.close()
exit(0)

