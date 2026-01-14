import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from statistics import mean

Action = Tuple[int, int]
State = Tuple[int, int, int, int]
Actions: List[Action] = [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]

def clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def get_path(x0, y0, x1, y1):
    path = [(x0, y0)]
    x, y = x0, y0

    dy = y1 - y0
    step_y = 1 if dy > 0 else -1
    for _ in range(abs(dy)):
        y += step_y
        path.append((x, y))

    dx = x1 - x0
    step_x = 1 if dx > 0 else -1
    for _ in range(abs(dx)):
        x += step_x
        path.append((x, y))

    return path


@dataclass
class GridEnvironment:
    grid: List[str]
    v_max: int = 2
    reward: int = -1

    def __post_init__(self):
            self.height = len(self.grid)
            self.width = len(self.grid[0]) if self.height > 0 else 0
            self.start_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "S"] # list
            self.target_cell = {(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "T"} # set
            if not self.start_cells:
                raise ValueError("No start cells 'S' found")
            if not self.target_cell:
                raise ValueError("No target cell 'T' found")

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        if not self.is_in_bounds(x, y):
            return False
        return self.grid[y][x] != "#"

    def is_start(self, x: int, y: int) -> bool:
        return self.is_in_bounds(x, y) and self.grid[y][x] == "S"

    def is_target(self, x: int, y: int) -> bool:
        return (x, y) in self.target_cell

    def reset(self) -> State:
        x, y = random.choice(self.start_cells)
        return (x, y, 0, 0)

    def step(self, s: State, a: Action) -> Tuple[State, float]:
        x, y, vx, vy = s
        ax, ay = a

        vx2 = clip(vx + ax, -self.v_max, self.v_max)
        vy2 = clip(vy + ay, -self.v_max, self.v_max)

        if vx2 == 0 and vy2 == 0 and not self.is_start(x, y):
            vx2, vy2 = vx, vy # keep previous velocity, v. comps. cannot both be zero except at the starting line

        x2 = x + vx2
        y2 = y + vy2

        path = get_path(x, y, x2, y2)

        for (px, py) in path[1:]:
            if self.is_target(px, py):
                return (px, py, vx2, vy2), self.reward, True

        for (px, py) in path[1:]:
            if self.is_wall(px, py) or not self.is_in_bounds(px, py):
                self.reset(), self.reward, False


        return (x2, y2, vx2, vy2), self.reward, False
    
# --- on-policy first-visit MC control (eps-soft policies), estimates pi=pi* ---
# c.f. lecture

# Q...action-value fct
# pi...policy

def eps_soft_pi_from_Q(Q: Dict[State, Dict[Action, float]], eps: float):
    # returns pi

    def pi(s: State) -> Action:
        if s not in Q:
            return random.choice(Actions)
        
        a_star = max(Q[s], key = Q[s].get)

        if random.random() < (1.0 - eps):
            return a_star
        
        return random.choice(Actions) 
        # a* is returned with probability 1 - eps + eps/len(Actions)
        # all other actions with probability eps/len(Actions)
    
    return pi


def generate_episode_following_pi(env: GridEnvironment, pi, 
                                  max_steps: int = 3000):
    
    s = env.reset()
    episode: List[Tuple[State, Action, int]] = []

    for _ in range(max_steps):
        a = pi(s)
        s2, r, end = env.step(s,a)
        episode.append((s,a,r))
        s = s2

        if end:
            break

    return episode

def mc_control_on_policy_first_visit(env: GridEnvironment,
                                     num_episodes: int = 3000, 
                                     eps: float = 0.1, gamma: float = 1.0, seed: int = 123):
    random.seed(seed)

    # initialize:
    pi = lambda s: random.choice(Actions)
    Q: Dict[State, Dict[Action, float]] = {}
    Returns: Dict[Tuple[State, Action], List[float]] = {}

    # repeat forever (for each epsiode):
    for _ in range(num_episodes):
        episode = generate_episode_following_pi(env, pi)

        G = 0.0
        visited = set() # to implement first-visit MC

        for (s, a, r) in reversed(episode):
            G = gamma * G + r

            if (s, a) in visited:
                continue

            visited.add((s, a))

            if s not in Q: # existence of Q
                Q[s] = {a: 0.0 for a in Actions} # # initialize with 0.0

            if (s, a) not in Returns:
                Returns[(s, a)] = []
            Returns[(s, a)].append(G)

            Q[s][a] = mean(Returns[(s, a)])

            pi = eps_soft_pi_from_Q(Q, eps)

    return Q, pi

#------plotting-----
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

def greedy_action(Q, s):
    if s not in Q:
        return random.choice(Actions)
    return max(Q[s], key = Q[s].get)

def learned_path(env, Q, max_steps = 2000):
    s = env.reset()
    cells = [(s[0], s[1])]
    end = False

    for _ in range(max_steps):
        a = greedy_action(Q, s)
        s2, r, end = env.step(s, a)

        cells.extend(get_path(s[0], s[1], s2[0], s2[1])[1:])

        s = s2

        if end:
            break

    return cells, end

def plot_grid_path(env, cells=None, title=None, show_legend=True):
    fig, ax = plt.subplots(figsize=(8, 6))

    for y in range(env.height):
        for x in range(env.width):
            ch = env.grid[y][x]
            if ch == "#":
                face = "black"
            elif ch == "T":
                face = "green"
            elif ch == "S":
                face = "blue"
            else:
                face = "white"

            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=face, edgecolor="none"))

    # lines
    for x in range(env.width + 1):
        ax.plot([x, x], [0, env.height], color="black", linewidth=1)
    for y in range(env.height + 1):
        ax.plot([0, env.width], [y, y], color="black", linewidth=1)

    ax.add_patch(Rectangle((0, 0), env.width, env.height, fill=False,
                            edgecolor="black", linewidth=6))

    # path
    if cells:
        xs = [p[0] + 0.5 for p in cells]
        ys = [p[1] + 0.5 for p in cells]
        ax.plot(xs, ys, linewidth=3, color="red")
        ax.scatter([xs[0]], [ys[0]], s=80, marker="o", color="red") # start
        ax.scatter([xs[-1]], [ys[-1]], s=80, marker="x", color="red") # end

    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if title:
        ax.set_title(title)

    if show_legend:
        legend_items = [
            Patch(facecolor="black", edgecolor="black", label="Obstacle"),
            Patch(facecolor="green", edgecolor="black", label="Target"),
            Patch(facecolor="blue", edgecolor="black", label="Starting line"),
        ]
        ax.legend(handles=legend_items, loc="center left", fontsize=12, 
                  bbox_to_anchor=(1.05, 0.5), frameon = False)

    plt.tight_layout()
    plt.show()


def main():
    # Desgin grid structure (#...wall/obstacle, T...target, S...starting line)
    grid = [
        "####################",
        "#.......#..........#",
        "#.......#......T...#",
        "#.......#..........#",
        "#.......#..........#",
        "#.......########...#",
        "#SSSSSSSSSSSSSSSSSS#",
        "####################"
    ]
    # create environment
    env = GridEnvironment(grid)

    # start reinforcement learning
    Q, pi = mc_control_on_policy_first_visit(env, num_episodes = 5)

    # how much state space was explored
    print("Number of visited states:", len(Q))

    # plot path 
    cells, end = learned_path(env, Q, max_steps=2000)
    plot_grid_path(env, cells, end) 

if __name__ == "__main__":
    main()