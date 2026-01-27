import random
from typing import List, Tuple, Dict
from dataclasses import dataclass
from statistics import mean
import time 
import argparse

#The action of the robot on the x and y-axis
Action = Tuple[int, int]  # (ax, ay)
#The current state of the robot
State = Tuple[int, int, int, int]  # (x, y, vx, vy)
#A list containing all 9 possible movement increments (3x3) 
Actions: List[Action] = [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]

def clip(v: int, lo: int, hi: int) -> int:
    """
    Limits a value 'v' to the inclusive range [lo, hi] (-2 to 2).
    """
    return max(lo, min(hi, v))

def get_path(x0, y0, x1, y1):
    """
    Determines all coordinates traversed during a single move.(Avoids skipping over walls)
    """
    path = [(x0, y0)]
    x, y = x0, y0

    #Vertical step
    dy = y1 - y0
    step_y = 1 if dy > 0 else -1
    for _ in range(abs(dy)):
        y += step_y
        path.append((x, y))

    #Horizontal step
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
        """
        Initializes grid dimensions and locates start/target cells.
        """
        self.height = len(self.grid)
        self.width = len(self.grid[0]) if self.height > 0 else 0
        self.start_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "S"] # list
        self.target_cell = {(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "T"} # set
        if not self.start_cells:
            raise ValueError("No start cells 'S' found")
        if not self.target_cell:
            raise ValueError("No target cell 'T' found")

    def is_in_bounds(self, x: int, y: int) -> bool:
        """
        Checks if a coordinate is inside the grid.
        """
        return 0 <= x < self.width and 0 <= y < self.height


    def is_wall(self, x: int, y: int) -> bool:
        """
        Checks if a coordinate is a wall. (Returns True if coordinate is #)
        """
        if not self.is_in_bounds(x, y):
            return False
        return self.grid[y][x] == "#"

    def is_start(self, x: int, y: int) -> bool:
        """
        Checks if a coordinate is the starting line. (Returns True if coordinate is S)
        """
        return self.is_in_bounds(x, y) and self.grid[y][x] == "S"

    def is_target(self, x: int, y: int) -> bool:
        """
        Checks if a coordinate is the target. (Returns True if coordinate is T)
        """
        return (x, y) in self.target_cell

    def reset(self) -> State:
        """
        Resets the robot to a random starting position and sets velocity to 0.
        """
        x, y = random.choice(self.start_cells)
        return (x, y, 0, 0)

    def step(self, s: State, a: Action) -> Tuple[State, float, bool]:
        """
        Executes a single step in the environment.
        """
        x, y, vx, vy = s
        ax, ay = a

        vx2 = clip(vx + ax, -self.v_max, self.v_max)
        vy2 = clip(vy + ay, -self.v_max, self.v_max)

        #Checks if velocity is zero and not at the starting line
        if vx2 == 0 and vy2 == 0 and not self.is_start(x, y):
            vx2, vy2 = vx, vy 

        x2 = x + vx2
        y2 = y + vy2

        path = get_path(x, y, x2, y2)

        #Checks if the robot is at the target
        for (px, py) in path[1:]:
            if self.is_target(px, py):
                return (px, py, vx2, vy2), self.reward, True

        #Checks if the robot hit a wall or is out of bounds
        for (px, py) in path[1:]:
            if self.is_wall(px, py) or not self.is_in_bounds(px, py):
                new_s = self.reset() 
                return new_s, self.reward, False


        return (x2, y2, vx2, vy2), self.reward, False
    

def eps_soft_pi_from_Q(Q: Dict[State, Dict[Action, float]], eps: float):
    """
    Returns a policy function that selects actions according to the epsilon-soft policy.(Exploitation vs Exploration)
    Q: Dict[State, Dict[Action, float]] - Action-value function
    eps: float - Exploration parameter
    """

    def pi(s: State) -> Action:
        # If the state has never been visited, choose randomly
        if s not in Q:
            return random.choice(Actions)
        
        a_star = max(Q[s], key = Q[s].get)

        # With probability (1 - eps), take the best action (a*)
        if random.random() < (1.0 - eps):
            return a_star
        
        # With probability eps, explore any of the 9 actions at random
        return random.choice(Actions) 
    
    return pi


def generate_episode(env: GridEnvironment, pi, max_steps: int = 3000):
    """
    Generates an episode following a given policy.
    """
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

def mc_control_on_policy_first_visit(env: GridEnvironment,num_episodes: int = 3000, eps: float = 0.1, gamma: float = 0.95, seed: int = 123, max_steps: int = 3000):
    """
    Implements on-policy first-visit MC control with epsilon-soft policies.
    """
    random.seed(seed)

    # Initialize: Policy starts random, Q-values and Returns history start empty
    pi = lambda s: random.choice(Actions)
    Q: Dict[State, Dict[Action, float]] = {}
    Returns: Dict[Tuple[State, Action], List[float]] = {}

    for i in range(num_episodes):

        current_eps = max(0.01, eps * (0.999 ** i))
        # Generate an episode following the current policy
        episode = generate_episode(env, pi, max_steps)

        G = 0.0
        visited = set() # to implement first-visit MC

        # Process the episode in reverse order (last to first) to calculate the return G
        for (s, a, r) in reversed(episode):
            G = gamma * G + r

            # If the state-action pair has already been visited, skip it
            if (s, a) in visited:
                continue
            visited.add((s, a))

            #Update Q-value for the state-action pair
            if s not in Q: 
                Q[s] = {a: 0.0 for a in Actions} 

            if (s, a) not in Returns:
                Returns[(s, a)] = []
            Returns[(s, a)].append(G)

            Q[s][a] = mean(Returns[(s, a)])

            #Improve the policy based on the new Q-values
            pi = eps_soft_pi_from_Q(Q, eps)

    return Q, pi

#------plotting-----
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

def greedy_action(Q, s):
    if s not in Q:
        return random.choice(Actions)
    return max(Q[s], key = Q[s].get)

def learned_path(env, Q, max_steps=3000):
    s = env.reset()
    segments = [[(s[0], s[1])]] 
    end = False

    for _ in range(max_steps):
        a = greedy_action(Q, s)
        s2, r, end = env.step(s, a)

        if env.is_start(s2[0], s2[1]) and not env.is_start(s[0], s[1]):
            segments.append([(s2[0], s2[1])])
        else:
            path_points = get_path(s[0], s[1], s2[0], s2[1])[1:]
            segments[-1].extend(path_points)

        s = s2
        if end:
            break

    return segments, end

def plot_grid_path(env, segments=None, title=None, show_legend=True):
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
    if segments:
        for seg in segments:
            if len(seg) < 2: continue
            
            xs = [p[0] + 0.5 for p in seg]
            ys = [p[1] + 0.5 for p in seg]
            
            
            ax.plot(xs, ys, linewidth=3, color="red", linestyle="-")

        first_x, first_y = segments[0][0]
        ax.scatter([first_x + 0.5], [first_y + 0.5], s=80, marker="o", color="red", zorder=5) 
        

        last_x, last_y = segments[-1][-1]
        ax.scatter([last_x + 0.5], [last_y + 0.5], s=80, marker="x", color="red", zorder=5)

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
    parser = argparse.ArgumentParser(description="Monte Carlo RL Grid World Trainer")
    
    parser.add_argument("--grid", type=int, default=1, choices=[1, 2, 3],
                        help="Choose grid: 1 (Simple), 2 (Zig-Zag), 3 (Bottleneck)")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of training episodes")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="Discount factor (prevents loops)")
    parser.add_argument("--max_steps", type=int, default=3000, 
                        help="Maximum steps per episode")
    parser.add_argument("--epsilon", type=float, default=0.2, 
                        help="Starting exploration rate")
    parser.add_argument("--seed", type=int, default=123, 
                        help="Seed for random number generator to ensure reproducibility")
    

    args = parser.parse_args()

    grid_simple = [
        "############", "#....#.....#", "#....#.....#", "#....#.....#", "#..........#",
        "#......T...#", "#....#.....#", "#....####..#", "#..........#", "#..........#",
        "#..........#", "#..........#", "#........###", "#..........#", "#..........#",
        "#..........#", "#..........#", "#SSSSSSSSSS#", "############"
    ]

    grid_medium = [
        "############", "#T.........#", "#..........#", "######.....#", "#..........#",
        "#..........#", "#......#####", "#..........#", "#..........#", "######.....#",
        "#..........#", "#..........#", "#......#####", "#..........#", "#..........#",
        "#..........#", "#..........#", "#SSSSSSSSSS#", "############"
    ]

    grid_complex = [
        "############", "#....T.....#", "#..........#", "#..........#", "#..........#",
        "######.#####", "#..........#", "#..........#", "#..........#", "#..........#",
        "########.###", "#..........#", "#..........#", "#..........#", "#..........#",
        "#..........#", "#..........#", "#SSSSSSSSSS#", "############"
    ]

    grid_map = {
        1: (grid_simple, "Simple Grid"),
        2: (grid_medium, "Zig-Zag Grid"),
        3: (grid_complex, "Bottleneck Grid")
    }
    
    selected_grid, name = grid_map[args.grid]

    print(f"\n--- Training {name} ---")
    print(f"Parameters: Episodes={args.episodes}, Gamma={args.gamma}, Starting Epsilon={args.epsilon}, Max Steps={args.max_steps} , Seed={args.seed}")
    env = GridEnvironment(selected_grid)

    start_time = time.time()
    
    Q, pi = mc_control_on_policy_first_visit(
        env, 
        num_episodes=args.episodes, 
        gamma=args.gamma, 
        eps=args.epsilon,
        seed=args.seed,
        max_steps=args.max_steps
    )
    
    duration = time.time() - start_time

    print(f"Training Time: {duration:.2f} seconds")
    print(f"Number of visited states: {len(Q)}")

    cells, end = learned_path(env, Q, max_steps=args.max_steps)
    plot_grid_path(env, cells, title=f"{name} (Solved: {end})")

if __name__ == "__main__":
    main()




