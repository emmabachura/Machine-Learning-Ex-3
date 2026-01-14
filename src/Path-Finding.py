import random
from typing import List, Tuple
from dataclasses import dataclass

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
            self.width = len(self.grid[0]) if self.H > 0 else 0
            self.start_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "S"]
            self.target_cell = {(x, y) for y in range(self.height) for x in range(self.width) if self.grid[y][x] == "F"}
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
        return x, y, 0, 0

    def step(self, s: State, a: Action) -> Tuple[State, float]:
        x, y, vx, vy = s
        ax, ay = a

        vx2 = clip(vx + ax, -self.v_max, self.v_max)
        vy2 = clip(vy + ay, -self.v_max, self.v_max)

        if vx2 == 0 and vy2 == 0 and not self.is_start(x, y):
            vx2, vy2 = vx, vy

        x2 = x + vx2
        y2 = y + vy2

        path = get_path(x, y, x2, y2)

        for (px, py) in path[1:]:
            if self.is_target(px, py):
                return (px, py, vx2, vy2), self.reward

        for (px, py) in path[1:]:
            if self.is_wall(px, py) or not self.is_in_bounds(px, py):
                self.reset()


        return (x2, y2, vx2, vy2), self.reward




def main():
    # Desgin grid structure
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

    # start episode

    # print the rewards


    return


if __name__ == "__main__":
    main()