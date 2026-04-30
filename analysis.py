import heapq
import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt


# Benchmark settings
SIZES = list(range(5, 101, 1))
TRIALS_PER_SIZE = 30
TIME_REPEATS = 15
OBSTACLE_PROB = 0.20
ALLOW_DIAGONAL = True
RANDOM_SEED = 42



# GRID GENERATION
def generate_grid(size: int, obstacle_prob: float) -> np.ndarray:
    """
    Generate a random square grid.

    1 = free cell
    0 = obstacle
    """
    grid = np.ones((size, size), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            if random.random() < obstacle_prob:
                grid[y, x] = 0

    # Keep start and goal free for every generated map
    grid[0, 0] = 1
    grid[size - 1, size - 1] = 1
    return grid


def is_valid(point: tuple[int, int], grid: np.ndarray) -> bool:
    x, y = point
    return 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and grid[y, x] == 1


def euclidean(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Euclidean distance between two grid points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors(
    node: tuple[int, int],
    grid: np.ndarray,
    allow_diagonal: bool
) -> list[tuple[int, int]]:
    """Return valid neighboring cells for the selected movement model."""
    x, y = node

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if allow_diagonal:
        directions += [(1, 1), (-1, -1), (1, -1), (-1, 1)]

    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy

        if not (0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]):
            continue
        if grid[ny, nx] == 0:
            continue

        # Prevent diagonal corner-cutting
        if allow_diagonal and dx != 0 and dy != 0:
            if grid[y, nx] == 0 or grid[ny, x] == 0:
                continue

        neighbors.append((nx, ny))

    return neighbors


def reconstruct_path(
    came_from: dict,
    start: tuple[int, int],
    goal: tuple[int, int]
) -> list[tuple[int, int]]:
    """Reconstruct the path from start to goal using parent links."""
    if goal not in came_from and goal != start:
        return []

    path = [goal]
    current = goal

    while current != start:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


# A* ALGORITHM
def astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int]
) -> tuple[list[tuple[int, int]], int]:
    """Run A* search and return the path and number of processed nodes."""
    open_heap = []
    heapq.heappush(open_heap, (euclidean(start, goal), 0.0, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    iterations = 0

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        iterations += 1

        if current == goal:
            return reconstruct_path(came_from, start, goal), iterations

        for neighbor in get_neighbors(current, grid, ALLOW_DIAGONAL):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + euclidean(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_neighbor = tentative_g + euclidean(neighbor, goal)
                heapq.heappush(open_heap, (f_neighbor, tentative_g, neighbor))

    return [], iterations


# DIJKSTRA ALGORITHM
def dijkstra(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int]
) -> tuple[list[tuple[int, int]], int]:
    """Run Dijkstra's algorithm and return the path and number of processed nodes."""
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    iterations = 0

    while open_heap:
        current_cost, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        closed.add(current)
        iterations += 1

        if current == goal:
            return reconstruct_path(came_from, start, goal), iterations

        for neighbor in get_neighbors(current, grid, ALLOW_DIAGONAL):
            if neighbor in closed:
                continue

            tentative_g = g_score[current] + euclidean(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative_g, neighbor))

    return [], iterations


# TIMING HELPER
def average_runtime(func, grid, start, goal, repeats=10):
    """
    Measure median runtime for one algorithm on one grid.

    A warm-up call is used before timing to reduce random measurement noise.
    """
    func(grid, start, goal)  # warm-up

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(grid, start, goal)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return float(np.median(times))


# BENCHMARK
def benchmark():
    """Run A* and Dijkstra on random grids of increasing size."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    astar_iter_means = []
    dijkstra_iter_means = []

    astar_time_means = []
    dijkstra_time_means = []

    for size in SIZES:
        astar_counts = []
        dijkstra_counts = []

        astar_times = []
        dijkstra_times = []

        attempts = 0
        max_attempts = TRIALS_PER_SIZE * 80

        while len(astar_counts) < TRIALS_PER_SIZE and attempts < max_attempts:
            attempts += 1

            grid = generate_grid(size=size, obstacle_prob=OBSTACLE_PROB)
            start = (0, 0)
            goal = (size - 1, size - 1)

            if not is_valid(start, grid) or not is_valid(goal, grid):
                continue

            # Use the same grid for both algorithms
            path_a, it_a = astar(grid, start, goal)
            path_d, it_d = dijkstra(grid, start, goal)

            # Ignore unsolved maps so both algorithms are compared fairly
            if not path_a or not path_d:
                continue

            astar_counts.append(it_a)
            dijkstra_counts.append(it_d)

            # More stable timing on the same grid
            astar_times.append(average_runtime(astar, grid, start, goal, TIME_REPEATS))
            dijkstra_times.append(average_runtime(dijkstra, grid, start, goal, TIME_REPEATS))

        if len(astar_counts) == 0:
            astar_iter_means.append(np.nan)
            dijkstra_iter_means.append(np.nan)
            astar_time_means.append(np.nan)
            dijkstra_time_means.append(np.nan)

            print(f"Size {size:3d} | no valid paths found")
        else:
            mean_it_a = float(np.mean(astar_counts))
            mean_it_d = float(np.mean(dijkstra_counts))

            # Median across trials is more robust for timing
            mean_t_a = float(np.median(astar_times))
            mean_t_d = float(np.median(dijkstra_times))

            astar_iter_means.append(mean_it_a)
            dijkstra_iter_means.append(mean_it_d)
            astar_time_means.append(mean_t_a)
            dijkstra_time_means.append(mean_t_d)

            print(
                f"Size {size:3d} | "
                f"A*: iter={mean_it_a:8.2f}, time={mean_t_a:.6f}s | "
                f"Dijkstra: iter={mean_it_d:8.2f}, time={mean_t_d:.6f}s | "
                f"trials: {len(astar_counts)}"
            )

    return (
        astar_iter_means,
        dijkstra_iter_means,
        astar_time_means,
        dijkstra_time_means,
    )


# PLOTTING HELPERS
def get_valid_series(x_values, y1, y2):
    """Remove NaN values before plotting benchmark results."""
    x = np.array(x_values, dtype=float)
    y1 = np.array(y1, dtype=float)
    y2 = np.array(y2, dtype=float)

    valid_mask = (~np.isnan(y1)) & (~np.isnan(y2))
    return x[valid_mask], y1[valid_mask], y2[valid_mask]


def plot_iteration_results(astar_means, dijkstra_means):
    """Plot the average number of processed nodes."""
    x, y_a, y_d = get_valid_series(SIZES, astar_means, dijkstra_means)

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y_a, label="A*", linewidth=1.8)
    plt.plot(x, y_d, label="Dijkstra", linewidth=1.8)

    plt.title("A* vs Dijkstra: Iteration Number")
    plt.xlabel("Map Size")
    plt.ylabel("Iteration Number")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_runtime_results(astar_times, dijkstra_times):
    """Plot median runtime for both algorithms."""
    x, y_a, y_d = get_valid_series(SIZES, astar_times, dijkstra_times)

    # Convert seconds to milliseconds
    y_a_ms = y_a * 1000.0
    y_d_ms = y_d * 1000.0

    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y_a_ms, label="A*", linewidth=1.8)
    plt.plot(x, y_d_ms, label="Dijkstra", linewidth=1.8)

    plt.title("A* vs Dijkstra: Runtime")
    plt.xlabel("Map Size")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    (
        astar_iter_means,
        dijkstra_iter_means,
        astar_time_means,
        dijkstra_time_means,
    ) = benchmark()

    plot_iteration_results(astar_iter_means, dijkstra_iter_means)
    plot_runtime_results(astar_time_means, dijkstra_time_means)