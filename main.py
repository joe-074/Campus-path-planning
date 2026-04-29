"""
Compare A* and Dijkstra on a prepared campus map.

The script reads a binary walkability mask and the corresponding satellite
image, builds a grid map, runs both planners from the same start point to the
same goal point, and visualizes:

1. the found paths on the binary grid;
2. the found paths on the satellite image;
3. the visited-node heatmaps.

Coordinates are stored as (x, y), while NumPy arrays are indexed as [y, x].
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq
import time


# Select which prepared map to use.
map_id = 4 # change this to 2, 3, 4, ... for other maps

# Start and goal coordinates for the selected map.
# Coordinates must be chosen on white/free areas of the binary mask.
start = (502, 394)
goal  = (1195, 682)



# Input files
mask_path = f"masks/mask{map_id}.png"
map_path  = f"maps/map{map_id}.png"


# Load the mask and satellite image
mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
map_img = cv2.imread(map_path)

if mask is None:
    print(f"Error: {mask_path} not found")
    exit()

if map_img is None:
    print(f"Error: {map_path} not found")
    exit()

map_rgb = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)


# Convert BGR image from OpenCV to RGB for Matplotlib
# Direct threshold: white -> 1 (free), black -> 0 (obstacle)
grid = (mask > 127).astype(np.uint8)



# VALIDATION
def is_valid_point(point, grid):
    """Check if a point is inside the map and located on a free cell."""
    x, y = point
    if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        return grid[y, x] == 1
    return False

# Check start and goal before running the algorithms
if not is_valid_point(start, grid):
    print(f"Error: start point {start} is not on a free cell.")
    exit()

if not is_valid_point(goal, grid):
    print(f"Error: goal point {goal} is not on a free cell.")
    exit()



# SHARED UTILITIES
def get_neighbors(node, grid):
    """Return valid neighboring cells using 8-connected movement."""
    x, y = node
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, -1), (1, -1), (-1, 1)
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
            if grid[ny, nx] == 1:
                neighbors.append((nx, ny))
    return neighbors

def euclidean(a, b):
    """Euclidean distance between two points."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def reconstruct_path(came_from, start, goal):
    """Reconstruct path from the parent dictionary."""
    path = []
    node = goal
    while node in came_from:
        path.append(node)
        node = came_from[node]
    path.append(start)
    path.reverse()
    return path

def path_length(path):
    """Calculate total path length in pixels."""
    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean(path[i], path[i + 1])
    return total



# A* ALGORITHM

def astar(grid, start, goal):
    """Find a shortest path using A* with Euclidean heuristic."""
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from     = {}
    g_score       = {start: 0.0}
    visited_count = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        visited_count += 1

        if current == goal:
            return reconstruct_path(came_from, start, goal), visited_count

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + euclidean(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_score             = tentative_g + euclidean(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return [], visited_count



# DIJKSTRA ALGORITHM
def dijkstra(grid, start, goal):
    """Find a shortest path using Dijkstra's algorithm."""
    open_set = []
    heapq.heappush(open_set, (0.0, start))

    came_from     = {}
    g_score       = {start: 0.0}
    visited       = set()
    visited_count = 0

    while open_set:
        cost, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)
        visited_count += 1

        if current == goal:
            return reconstruct_path(came_from, start, goal), visited_count

        for neighbor in get_neighbors(current, grid):
            if neighbor in visited:
                continue

            tentative_g = g_score[current] + euclidean(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))

    return [], visited_count


# Run both algorithms
print(f"\n{'='*45}")
print(f"  Map ID : {map_id}")
print(f"  Start  : {start}")
print(f"  Goal   : {goal}")
print(f"{'='*45}")

t0 = time.perf_counter()
path_astar, visited_astar = astar(grid, start, goal)
t1 = time.perf_counter()
time_astar = t1 - t0

t0 = time.perf_counter()
path_dijkstra, visited_dijkstra = dijkstra(grid, start, goal)
t1 = time.perf_counter()
time_dijkstra = t1 - t0

# Stop if one of the algorithms failed
if not path_astar:
    print("A*: No path found.")
    exit()

if not path_dijkstra:
    print("Dijkstra: No path found.")
    exit()


# PRINT COMPARISON TABLE
# Numerical comparison
len_astar    = path_length(path_astar)
len_dijkstra = path_length(path_dijkstra)

print(f"\n{'Metric':<25} {'A*':>15} {'Dijkstra':>15}")
print(f"{'-'*55}")
print(f"{'Path points':<25} {len(path_astar):>15} {len(path_dijkstra):>15}")
print(f"{'Path length (px)':<25} {len_astar:>15.2f} {len_dijkstra:>15.2f}")
print(f"{'Visited nodes':<25} {visited_astar:>15} {visited_dijkstra:>15}")
print(f"{'Runtime (s)':<25} {time_astar:>15.4f} {time_dijkstra:>15.4f}")
print(f"{'-'*55}")

ratio_visited = visited_dijkstra / visited_astar if visited_astar > 0 else 0
ratio_time    = time_dijkstra / time_astar if time_astar > 0 else 0
print(f"\nDijkstra visited {ratio_visited:.1f}x more nodes than A*")
print(f"Dijkstra took    {ratio_time:.1f}x longer than A*")
print(f"Path length difference: {abs(len_astar - len_dijkstra):.4f} px (both optimal)")


# VISUALIZE — GRID WITH BOTH PATHS
# Show paths on the binary grid
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"Path Planning Comparison — Map {map_id}", fontsize=14, fontweight='bold')

for ax, path, title, color in zip(
    axes,
    [path_astar, path_dijkstra],
    [f"A*  |  visited: {visited_astar}  |  t: {time_astar:.4f}s",
     f"Dijkstra  |  visited: {visited_dijkstra}  |  t: {time_dijkstra:.4f}s"],
    ['red', 'blue']
):
    ax.imshow(grid, cmap='gray')
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=1.5, label='Path')
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0],  goal[1],  'mo', markersize=10, label='Goal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()


# VISUALIZE — ORIGINAL MAP WITH BOTH PATHS
# Show paths on the satellite map
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))
fig2.suptitle(f"Path on Satellite Map — Map {map_id}", fontsize=14, fontweight='bold')

for ax, path, title, color in zip(
    axes2,
    [path_astar, path_dijkstra],
    [f"A*  |  length: {len_astar:.1f} px",
     f"Dijkstra  |  length: {len_dijkstra:.1f} px"],
    ['red', 'blue']
):
    ax.imshow(map_rgb)
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, color=color, linewidth=2.0, label='Path')
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0],  goal[1],  'mo', markersize=10, label='Goal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()



# VISUALIZE — VISITED NODES HEATMAP
def astar_heatmap(grid, start, goal):
    """Create a visit-order heatmap for A*."""
    open_set  = []
    heapq.heappush(open_set, (0.0, start))
    g_score   = {start: 0.0}
    visit_map = np.zeros(grid.shape, dtype=np.int32)
    counter   = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        x, y = current
        if visit_map[y, x] == 0:
            counter += 1
            visit_map[y, x] = counter
        if current == goal:
            break
        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + euclidean(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f = tentative_g + euclidean(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return visit_map

def dijkstra_heatmap(grid, start, goal):
    """Create a visit-order heatmap for Dijkstra."""
    open_set  = []
    heapq.heappush(open_set, (0.0, start))
    g_score   = {start: 0.0}
    visited   = set()
    visit_map = np.zeros(grid.shape, dtype=np.int32)
    counter   = 0

    while open_set:
        cost, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        counter += 1
        visit_map[y, x] = counter
        if current == goal:
            break
        for neighbor in get_neighbors(current, grid):
            if neighbor in visited:
                continue
            tentative_g = g_score[current] + euclidean(current, neighbor)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
    return visit_map

# Build heatmaps
heatmap_astar    = astar_heatmap(grid, start, goal)
heatmap_dijkstra = dijkstra_heatmap(grid, start, goal)

# Show visited-node heatmaps
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 8))
fig3.suptitle(f"Visited Nodes Heatmap — Map {map_id}", fontsize=14, fontweight='bold')

for ax, hmap, path, title in zip(
    axes3,
    [heatmap_astar, heatmap_dijkstra],
    [path_astar, path_dijkstra],
    [f"A*  |  visited: {visited_astar}",
     f"Dijkstra  |  visited: {visited_dijkstra}"]
):
    display = np.zeros((*grid.shape, 3), dtype=np.uint8)
    display[grid == 1] = [240, 240, 240]
    display[grid == 0] = [30,  30,  30]

    ax.imshow(display)
    masked = np.ma.masked_where(hmap == 0, hmap)
    ax.imshow(masked, cmap='YlOrRd', alpha=0.7,
              vmin=1, vmax=hmap.max() if hmap.max() > 0 else 1)

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, 'b-', linewidth=1.5, label='Final path')
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0],  goal[1],  'mo', markersize=10, label='Goal')
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.show()