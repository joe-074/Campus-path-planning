import cv2
import numpy as np
import matplotlib.pyplot as plt
import heapq


# Map and route settings
map_id = 4

start = (502, 394)
goal  = (1195, 682)


# Robot and obstacle behavior
SENSOR_RADIUS   = 12   # detection range (pixels)
ROBOT_STEP_SIZE = 2    # steps robot advances per iteration

# Obstacles are placed along the initial path:
# first one is temporary, second one is permanent.
OBSTACLE_FRAC = [0.35, 0.65]   # placement along initial path

# How many steps robot waits before deciding obstacle is permanent
PATIENCE_STEPS = 40
TEMPORARY_CLEAR_AFTER = 8   # obstacle starts drifting after 8 steps of detection
DRIFT_SPEED = 8   # faster drift to clear path quickly


# Input files
mask_path = f"masks/mask{map_id}.png"
map_path  = f"maps/map{map_id}.png"

mask    = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
map_img = cv2.imread(map_path)

if mask is None:
    print(f"Error: {mask_path} not found")
    exit()
if map_img is None:
    print(f"Error: {map_path} not found")
    exit()

map_rgb   = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)

# Base grid is kept unchanged during the simulation.
# Dynamic obstacles are added only to temporary copies of this grid.
base_grid = (mask > 127).astype(np.uint8)   # NEVER modified


def is_valid(pt, grid):
    """Check if a point is inside the map and on a free cell."""
    x, y = pt
    return (0 <= x < grid.shape[1] and
            0 <= y < grid.shape[0] and
            grid[y, x] == 1)

def euclidean(a, b):
    """Euclidean distance between two points."""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def get_neighbors(node, grid):
    """Return valid neighboring cells using 8-connected movement."""
    x, y = node
    result = []
    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),
                   (1,1),(-1,-1),(1,-1),(-1,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
            if grid[ny, nx] == 1:
                result.append((nx, ny))
    return result

def reconstruct_path(came_from, start, goal):
    """Reconstruct a path from the parent dictionary."""
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
    return sum(euclidean(path[i], path[i+1]) for i in range(len(path)-1))

def build_grid_with_obstacles(base_grid, active_obs, radius=4):
    """
    Build a temporary planning grid with active obstacles added.

    The original base_grid is not modified.
    """
    grid = base_grid.copy()
    for ox, oy in active_obs:
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                nx, ny = ox+dx, oy+dy
                if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
                    grid[ny, nx] = 0
    return grid

# A*
def astar(grid, start, goal):
    """Find a shortest path using A* with Euclidean heuristic."""
    open_set = []
    heapq.heappush(open_set, (0.0, start))
    came_from, g_score = {}, {start: 0.0}
    visited = 0
    while open_set:
        _, cur = heapq.heappop(open_set)
        visited += 1
        if cur == goal:
            return reconstruct_path(came_from, start, goal), visited
        for nb in get_neighbors(cur, grid):
            tg = g_score[cur] + euclidean(cur, nb)
            if nb not in g_score or tg < g_score[nb]:
                came_from[nb] = cur
                g_score[nb]   = tg
                heapq.heappush(open_set, (tg + euclidean(nb, goal), nb))
    return [], visited


# Check selected points before planning
if not is_valid(start, base_grid):
    print(f"Error: start {start} not on free cell.")
    exit()
if not is_valid(goal, base_grid):
    print(f"Error: goal {goal} not on free cell.")
    exit()


# Build the initial path before adding dynamic obstacles
print(f"\n{'='*54}")
print(f"  Map {map_id}  |  Start {start}  |  Goal {goal}")
print(f"{'='*54}")

initial_path, _ = astar(base_grid, start, goal)
if not initial_path:
    print("No initial path found."); exit()
print(f"Initial path: {len(initial_path)} pts, "
      f"length = {path_length(initial_path):.1f} px\n")


# Place one temporary and one permanent obstacle on the initial route
obstacles = []
obs_types = ['temporary', 'permanent']

for i, frac in enumerate(OBSTACLE_FRAC[:2]):
    idx = int(frac * (len(initial_path) - 1))
    pos = list(initial_path[idx])
    obs_type = obs_types[i]

    # Temporary obstacle moves away perpendicular to the route.
    if idx + 1 < len(initial_path):
        p1 = initial_path[idx]
        p2 = initial_path[idx + 1]
        raw_dx = p2[0] - p1[0]
        raw_dy = p2[1] - p1[1]
        # Perpendicular: rotate 90 degrees
        perp = [-raw_dy, raw_dx]
        norm = max(abs(perp[0]), abs(perp[1]), 1)
        drift_dir = [int(perp[0]/norm * DRIFT_SPEED),
                     int(perp[1]/norm * DRIFT_SPEED)]
    else:
        drift_dir = [DRIFT_SPEED, 0]

    obstacles.append({
        'pos':          pos,
        'type':         obs_type,
        'cleared':      False,
        'wait_steps':   0,       # steps robot waited for this obstacle
        'steps_active': 0,       # steps since first detected
        'drift_dir':    drift_dir,
        'initial_pos':  list(pos),  # for visualization
    })
    print(f"  Obstacle {i+1} [{obs_type.upper()}] "
          f"at path[{idx}] = {tuple(pos)}")


# Simulation state
robot_pos        = start
path_index       = 0
current_path     = initial_path[:]
robot_trajectory = [robot_pos]

replan_count  = 0
replan_events = []   # full replan (permanent obstacle)
wait_events   = []   # waiting events (temporary obstacle)

robot_waiting    = False   # True when robot is stopped waiting
waiting_for_obs  = None    # which obstacle index robot is waiting for
step             = 0
max_steps        = 3000
goal_reached     = False

print(f"\nRunning simulation...")
print(f"{'─'*54}")

while step < max_steps:
    step += 1

    # Move temporary obstacles after the robot has detected them
    for obs in obstacles:
        if obs['cleared'] or obs['type'] != 'temporary':
            continue
        # Increment every step once robot has first detected this obstacle
        if obs['wait_steps'] > 0:
            obs['steps_active'] += 1
        if obs['steps_active'] >= TEMPORARY_CLEAR_AFTER:
            # Drift away
            ox, oy = obs['pos']
            dx, dy = obs['drift_dir']
            nx, ny = ox + dx, oy + dy
            if (0 <= nx < base_grid.shape[1] and
                    0 <= ny < base_grid.shape[0] and
                    base_grid[ny, nx] == 1):
                obs['pos'] = [nx, ny]
            else:
                # Reverse drift direction and try again
                obs['drift_dir'][0] *= -1
                obs['drift_dir'][1] *= -1
                dx, dy = obs['drift_dir']
                nx, ny = ox + dx, oy + dy
                if (0 <= nx < base_grid.shape[1] and
                        0 <= ny < base_grid.shape[0] and
                        base_grid[ny, nx] == 1):
                    obs['pos'] = [nx, ny]
                else:
                    obs['cleared'] = True
                    print(f"  Step {step:4d} | Obstacle "
                          f"[TEMPORARY] fully cleared path")
        # Remove temporary obstacle after it moves far enough from its start
        if not obs['cleared'] and obs['steps_active'] >= TEMPORARY_CLEAR_AFTER:
            dist = np.sqrt(
                (obs['pos'][0] - obs['initial_pos'][0])**2 +
                (obs['pos'][1] - obs['initial_pos'][1])**2
            )
            if dist > 35:
                obs['cleared'] = True
                print(f"  Step {step:4d} | Obstacle [TEMPORARY] "
                      f"drifted {dist:.1f}px away — path clear")

    # Add current obstacles to a temporary grid
    active_obs   = [tuple(o['pos']) for o in obstacles if not o['cleared']]
    current_grid = build_grid_with_obstacles(base_grid, active_obs, radius=4)

    # Check if an obstacle blocks the remaining route
    blocking_obs = None
    blocking_idx = None
    for i, obs in enumerate(obstacles):
        if obs['cleared']:
            continue
        if euclidean(robot_pos, tuple(obs['pos'])) > SENSOR_RADIUS:
            continue
        remaining = current_path[path_index:]
        if any(euclidean(tuple(obs['pos']), node) <= SENSOR_RADIUS * 0.7
               for node in remaining[:60]):
            if obs['steps_active'] == 0:
                obs['steps_active'] = 1
            blocking_obs = obs
            blocking_idx = i
            break  # handle one obstacle at a time

    # Decide whether to wait or replan
    if blocking_obs is not None:
        obs = blocking_obs
        i   = blocking_idx

        if obs['type'] == 'temporary':
            robot_waiting   = True
            waiting_for_obs = i
            obs['wait_steps'] += 1

            if obs['wait_steps'] == 1:
                wait_events.append({
                    'step':    step,
                    'robot':   robot_pos,
                    'obs_pos': tuple(obs['pos']),
                    'obs_idx': i,
                    'path':    current_path[path_index:path_index+40],
                })
                print(f"  Step {step:4d} | [WAIT] Temporary obstacle detected "
                      f"at {tuple(obs['pos'])} — robot waiting...")

            # Resume if cleared or no longer on path
            remaining    = current_path[path_index:]
            still_blocks = any(
                euclidean(tuple(obs['pos']), node) <= SENSOR_RADIUS * 0.7
                for node in remaining[:60]
            )
            if obs['cleared'] or not still_blocks:
                print(f"  Step {step:4d} | [RESUME] Temporary obstacle cleared "
                      f"(waited {obs['wait_steps']} steps) — robot resumes")
                robot_waiting   = False
                waiting_for_obs = None

        elif obs['type'] == 'permanent':
            robot_waiting   = True
            waiting_for_obs = i
            obs['wait_steps'] += 1

            if obs['wait_steps'] == 1:
                print(f"  Step {step:4d} | [WAIT] Permanent obstacle at "
                      f"{tuple(obs['pos'])} — waiting {PATIENCE_STEPS} steps...")

            if obs['wait_steps'] >= PATIENCE_STEPS:
                new_path, _ = astar(current_grid, robot_pos, goal)
                if new_path and len(new_path) > 1:
                    replan_count += 1
                    replan_events.append({
                        'step':     step,
                        'robot':    robot_pos,
                        'old_path': current_path[path_index:path_index+50],
                        'new_path': new_path[:50],
                        'obs_pos':  tuple(obs['pos']),
                        'obs_idx':  i,
                    })
                    current_path    = new_path
                    path_index      = 0
                    robot_waiting   = False
                    waiting_for_obs = None
                    obs['cleared']  = True

                    print(f"  Step {step:4d} | [REPLAN #{replan_count}] "
                          f"Patience expired — new route found, "
                          f"length: {path_length(new_path):.1f} px")
                else:
                    print(f"  Step {step:4d} | [REPLAN FAILED] No alternative path.")
                    break

    else:
        # No blocking obstacle — resume if was waiting
        if robot_waiting:
            print(f"  Step {step:4d} | [RESUME] Path clear — robot moving")
            robot_waiting   = False
            waiting_for_obs = None

    # Move the robot only if it is not waiting
    if not robot_waiting:
        next_idx   = min(path_index + ROBOT_STEP_SIZE, len(current_path) - 1)
        robot_pos  = current_path[next_idx]
        path_index = next_idx
        robot_trajectory.append(robot_pos)

    # Check goal
    # Stop when the goal is reached
    if euclidean(robot_pos, goal) < ROBOT_STEP_SIZE * 3:
        robot_pos = goal
        robot_trajectory.append(goal)
        goal_reached = True
        print(f"\n  Goal reached at step {step}!")
        break

    # Safety check if the path ends before the goal
    if path_index >= len(current_path) - 1 and not robot_waiting:
        new_path, _ = astar(current_grid, robot_pos, goal)
        if new_path:
            current_path = new_path
            path_index   = 0
        else:
            print(f"  Step {step}: No path — stopping."); break


# Print simulation summary
print(f"\n{'='*54}")
print(f"  Steps              : {step}")
print(f"  Wait events        : {len(wait_events)}")
print(f"  Replan events      : {replan_count}")
print(f"  Goal reached       : {goal_reached}")
print(f"  Initial path len   : {path_length(initial_path):.1f} px")
if current_path:
    print(f"  Final path len     : {path_length(current_path):.1f} px")
print(f"{'='*54}\n")


# Main result visualization
obs_colors = {0: 'orange', 1: 'red'}
obs_labels = {0: 'Obstacle 1 [Temporary]', 1: 'Obstacle 2 [Permanent]'}

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle(
    f"Dynamic Obstacle Avoidance — Map {map_id}  |  "
    f"Wait events: {len(wait_events)}  |  "
    f"Replans: {replan_count}  |  "
    f"Goal reached: {goal_reached}",
    fontsize=12, fontweight='bold'
)

for ax, use_sat in zip(axes, [False, True]):
    ax.imshow(map_rgb if use_sat else base_grid,
              cmap=None if use_sat else 'gray')
    ax.set_title("Satellite View" if use_sat else "Grid View", fontsize=11)

    # Initial planned path
    ix = [p[0] for p in initial_path]
    iy = [p[1] for p in initial_path]
    ax.plot(ix, iy, '--', color='deepskyblue', linewidth=3.0,
            alpha=1.0, label='Initial path', zorder=2)

    # Robot trajectory — solid green
    tx = [p[0] for p in robot_trajectory]
    ty = [p[1] for p in robot_trajectory]
    ax.plot(tx, ty, '-', color='limegreen', linewidth=2.5,
            alpha=0.9, label='Robot trajectory', zorder=3)

    # Waiting points — yellow diamonds
    for ev in wait_events:
        ax.plot(ev['robot'][0], ev['robot'][1], 'D',
                color='yellow', markersize=11,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=5)
    if wait_events:
        ax.plot([], [], 'D', color='yellow', markersize=9,
                markeredgecolor='black',
                label=f'Wait point ({len(wait_events)}x)')

    # Replan event markers — white stars
    for ev in replan_events:
        ax.plot(ev['robot'][0], ev['robot'][1], '*',
                color='white', markersize=14,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=5)
    if replan_events:
        ax.plot([], [], '*', color='white', markersize=10,
                markeredgecolor='black',
                label=f'Replan point ({replan_count}x)')

    # Obstacles and their sensor zones
    for i, obs in enumerate(obstacles):
        c  = obs_colors.get(i, 'magenta')
        lbl = obs_labels.get(i, f'Obstacle {i+1}')
        ox, oy = obs['pos']
        ax.plot(
            ox, oy, 
            's', 
            color=c, 
            markersize=13,
            markeredgecolor='black', 
            markeredgewidth=1.2,
            zorder=6, 
            label=lbl
        )
        
        # Sensor circle
        ax.add_patch(
            plt.Circle(
                (ox, oy), 
                SENSOR_RADIUS,
                color=c, 
                fill=True, 
                alpha=0.12, 
                zorder=1
            )
        )

        ax.add_patch(
            plt.Circle(
                (ox, oy), 
                SENSOR_RADIUS,
                color=c, 
                fill=False,
                linestyle='--', 
                linewidth=1.3, 
                zorder=2
            )
        )

    # Start, goal, and final robot position
    ax.plot(
        start[0], 
        start[1], 
        'o', 
        color='limegreen', 
        markersize=12,
        markeredgecolor='black', 
        markeredgewidth=1.5,
        zorder=7, 
        label='Start'
    )

    ax.plot(
        goal[0], 
        goal[1], 
        'o', 
        color='magenta', 
        markersize=12,
        markeredgecolor='black', 
        markeredgewidth=1.5,
        zorder=7, 
        label='Goal'
    )

    ax.plot(
        robot_pos[0], 
        robot_pos[1], 
        '^', 
        color='blue', 
        markersize=12,
        markeredgecolor='black', 
        markeredgewidth=1.5,
        zorder=7, 
        label='Robot (final)'
    )

    ax.axis('off')
    ax.legend(loc='lower right', fontsize=8,
              framealpha=0.88, edgecolor='gray')

plt.tight_layout()


# Event close-up figures
n_closeups = (1 if wait_events else 0) + (1 if replan_events else 0)

if n_closeups > 0:
    fig2, axes2 = plt.subplots(1, n_closeups, figsize=(8 * n_closeups, 8))

    if n_closeups == 1:
        axes2 = [axes2]

    fig2.suptitle(
        "Event Close-ups: Wait (Temporary) vs Replan (Permanent)",
        fontsize=13, 
        fontweight='bold'
    )

    col = 0

    # Close-up for temporary obstacle waiting
    if wait_events:
        ev  = wait_events[0]
        ax  = axes2[col]; 
        col += 1

        rx, ry = ev['robot']
        margin = 130

        x0 = max(0, rx-margin); 
        x1 = min(base_grid.shape[1], rx+margin)
        y0 = max(0, ry-margin); 
        y1 = min(base_grid.shape[0], ry+margin)

        ax.imshow(
            base_grid[y0:y1, x0:x1], 
            cmap='gray',
            extent=[x0, x1, y1, y0]
        )

        # Original path continues unchanged
        if ev['path']:
            px = [p[0] for p in ev['path']]
            py = [p[1] for p in ev['path']]

            ax.plot(
                px, py, 
                '-', 
                color='deepskyblue', 
                linewidth=2.5,
                label='Original path (unchanged)', 
                zorder=3
            )

        # Obstacle
        ox, oy = ev['obs_pos']

        ax.plot(
            ox, oy, 
            's', 
            color='orange', 
            markersize=13,
            markeredgecolor='black', 
            zorder=5,
            label='Temporary obstacle\n(person crossing)'
        )

        ax.add_patch(
            plt.Circle(
                (ox, oy), 
                SENSOR_RADIUS,
                color='orange', 
                fill=True, 
                alpha=0.15)
            )
        
        ax.add_patch(
            plt.Circle(
                (ox, oy), 
                SENSOR_RADIUS,
                color='orange', 
                fill=False,
                linestyle='--', 
                linewidth=1.5)
            )

        # Robot waiting
        ax.plot(rx, ry, 'D', color='yellow', markersize=13,
                markeredgecolor='black', zorder=6,
                label='Robot (waiting)')
        ax.add_patch(plt.Circle((rx, ry), SENSOR_RADIUS,
                                color='blue', fill=False,
                                linestyle=':', linewidth=1.5, alpha=0.7))

        # Arrow: direction obstacle moves away — computed from initial to final position
        i_obs    = ev['obs_idx']
        obs_obj  = obstacles[i_obs]
        init_pos = obs_obj['initial_pos']
        final_pos= obs_obj['pos']
        # Vector from initial to final position
        vec_x = final_pos[0] - init_pos[0]
        vec_y = final_pos[1] - init_pos[1]
        vec_len = max(np.sqrt(vec_x**2 + vec_y**2), 1.0)
        arrow_scale = 55
        adx = (vec_x / vec_len) * arrow_scale
        ady = (vec_y / vec_len) * arrow_scale
        # If obstacle didn't move yet, use drift_dir from settings
        if vec_len < 2:
            dd     = obs_obj['drift_dir']
            dd_len = max(abs(dd[0]), abs(dd[1]), 1)
            adx    = (dd[0] / dd_len) * arrow_scale
            ady    = (dd[1] / dd_len) * arrow_scale
        ax.annotate(
            '',
            xy=(ox + adx, oy + ady),
            xytext=(ox, oy),
            arrowprops=dict(
                arrowstyle='-|>',
                color='lime',
                lw=4.0,
                mutation_scale=30
            ),
            zorder=9
        )

        # Add arrow direction to legend
        ax.annotate('', xy=(0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='lime', lw=2.5,
                                    mutation_scale=15))
        ax.plot([], [], '-', color='lime', linewidth=3,
                label='Obstacle movement direction')

        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title('')
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.88)

    # Close-up for permanent obstacle replanning
    if replan_events:
        ev  = replan_events[0]
        ax  = axes2[col]
        rx, ry = ev['robot']
        margin = 100
        x0 = max(0, rx-margin); x1 = min(base_grid.shape[1], rx+margin)
        y0 = max(0, ry-margin); y1 = min(base_grid.shape[0], ry+margin)

        ax.imshow(base_grid[y0:y1, x0:x1], cmap='gray',
                  extent=[x0, x1, y1, y0])

        # Old (blocked) path
        if ev['old_path']:
            ox_ = [p[0] for p in ev['old_path']]
            oy_ = [p[1] for p in ev['old_path']]
            ax.plot(ox_, oy_, '--', color='red', linewidth=2.5,
                    alpha=0.8, label='Blocked path (impassable)')

        # New replanned path
        if ev['new_path']:
            nx_ = [p[0] for p in ev['new_path']]
            ny_ = [p[1] for p in ev['new_path']]
            ax.plot(nx_, ny_, '-', color='limegreen', linewidth=2.5,
                    label='Replanned path (new route)')

        # Obstacle
        ox, oy = ev['obs_pos']
        ax.plot(ox, oy, 's', color='red', markersize=13,
                markeredgecolor='black', zorder=5,
                label='Permanent obstacle\n(road blocked)')
        ax.add_patch(plt.Circle((ox, oy), SENSOR_RADIUS,
                                color='red', fill=True, alpha=0.15))
        ax.add_patch(plt.Circle((ox, oy), SENSOR_RADIUS,
                                color='red', fill=False,
                                linestyle='--', linewidth=1.5))

        # Robot replanning
        ax.plot(rx, ry, '*', color='white', markersize=16,
                markeredgecolor='black', zorder=6,
                label='Robot (replanning)')
        ax.add_patch(plt.Circle((rx, ry), SENSOR_RADIUS,
                                color='blue', fill=False,
                                linestyle=':', linewidth=1.5, alpha=0.7))

        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
        ax.set_title('')
        ax.axis('off')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.88)

    # Close-up captions
    # Wait event caption - orange
    if wait_events:
        ev_w = wait_events[0]
        fig2.text(
            0.25, 0.01,
            f"WAIT EVENT  |  Step {ev_w['step']}  |  "
            f"Robot detects temporary obstacle (person crossing)\n"
            f"Robot waits → obstacle moves away (green arrow) → "
            f"robot resumes SAME path — no replanning",
            ha='center', va='bottom',
            fontsize=10, color='orange', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='black', alpha=0.75)
        )

    # Right caption (replan event) — red
    if replan_events:
        ev_r = replan_events[0]
        x_pos = 0.75 if n_closeups == 2 else 0.5
        fig2.text(
            x_pos, 0.01,
            f"REPLAN EVENT  |  Step {ev_r['step']}  |  "
            f"Permanent obstacle (road blocked) detected\n"
            f"Robot waits {PATIENCE_STEPS} steps → patience expires → "
            f"replans to DIFFERENT route",
            ha='center', va='bottom',
            fontsize=10, color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='black', alpha=0.75)
        )

    plt.tight_layout(rect=[0, 0.10, 1, 1])

plt.show()
print("Simulation complete.")