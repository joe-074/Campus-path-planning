# Campus Path Planning for a Mobile Robot

This project implements and compares path-planning algorithms for a mobile robot on a campus map.

The environment is represented as a binary grid map, where free cells are available for movement and obstacle cells are blocked. The project includes static path planning, algorithm comparison, synthetic benchmarking, and dynamic obstacle simulation.

## Project structure

```text
campus_path_planning/
|-- main.py
|-- dynamic.py
|-- analysis.py
|-- location_test.py
|-- requirements.txt
|-- maps/
|-- masks/
|-- outputs/
```

## Files

- `main.py`  
  Runs A* and Dijkstra on a prepared campus map and compares path length, visited nodes, and runtime.

- `dynamic.py`  
  Simulates robot movement in a dynamic environment with temporary and permanent obstacles.

- `analysis.py`  
  Runs synthetic benchmark experiments on random grids of increasing size.

- `location_test.py`  
  Allows selecting start and goal coordinates by clicking on a binary mask.

- `maps/`  
  Contains satellite map images.

- `masks/`  
  Contains binary walkability masks.

- `outputs/`  
  Can be used to store generated result images.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
matplotlib
opencv-python
```

## How to run

Run the main comparison between A* and Dijkstra:

```bash
python main.py
```

Run the dynamic obstacle simulation:

```bash
python dynamic.py
```

Run the synthetic benchmark:

```bash
python analysis.py
```

Select start and goal points manually:

```bash
python location_test.py
```

## Map representation

The binary mask is converted into a grid:

```text
1 = free cell
0 = obstacle
```

Coordinates are written as:

```text
(x, y)
```

while NumPy arrays are indexed as:

```text
grid[y, x]
```

## Algorithms

The project uses two graph-search algorithms:

- Dijkstra's algorithm
- A* algorithm with Euclidean heuristic

Dijkstra is used as a baseline method. A* is used as the main planner because it finds the same optimal path length while usually reducing the number of visited nodes and runtime.

## Dynamic obstacle handling

The dynamic simulation includes two obstacle types:

- temporary obstacle: the robot waits until the path becomes clear;
- permanent obstacle: the robot waits for a limited time and then replans a new route.

This demonstrates adaptive path planning in a changing environment.

## Notes

This project was developed as part of a master's thesis on path-planning algorithms for a mobile robot on a campus territory.