import cv2
import numpy as np
import matplotlib.pyplot as plt


# Select the mask used for choosing start and goal points
map_id = 4  # change this to select the map

mask_path = f"masks/mask{map_id}.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

if mask is None:
    print(f"Error: {mask_path} not found")
    exit()

# Convert the mask to a binary grid:
# 1 = free cell, 0 = obstacle
grid = (mask > 127).astype(np.uint8)

clicked_points = []

def onclick(event):
    """Store and display clicked points on the map."""
    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        print(f"Clicked: ({x}, {y}), grid value = {grid[y, x]}")
        clicked_points.append((x, y))

        # First point is start, second point is goal
        color = 'go' if len(clicked_points) == 1 else 'ro'
        plt.plot(x, y, color, markersize=8)
        plt.draw()

        if len(clicked_points) >= 2:
            print("Use these values:")
            print(f"start = {clicked_points[0]}")
            print(f"goal  = {clicked_points[1]}")

# Show the binary mask and wait for user clicks
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="gray")
plt.title(f"Map {map_id} — Click START (green) then GOAL (red) on white paths")
plt.axis("off")

plt.gcf().canvas.mpl_connect("button_press_event", onclick)
plt.show()