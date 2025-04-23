import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set the length of the long bars and half-length bar (r)
L = 10.0
r = L / 2.0

# Fixed points:
A = np.array([0, L])      # Top pivot point (fixed)
D = np.array([0, 0])      # Fixed bottom-left point
B = np.array([0, L/2])    # Midpoint of left long bar

# Create the figure and axis with extended limits
fig, ax = plt.subplots()
ax.set_xlim(-15, 15)   # Extended left limit
ax.set_ylim(-5, 25)    # Extended top limit
ax.set_aspect('equal', adjustable='box')

# Initialize plot elements representing the pantograph's structure
left_bar, = ax.plot([], [], 'b-o', label='A-D Bar')
right_bar, = ax.plot([], [], 'r-o', label='A-F Bar')
half_bars, = ax.plot([], [], 'g-o', label='B-E & C-E Bars')

# Traced arcs for the paths of Points F and E
trace_F, = ax.plot([], [], 'm--', label='Arc of F')
trace_E, = ax.plot([], [], 'c--', label='Arc of E')

ax.legend(loc='upper left')

# Lists to store historical positions for arc tracing
F_trail = []
E_trail = []

# Define theta values covering a full circle
thetas = np.linspace(-np.pi/2 + 0.01, 3 * np.pi/2 + 0.01, 360)

# Add static labels for fixed points
ax.text(A[0] - 1, A[1], 'A', fontsize=10, fontweight='bold', color='black', ha='center')
ax.text(D[0] - 1, D[1], 'D', fontsize=10, fontweight='bold', color='black', ha='center')
ax.text(B[0] - 1, B[1], 'B', fontsize=10, fontweight='bold', color='black', ha='center')

# Create moving labels for dynamic points: E, F, and C
label_E = ax.text(0, 0, 'E', fontsize=10, fontweight='bold', color='black', ha='center')
label_F = ax.text(0, 0, 'F', fontsize=10, fontweight='bold', color='black', ha='center')
label_C = ax.text(0, 0, 'C', fontsize=10, fontweight='bold', color='black', ha='center')


def update(theta):
    """
    Update the pantograph animation for a given frame, showing full-circle motions
    for Points E and F.

    This function recalculates the positions of the moving points based on the current
    angle (theta), updates the positions of the pantograph's bars, traces the paths of
    Points F and E, and repositions the dynamic labels accordingly.

    Parameters:
        theta (float): The current angle in radians, ranging from -π/2 + 0.01 to 3π/2 + 0.01,
                       representing a full circular motion of Point F around A.

    Returns:
        None
    """
    global F_trail, E_trail

    # When a full cycle restarts (theta equals the initial value), clear the trails
    if theta == thetas[0]:
        F_trail.clear()
        E_trail.clear()

    # Compute the position of Point F (rotating around fixed point A)
    F = A + L * np.array([np.cos(theta), np.sin(theta)])
    # Compute Point C as the midpoint of A and F
    C = (A + F) / 2.0

    # Compute the position of Point E as the intersection of two circles:
    #   1. Centered at B with radius r
    #   2. Centered at C with radius r
    d = np.linalg.norm(C - B)
    if d < 1e-6:
        E = B  # Edge case: if C and B are nearly identical, set E = B
    else:
        a = d / 2.0
        if r**2 - a**2 < 0:
            return  # Skip configuration if circles do not intersect
        h = np.sqrt(r**2 - a**2)
        M = B + (a / d) * (C - B)  # Midpoint between B and C
        perp = np.array([-(C - B)[1], (C - B)[0]]) / d  # Perpendicular vector to BC
        candidate1 = M + h * perp
        candidate2 = M - h * perp
        # Select the candidate with the lower y-coordinate as the proper intersection
        E = candidate1 if candidate1[1] < candidate2[1] else candidate2

    # Append current positions to the trails for arc tracing
    F_trail.append(F)
    E_trail.append(E)

    # Convert stored trail positions into x and y coordinate lists
    F_x, F_y = zip(*F_trail)
    E_x, E_y = zip(*E_trail)

    # Update the coordinates for the pantograph's bars
    left_bar.set_data([A[0], B[0], D[0]], [A[1], B[1], D[1]])
    right_bar.set_data([A[0], C[0], F[0]], [A[1], C[1], F[1]])
    half_bars.set_data([B[0], E[0], C[0]], [B[1], E[1], C[1]])

    # Update the traced arcs for Points F and E
    trace_F.set_data(F_x, F_y)
    trace_E.set_data(E_x, E_y)

    # Update the positions of the moving labels:
    # Points E and F labels are moved 1.5 units lower, and the label for C is moved 0.5 units higher.
    label_E.set_position((E[0], E[1] - 1.5))
    label_F.set_position((F[0], F[1] - 1.5))
    label_C.set_position((C[0], C[1] + 0.5))

    # Update the title with the current angle (in radians)
    ax.set_title(f'Drawing with a Pantograph (θ = {theta:.2f} rad)')


# Create the animation using the update function and the specified theta values
anim = animation.FuncAnimation(fig, update, frames=thetas, interval=10, repeat=True)

# Save the animation as a GIF
anim.save("pantograph_animation.gif", writer="pillow", fps=30)