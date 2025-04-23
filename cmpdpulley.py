import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global physics parameters.
g = 9.81  # gravitational acceleration (m/s^2)
m = 0.8  # mass of the load (kg)
F_in = 1.0  # constant input force (N)
MA = 8.0  # mechanical advantage (8:1 system)
b = 1.0  # damping coefficient (N·s/m)
dt = 0.06  # time step (s)

x_state = 0.0  # initial load displacement
v_state = 0.0  # initial load velocity

F_pos = np.array([-2, 4])  # Center for fixed (stationary) pulley F
A_init = np.array([-1.4, 2])  # Initial center for movable pulley A
B_init = np.array([-0.8, 1])  # Initial center for movable pulley B
C_init = np.array([-0.2, 0])  # Initial center for movable pulley C
load_init = np.array([-0.2, -2])  # Initial center for the load

pulley_radius = 0.3

load_width, load_height = 2, 1

# Fixed anchor points for the vertical rope segments (above each pulley's exit).
Anchor1 = np.array([-0.7, 4.5])  # Anchor for rope segment from pulley A
Anchor2 = np.array([0.3, 4.5])  # Anchor for rope segment from pulley B
Anchor3 = np.array([1.3, 4.5])  # Anchor for rope segment from pulley C


def get_arc_points(center, radius, pt_start, pt_end, num_points=20, direction=None):
    """
    Returns points along the circular arc between two given points on the circumference
    of a circle with center `center` and radius `radius`.

    Parameters:
      center: (x, y) coordinates of the circle’s center.
      radius: Radius of the circle.
      pt_start: Starting point on the circumference.
      pt_end: Ending point on the circumference.
      num_points: Number of points along the arc.
      direction: If 'clockwise' or 'counterclockwise', forces the arc in that direction.
                 Otherwise, the shortest arc is computed.

    Returns:
      An array of shape (num_points, 2) with the (x, y) coordinates along the arc.
    """
    angle_start = np.arctan2(pt_start[1] - center[1], pt_start[0] - center[0])
    angle_end = np.arctan2(pt_end[1] - center[1], pt_end[0] - center[0])
    if direction == 'clockwise':
        if angle_start < angle_end:
            angle_start += 2 * np.pi
        dtheta = angle_end - angle_start
    elif direction == 'counterclockwise':
        if angle_end < angle_start:
            angle_end += 2 * np.pi
        dtheta = angle_end - angle_start
    else:
        dtheta = angle_end - angle_start
        if dtheta > np.pi:
            dtheta -= 2 * np.pi
        elif dtheta < -np.pi:
            dtheta += 2 * np.pi
    angles = np.linspace(angle_start, angle_start + dtheta, num_points)
    arc = np.column_stack((center[0] + radius * np.cos(angles),
                           center[1] + radius * np.sin(angles)))
    return arc


# Set up the plot.
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-4.5, 4.5)
ax.set_aspect('equal')
ax.set_title("Compound Pulley System (8:1 Mechanical Advantage)")

# Draw the fixed pulley and movable pulleys, and add the load.
fixed_pulley = plt.Circle(F_pos, pulley_radius, fc='gray', ec='black')
ax.add_patch(fixed_pulley)
movable_A = plt.Circle(A_init, pulley_radius, fc='lightblue', ec='black')
movable_B = plt.Circle(B_init, pulley_radius, fc='lightgreen', ec='black')
movable_C = plt.Circle(C_init, pulley_radius, fc='lightcoral', ec='black')
ax.add_patch(movable_A)
ax.add_patch(movable_B)
ax.add_patch(movable_C)
load_rect = plt.Rectangle((load_init[0] - load_width / 2, load_init[1] - load_height / 2),
                      load_width, load_height, fc='brown', ec='black')
ax.add_patch(load_rect)

# Create text labels for the pulleys.
F_label = ax.text(F_pos[0], F_pos[1], "F", ha="center", va="center", fontsize=12, color="white")
A_label = ax.text(A_init[0], A_init[1], "A", ha="center", va="center", fontsize=12, color="black")
B_label = ax.text(B_init[0], B_init[1], "B", ha="center", va="center", fontsize=12, color="black")
C_label = ax.text(C_init[0], C_init[1], "C", ha="center", va="center", fontsize=12, color="black")

# Create text labels for forces.
input_label = ax.text(0, 0, "1 N", ha="right", va="center", fontsize=8, fontweight="bold", color="blue")
force_label_2N = ax.text(0, 0, "2 N", ha="right", va="center", fontsize=10, fontweight="bold", color="green")
force_label_4N = ax.text(0, 0, "4 N", ha="right", va="center", fontsize=12, fontweight="bold", color="red")
output_label = ax.text(0, 0, "8 N", ha="left", va="center", fontsize=14, fontweight="bold", color="black")

# Create four line objects for the four rope segments.
rope_line1, = ax.plot([], [], lw=2, color='black')
rope_line2, = ax.plot([], [], lw=2, color='black')
rope_line3, = ax.plot([], [], lw=2, color='black')
rope_line4, = ax.plot([], [], lw=2, color='black')


def update_physics():
    """
    Integrates the load's dynamics using Euler's method.

    The net force on the load is:
        F_net = MA * F_in - m*g - b*v
    where MA*F_in is the effective upward force,
          m*g is the gravitational force, and
          b*v is a damping force.
    The acceleration is a = F_net/m. The velocity and displacement are updated:
        v_new = v_old + a*dt
        x_new = x_old + v_new*dt
    """
    global x_state, v_state
    F_net = MA * F_in - m * g - b * v_state
    a = F_net / m
    v_state = v_state + a * dt
    x_state = x_state + v_state * dt
    return x_state, v_state


def init_anim():
    """
    Initialization function for the animation.

    Clears the rope segment data and sets the initial positions for the text labels.
    """
    for rl in [rope_line1, rope_line2, rope_line3, rope_line4]:
        rl.set_data([], [])
    F_label.set_position(F_pos)
    A_label.set_position(A_init)
    B_label.set_position(B_init)
    C_label.set_position(C_init)
    input_label.set_position(F_pos + np.array([-1, 0]))
    output_label.set_position(load_init + np.array([0, load_height / 2]))
    force_label_2N.set_position(np.array([0, 0]))
    force_label_4N.set_position(np.array([0, 0]))
    return (rope_line1, rope_line2, rope_line3, rope_line4,
            movable_A, movable_B, movable_C, load_rect,
            F_label, A_label, B_label, C_label,
            input_label, output_label, force_label_2N, force_label_4N)


def update_anim(frame):
    """
    Update function for the animation.

    1. The load's physics are updated by integrating the equations of motion.
    2. The free rope displacement is computed as p = MA * x_state.
    3. The positions of the movable pulleys and the load are updated based on p:
         - Pulley A moves upward by p/8.
         - Pulley B moves upward by p/12.
         - Pulley C and the load move upward by p/16.
    4. The rope segments are recalculated accordingly.
    5. The text labels for the pulleys and force indicators ("1 N", "2 N", "4 N", "8 N")
       are updated so they track the free end and the midpoints of specific rope segments.
    """
    # Update physics:
    x, v = update_physics()
    # Compute free rope displacement (p = MA * x).
    p = MA * x

    # Update positions based on p:
    A_pos_new = A_init + np.array([0, p / 8])
    B_pos_new = B_init + np.array([0, p / 12])
    C_pos_new = C_init + np.array([0, p / 16])
    load_pos_new = load_init + np.array([0, p / 16])

    # Update movable pulley and load positions.
    movable_A.center = A_pos_new
    movable_B.center = B_pos_new
    movable_C.center = C_pos_new
    load_rect.set_xy((load_pos_new[0] - load_width / 2, load_pos_new[1] - load_height / 2))

    # Update pulley labels.
    A_label.set_position(A_pos_new)
    B_label.set_position(B_pos_new)
    C_label.set_position(C_pos_new)

    # Segment 1: Free End -> Fixed Pulley -> Pulley A
    free_end1 = F_pos + np.array([-1, -p])
    T_fixed_in = F_pos + np.array([-pulley_radius, 0])
    T_fixed_out = F_pos + np.array([pulley_radius, 0])
    seg1_part1 = np.array([free_end1, T_fixed_in])
    arc_fixed = get_arc_points(F_pos, pulley_radius, T_fixed_in, T_fixed_out,
                               num_points=20, direction='clockwise')
    A_in_new = A_pos_new + np.array([-pulley_radius, 0])
    seg1_part2 = np.array([T_fixed_out, A_in_new])
    A_out_pt_new = A_pos_new + np.array([pulley_radius, 0])
    arc_A = get_arc_points(A_pos_new, pulley_radius, A_in_new, A_out_pt_new,
                           num_points=20, direction='counterclockwise')
    seg1_part3 = np.array([A_out_pt_new, Anchor1])
    segment1 = np.vstack([seg1_part1, arc_fixed, seg1_part2, arc_A, seg1_part3])
    rope_line1.set_data(segment1[:, 0], segment1[:, 1])

    # Update "1 N" label to track the free end.
    input_label.set_position(free_end1 + np.array([-0.2, 0]))

    # Segment 2: From Pulley A's Hook -> Pulley B
    A_hook = A_pos_new + np.array([0, -pulley_radius])
    B_in_new = B_pos_new + np.array([-pulley_radius, 0])
    seg2_part1 = np.array([A_hook, B_in_new])
    B_out_pt_new = B_pos_new + np.array([pulley_radius, 0])
    arc_B = get_arc_points(B_pos_new, pulley_radius, B_in_new, B_out_pt_new,
                           num_points=20, direction='counterclockwise')
    seg2_part2 = np.array([B_out_pt_new, Anchor2])
    segment2 = np.vstack([seg2_part1, arc_B, seg2_part2])
    rope_line2.set_data(segment2[:, 0], segment2[:, 1])

    # Update "2 N" label to track the midpoint of the segment from A's hook to B's entry.
    mid_seg2 = (A_hook + B_in_new) / 2
    force_label_2N.set_position(mid_seg2 + np.array([-0.2, 0]))

    # Segment 3: From Pulley B's Hook -> Pulley C
    B_hook = B_pos_new + np.array([0, -pulley_radius])
    C_in_new = C_pos_new + np.array([-pulley_radius, 0])
    seg3_part1 = np.array([B_hook, C_in_new])
    C_out_pt_new = C_pos_new + np.array([pulley_radius, 0])
    arc_C = get_arc_points(C_pos_new, pulley_radius, C_in_new, C_out_pt_new,
                           num_points=20, direction='counterclockwise')
    seg3_part2 = np.array([C_out_pt_new, Anchor3])
    segment3 = np.vstack([seg3_part1, arc_C, seg3_part2])
    rope_line3.set_data(segment3[:, 0], segment3[:, 1])

    # Update "4 N" label to track the midpoint of the segment from B's hook to C's entry.
    mid_seg3 = (B_hook + C_in_new) / 2
    force_label_4N.set_position(mid_seg3 + np.array([-0.2, 0]))

    # Segment 4: From Pulley C's Hook -> Load Attachment
    C_hook = C_pos_new + np.array([0, -pulley_radius])
    load_attach = load_pos_new + np.array([0, load_height / 2])
    segment4 = np.array([C_hook, load_attach])
    rope_line4.set_data(segment4[:, 0], segment4[:, 1])

    # Update "8 N" label to track the midpoint of the segment from pulley C's hook to the load.
    mid_seg4 = (C_hook + load_attach) / 2
    output_label.set_position(mid_seg4 + np.array([0.2, 0]))

    return (rope_line1, rope_line2, rope_line3, rope_line4,
            movable_A, movable_B, movable_C, load_rect,
            F_label, A_label, B_label, C_label,
            input_label, output_label, force_label_2N, force_label_4N)


anim = animation.FuncAnimation(fig, update_anim, frames=np.arange(0, 120),
                               init_func=init_anim, interval=50, blit=True, repeat=False)

anim.save("compound_pulley.gif", writer="pillow", fps=15)