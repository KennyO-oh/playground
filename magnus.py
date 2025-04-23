import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# Constants for the simulation
g = 9.81  # Acceleration due to gravity (m/s²)
m = 0.625  # Mass of the ball (kg)
r_ball = 0.305  # Radius of the ball (m)
A = np.pi * r_ball ** 2  # Cross-sectional area (m²)
rho = 1.225  # Air density (kg/m³)
C_d = 0.47  # Drag coefficient for a sphere
omega = 9.5  # Initial spin rate (rad/s)
k_lift = 1.2  # Lift coefficient scaling factor


def reset_state():
    """
    Reset the simulation state variables.

    Sets the ball's position, velocity, time, and clears the trail.
    """
    global x, y, vx, vy, t, trail_x, trail_y
    x = 0.0                    # horizontal position (m)
    y = 40.0                   # vertical position (m)
    vx = 0.0                   # horizontal velocity (m/s)
    vy = 0.0                   # vertical velocity (m/s)
    t = 0.0                    # simulation time (s)
    trail_x = []
    trail_y = []


reset_state()

dt = 0.01  # Time step (s)

# Set up the figure and axes
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)  # Leave space for buttons and info panel
ax.set_xlim(-5, 145)
ax.set_ylim(0, 50)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.set_title("Basketball Drop: Magnus Effect")

# Create the ball marker and its trail for animation
ball_marker, = ax.plot([x], [y], 'o', markersize=5,
                         mfc='orange', mec='black', animated=True)
trail_line, = ax.plot([], [], linestyle='--', color='gray', animated=True)

# Text objects to display spin and speed
spin_text = ax.text(0.75, 0.90, f'Spin: {omega:.1f} rad/s',
                    transform=ax.transAxes, fontsize=10, color='red')
speed_text_h = ax.text(x, y + 3, '', fontsize=9, color='green')
speed_text_v = ax.text(x, y + 2, '', fontsize=9, color='blue')


def update(frame):
    """
    Update the simulation state for each animation frame.

    This function computes the new position and velocity of the ball,
    taking into account drag and the Magnus (lift) effect. It then updates
    the ball's marker, its trail, and text annotations.
    """
    global x, y, vx, vy, t, trail_x, trail_y, omega

    steps_per_frame = 5  # Multiple integration steps per frame for smoother motion
    for _ in range(steps_per_frame):
        # Calculate the speed and ensure a minimum value for lift calculations
        v = np.sqrt(vx ** 2 + vy ** 2)
        v_for_lift = max(v, 1.0)  # Avoid division by zero in lift coefficient
        # Lift coefficient based on spin and ball radius (scaled by k_lift)
        C_l = k_lift * (r_ball * omega) / v_for_lift

        # Drag accelerations (proportional to v * velocity component)
        a_drag_x = -0.5 * C_d * rho * A * v * vx / m
        a_drag_y = -0.5 * C_d * rho * A * v * vy / m

        # Magnus (lift) accelerations, perpendicular to velocity
        a_magnus_x = 0.5 * C_l * rho * A * v * (-vy) / m
        a_magnus_y = 0.5 * C_l * rho * A * v * (vx) / m

        # Total acceleration (include gravity in the y-direction)
        ax_total = a_drag_x + a_magnus_x
        ay_total = a_drag_y + a_magnus_y - g

        # Euler integration for velocity and position updates
        vx += ax_total * dt
        vy += ay_total * dt
        x += vx * dt
        y += vy * dt
        t += dt

        # Stop the simulation if the ball hits the ground
        if y <= r_ball:
            anim.event_source.stop()
            break

    # Update the trail and marker positions
    trail_x.append(x)
    trail_y.append(y)
    trail_line.set_data(trail_x, trail_y)
    ball_marker.set_data([x], [y])

    # Update text displays for speeds and spin
    speed_text_h.set_position((x - 2, y + 4))
    speed_text_h.set_text(f'Horiz Speed: {vx:.2f} m/s')
    speed_text_v.set_position((x - 2, y + 1))
    speed_text_v.set_text(f'Vert Speed: {vy:.2f} m/s')
    spin_text.set_text(f'Spin: {omega:.1f} rad/s')

    return ball_marker, spin_text, speed_text_h, speed_text_v, trail_line


# Create the animation
anim = FuncAnimation(fig, update, frames=range(5000), interval=30, blit=True)

# Create buttons for adjusting the spin rate
ax_increase = plt.axes([0.68, 0.1, 0.1, 0.075])
ax_decrease = plt.axes([0.79, 0.1, 0.1, 0.075])
btn_increase = Button(ax_increase, 'Spin +')
btn_decrease = Button(ax_decrease, 'Spin -')


def increase_spin(event):
    """
    Increase the spin rate by 0.5 rad/s (up to a maximum of 19 rad/s) and restart.
    """
    global omega
    if omega < 19:
        omega += 0.5
    else:
        omega = 19
    reset_state()
    anim.event_source.start()


def decrease_spin(event):
    """
    Decrease the spin rate by 0.5 rad/s (down to a minimum of 0) and restart.
    """
    global omega
    if omega > 0:
        omega -= 0.5
    else:
        omega = 0
    reset_state()
    anim.event_source.start()


btn_increase.on_clicked(increase_spin)
btn_decrease.on_clicked(decrease_spin)

# Additional info on the Magnus Effect and simulation.
info_str = (
    "Lift force: Fₗ = 0.5 * Cₗ * ρ * A * v²\n"
    "6.33 rads/s = ~1 rev/s"
)
fig.text(0.06, 0.12, info_str, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

plt.show()