import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

# ===========================
# Constants & Aircraft Data
# ===========================
g = 32.174  # gravitational acceleration (ft/s²)
weight = 70000.0  # in lbs
mass = weight / g
A = 120.0  # frontal area of the aircraft (ft²)
C_d = 0.2  # baseline drag coefficient (dimensionless)

# Use IAS (indicated airspeed) in knots as input; simulation uses TAS (true airspeed in ft/s)
IAS_knots = 250.0
kts_to_ftps = 1.6878  # conversion factor: 1 knot = 1.6878 ft/s

# Standard sea-level air density (slug/ft³), used for TAS conversion.
rho_sl = 0.002377

# ===========================
# Atmospheric Data
# ===========================
# Interpolate the atmospheric density at a given altitude.
altitude_points = np.array([25000, 30000, 35000, 40000])
density_points = np.array([0.001066, 0.000891, 0.000738, 0.00060])

# ===========================
# Phase 1: Powered Ascent Parameters
# ===========================
gamma_target = math.radians(50)  # target climb angle: 50°
g_eff_max = 1.8  # maximum effective g allowed during turning maneuvers
k1 = 0.5  # gain for pitch-up control during Phase 1
tau_turn = 1.0  # lag time constant for the turning rate in Phase 1 (s)

# ===========================
# Phase 2: Parabolic Maneuver Parameters
# ===========================
v_target_drag = 200.0  # speed (ft/s) below which extra drag is not applied
k_drag = 0.005  # gain for the variable drag multiplier during the parabolic maneuver
gamma_rev_trigger = math.radians(-50)  # trigger recovery when flight-path angle reaches -50°


# ===========================
# Phase 3: Ballistic Free-Fall
# ===========================
# Run Phase 3 until free-fall brings flight-path angle near -50°.
# ===========================
# Phase 4: Recovery Exit Parameters
# ===========================
def ias_to_tas(IAS, alt):
    """
    Convert Indicated Airspeed (IAS in knots) to True Airspeed (TAS in ft/s) at a given altitude.

    This function uses the standard sea-level density and interpolates the local air density
    based on altitude. The conversion is based on the square root of the density ratio.
    """
    rho = np.interp(alt, altitude_points, density_points)
    return IAS * kts_to_ftps * math.sqrt(rho_sl / rho)


v_target_exit = ias_to_tas(250.0, 25000)  # Compute target TAS for exit phase (~630 ft/s)

k_exit = 0.7  # gain for pitch recovery during Phase 4
k_speed = 0.5  # gain for speed recovery during Phase 4
tau_exit = 1.0  # lag time constant for turning rate in Phase 4

tol = 0.01  # tolerance in radians for phase transitions (~0.57°)


# ===========================
# Helper Functions
# ===========================
def get_density(alt):
    """
    Interpolate air density (slug/ft³) based on altitude (ft).

    Uses linear interpolation between given altitude_points and density_points.
    """
    return np.interp(alt, altitude_points, density_points)


def compute_TAS(rho):
    """
    Compute True Airspeed (TAS in ft/s) from IAS_knots and local air density.

    The conversion uses the square root of the density ratio (sea-level to current).
    """
    return IAS_knots * kts_to_ftps * math.sqrt(rho_sl / rho)


def compute_drag(rho, speed, multiplier=1.0):
    """
    Compute drag force using the formula:
      F_drag = 0.5 * (multiplier * rho) * speed² * A * C_d
    """
    return 0.5 * multiplier * rho * speed ** 2 * A * C_d


def max_turn_rate_allowed(gamma, TAS):
    """
    Compute the maximum allowable turning rate (rad/s) so that the effective g-force does not exceed g_eff_max.

    The formula used is:
      dγ/dt_max = (g/TAS) * sqrt((g_eff_max*cos(gamma))² - 1)
    where gamma is the current flight-path angle.
    """
    term = (g_eff_max * math.cos(gamma)) ** 2 - 1
    if term < 0:
        return 0.0
    return (g / TAS) * math.sqrt(term)


def rotate_shape(shape, angle):
    """
    Rotate a 2D shape represented as an Nx2 NumPy array by a given angle.
    """
    R = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    return shape @ R.T


def angle_color(angle_deg, g_eff):
    """
    Compute a color tuple for the Ascent/Descent Angle text.

    The color is interpolated from blue (0° → (0,0,1)) to red (50° → (1,0,0)) based on the absolute angle.
    """
    norm = min(abs(angle_deg) / 50.0, 1.0)
    r = norm
    g_val = 0
    b = 1 - norm
    return (r, g_val, b)


def speed_color(TAS):
    """
    Compute a color for the True Airspeed (TAS) text display.

    Color mapping:
      - TAS <= 400 ft/s  -> blue (0, 0, 1)
      - TAS >= 700 ft/s  -> dark green (0, 0.8, 0)
      - Linear interpolation in between.
    """
    lower_bound = 400.0
    upper_bound = 700.0
    if TAS <= lower_bound:
        return (0, 0, 1)
    elif TAS >= upper_bound:
        return (0, 0.8, 0)
    else:
        norm = (TAS - lower_bound) / (upper_bound - lower_bound)
        green = norm * 0.8
        blue = 1 - norm
        return (0, green, blue)


def gforce_color(g_eff):
    """
    Compute a color for the G-force text display.

    Color mapping:
      - g_eff == 0: dark green (0, 0.6, 0)
      - 0 < g_eff <= 1.0: yellow (1, 1, 0)
      - 1.0 < g_eff <= 1.8: interpolate from yellow to red.
    """
    if g_eff == 0:
        return 0, 0.6, 0
    elif g_eff <= 1.0:
        return 1, 1, 0
    else:
        norm = (g_eff - 1.0) / (1.8 - 1.0)
        norm = min(max(norm, 0), 1)
        green = 1 - norm
        return 1, green, 0


# ===========================
# Simulation Setup
# ===========================
dt = 0.01  # simulation time step (s)
t = 0.0  # initial simulation time

# Initial conditions: level flight at 25,000 ft altitude.
x = 0.0
y = 25000.0
gamma = 0.0  # flight-path angle (0 for level flight)
dgamma_dt = 0.0  # initial turning rate

# Prepare lists to record simulation data for later plotting and animation.
time_list = []
x_list = []
y_list = []
gamma_list = []  # flight-path angle in degrees (for display)
speed_list = []  # True Airspeed (TAS) in ft/s
g_eff_list = []  # effective g-force in g's

# --------------------------
# Phase 1: Powered Ascent (Thrust On)
# --------------------------
# In this phase the aircraft climbs under power until it reaches a target climb angle.
while not (t >= 2.0 and gamma >= gamma_target - tol):
    rho = get_density(y)
    TAS = compute_TAS(rho)
    drag = compute_drag(rho, TAS)
    T_required = drag + weight * math.sin(gamma)

    if t < 2.0:
        # For the first 2 seconds, maintain level flight.
        gamma = 0.0
        dgamma_dt = 0.0
    else:
        # Use a proportional control (with lag) to pitch up toward gamma_target.
        desired_turn = k1 * (gamma_target - gamma)
        dgamma_dt += (desired_turn - dgamma_dt) * dt / tau_turn
        cap = max_turn_rate_allowed(gamma, TAS) if TAS > 0 else 0.0
        if dgamma_dt >= 0:
            dgamma_dt = min(dgamma_dt, cap)
        else:
            dgamma_dt = max(dgamma_dt, -cap)
        gamma += dgamma_dt * dt
        if gamma > gamma_target:
            gamma = gamma_target
            dgamma_dt = 0.0

    # Update velocities and positions based on current TAS and gamma.
    vx = TAS * math.cos(gamma)
    vy = TAS * math.sin(gamma)
    x += vx * dt
    y += vy * dt

    # Record simulation data.
    time_list.append(t)
    x_list.append(x)
    y_list.append(y)
    gamma_list.append(math.degrees(gamma))
    speed_list.append(TAS)
    g_eff_list.append((1 / math.cos(gamma)) * math.sqrt(1 + (TAS * dgamma_dt / g) ** 2))

    t += dt

# Save state at end of Phase 1.
x_phase1 = x
y_phase1 = y
gamma_phase1 = gamma  # should be near 50° (target climb angle)
TAS_phase1 = TAS
vx = TAS_phase1 * math.cos(gamma_phase1)
vy = TAS_phase1 * math.sin(gamma_phase1)

# --------------------------
# Phase 2: Parabolic Maneuver (Thrust Off, Variable Drag)
# --------------------------
# In this phase the aircraft reverses its pitch by turning nose-down.
while gamma > gamma_rev_trigger + tol:
    rho = get_density(y)
    speed = math.sqrt(vx ** 2 + vy ** 2)
    TAS = speed
    T_current = 0.0

    # Increase drag if TAS is above the target drag speed.
    M = 1 + k_drag * (TAS - v_target_drag) if TAS > v_target_drag else 1.0
    drag = compute_drag(rho, TAS, multiplier=M)
    if speed > 0:
        drag_x = drag * (vx / speed)
        drag_y = drag * (vy / speed)
    else:
        drag_x, drag_y = 0.0, 0.0

    weight_x = 0.0
    weight_y = -weight
    F_net_x = -drag_x + weight_x
    F_net_y = -drag_y + weight_y
    ax_net = F_net_x / mass
    ay_net = F_net_y / mass

    # Update velocity and position.
    vx += ax_net * dt
    vy += ay_net * dt
    x += vx * dt
    y += vy * dt

    # Flight-path angle becomes the angle of the velocity vector.
    gamma = math.atan2(vy, vx)

    g_eff = 0.0  # In the parabolic maneuver, the aim is weightlessness.

    # Record simulation data.
    time_list.append(t)
    x_list.append(x)
    y_list.append(y)
    gamma_list.append(math.degrees(gamma))
    speed_list.append(TAS)
    g_eff_list.append(g_eff)

    t += dt

# --------------------------
# Phase 3: Ballistic Free-Fall (Thrust Off, Normal Drag)
# --------------------------
# The aircraft now follows a ballistic trajectory until the flight-path angle nears -50°.
while y > 15000 and gamma > math.radians(-50) - tol:
    rho = get_density(y)
    speed = math.sqrt(vx ** 2 + vy ** 2)
    TAS = speed
    T_current = 0.0
    if speed > 0:
        drag = compute_drag(rho, TAS)
        drag_x = drag * (vx / speed)
        drag_y = drag * (vy / speed)
    else:
        drag_x, drag_y = 0.0, 0.0
    weight_x = 0.0
    weight_y = -weight

    F_net_x = -drag_x + weight_x
    F_net_y = -drag_y + weight_y
    ax_net = F_net_x / mass
    ay_net = F_net_y / mass

    vx += ax_net * dt
    vy += ay_net * dt
    x += vx * dt
    y += vy * dt

    gamma = math.atan2(vy, vx) if speed > 0 else 0.0

    g_eff_current = 0.0

    time_list.append(t)
    x_list.append(x)
    y_list.append(y)
    gamma_list.append(math.degrees(gamma))
    speed_list.append(TAS)
    g_eff_list.append(g_eff_current)

    t += dt

# Save state at end of Phase 3.
x_phase3 = x
y_phase3 = y
gamma_phase3 = gamma  # should be near -50° (ready for recovery)
speed_phase3 = math.sqrt(vx ** 2 + vy ** 2)

# --------------------------
# Phase 4: Recovery Exit (Thrust On, Pitch-Up and Accelerate)
# --------------------------
# In the recovery exit phase, the aircraft pitches up and accelerates to the target TAS.
v_target_exit = ias_to_tas(250.0, 25000)  # ~630 ft/s
k_exit = 0.7  # gain for pitch recovery in Phase 4
k_speed = 0.5  # gain for speed recovery in Phase 4

gamma = gamma_phase3
speed = speed_phase3
vx = speed * math.cos(gamma)
vy = speed * math.sin(gamma)

while (abs(gamma) > tol) or (abs(speed - v_target_exit) > 5.0):
    error_gamma = 0.0 - gamma
    desired_turn = k_exit * error_gamma
    cap = max_turn_rate_allowed(gamma, speed) if speed > 0 else 0.0
    dgamma_dt = max(-cap, min(desired_turn, cap))
    gamma += dgamma_dt * dt

    error_speed = v_target_exit - speed
    a_command = k_speed * error_speed
    rho = get_density(y)
    drag = compute_drag(rho, speed)
    T_required = mass * a_command + drag + weight * math.sin(gamma)
    F_net = T_required - drag - weight * math.sin(gamma)
    a_net = F_net / mass
    speed += a_net * dt

    vx = speed * math.cos(gamma)
    vy = speed * math.sin(gamma)

    x += vx * dt
    y += vy * dt

    time_list.append(t)
    x_list.append(x)
    y_list.append(y)
    gamma_list.append(math.degrees(gamma))
    speed_list.append(speed)
    g_eff_list.append((1 / math.cos(gamma)) * math.sqrt(1 + (speed * dgamma_dt / g) ** 2))

    t += dt

# --------------------------
# Phase 5: Final Level Flight (2 seconds)
# --------------------------
# After recovery, the aircraft enters level flight for 2 seconds.
t_phase5 = 0.0
while t_phase5 < 2.0:
    gamma = 0.0  # level flight
    dgamma_dt = 0.0
    speed = v_target_exit  # maintain target TAS (~630 ft/s)
    vx = speed * math.cos(gamma)
    vy = speed * math.sin(gamma)

    x += vx * dt
    y += vy * dt

    T_current = 0.0
    g_eff = 1.0  # in level flight, occupants feel 1g

    time_list.append(t)
    x_list.append(x)
    y_list.append(y)
    gamma_list.append(math.degrees(gamma))
    speed_list.append(speed)
    g_eff_list.append(g_eff)

    t += dt
    t_phase5 += dt

# ===========================
# Time Compression for Animation
# ===========================
# Compress the simulation time into a fixed number of animation frames.
num_anim_frames = 900
frame_indices = np.linspace(0, len(time_list) - 1, num_anim_frames, dtype=int)

# ===========================
# Plot & Animate the Trajectory
# ===========================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('Horizontal Distance (ft)')
ax.set_ylabel('Altitude (ft)')
ax.set_title('"Vomit Comet" Aircraft Flight \n'
             'Achieving Sub-Orbital Zero-G Environment')
ax.set_facecolor('lightskyblue')
ax.set_xlim(min(x_list) - 100, max(x_list) + 100)
ax.set_ylim(min(y_list) - 1000, max(y_list) + 1000)

# Define a custom aircraft object shape (a triangular arrow) for display.
arrow_shape = np.array([[-420, -105],
                        [0, 0],
                        [-420, 105]])
aircraft_patch = Polygon(arrow_shape, closed=True, color='red')
ax.add_patch(aircraft_patch)

# Create separate text objects for displaying simulation info.
info_text = ax.text(0.80, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
angle_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
speed_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
gforce_text = ax.text(0.02, 0.89, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

traj_line, = ax.plot([], [], 'b-', lw=1)


def init():
    """
    Initialization function for the animation.

    Clears the trajectory line and resets the aircraft patch and info texts.
    """
    traj_line.set_data([], [])
    aircraft_patch.set_xy(arrow_shape)
    info_text.set_text('')
    angle_text.set_text('')
    speed_text.set_text('')
    gforce_text.set_text('')
    return traj_line, aircraft_patch, info_text, angle_text, speed_text, gforce_text


def animate(frame):
    """
    Animation function called for each frame.

    Updates the trajectory, rotates and translates the aircraft shape, and
    updates the displayed simulation information (time, altitude, distance,
    flight-path angle, TAS, and G-force).
    """
    i = frame_indices[frame]
    traj_line.set_data([x_list[:i]], [y_list[:i]])
    rotated_arrow = rotate_shape(arrow_shape, math.radians(gamma_list[i]))
    translated_arrow = rotated_arrow + np.array([x_list[i], y_list[i]])
    aircraft_patch.set_xy(translated_arrow)

    info_str = (f'Sim Time: {time_list[i]:.2f} s\n'
                f'Altitude: {y_list[i]:.0f} ft\n'
                f'Distance: {x_list[i]:.0f} ft')
    info_text.set_text(info_str)

    angle_val = gamma_list[i]
    angle_str = f'Ascent/Descent Angle: {angle_val:.1f}°'
    angle_color_val = angle_color(angle_val, g_eff_list[i])
    angle_text.set_text(angle_str)
    angle_text.set_color(angle_color_val)

    TAS_val = speed_list[i]
    speed_str = f'True Airspeed: {TAS_val:.0f} ft/s'
    speed_color_val = speed_color(TAS_val)
    speed_text.set_text(speed_str)
    speed_text.set_color(speed_color_val)

    gforce_val = g_eff_list[i]
    gforce_str = f'G-force: {gforce_val:.2f} g'
    gforce_color_val = gforce_color(gforce_val)
    if gforce_val == 0:
        gforce_fontsize = 12
        gforce_fontweight = 'bold'
    else:
        gforce_fontsize = 10
        gforce_fontweight = 'normal'
    gforce_text.set_text(gforce_str)
    gforce_text.set_color(gforce_color_val)
    gforce_text.set_fontsize(gforce_fontsize)
    gforce_text.set_fontweight(gforce_fontweight)

    return traj_line, aircraft_patch, info_text, angle_text, speed_text, gforce_text


ani = animation.FuncAnimation(fig, animate, frames=num_anim_frames,
                              init_func=init, interval=33.33, blit=True)
plt.show()