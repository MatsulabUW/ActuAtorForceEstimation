import numpy as np
import matplotlib.pyplot as plt


def make_straight_line(
    N_node : int,	# number of nodes
    L_etoe : float, # end-to-end distance
    point_dist: str = "unif"
) -> np.ndarray:

	if point_dist == "unif":
		x_coords = np.linspace(0, L_etoe, N_node)
	elif point_dist == "rand":
		x_coords = np.random.uniform(0, L_etoe, N_node)
		x_coords = np.sort(x_coords)
	else:
		raise ValueError("Unsupported distribution type: ", point_dist)

	y_coords = np.zeros(N_node)
	coords = np.column_stack((x_coords, y_coords))
	return coords


def make_parabola(
    N_node : int,	# number of nodes
    L_etoe : float, # end-to-end distance
    height: float, 
    point_dist: str = "unif",
    noise_level: float = 0.0, 
    noise_type: str = "normal" # "normal" for Gaussian, "uniform" for uniform
) -> np.ndarray:

	if point_dist == "unif":
		x_coords = np.linspace(0, L_etoe, N_node)
	elif point_dist == "rand":
		x_coords = np.random.uniform(0, L_etoe, N_node)
		x_coords = np.sort(x_coords)
	else:
		raise ValueError("Unsupported distribution type: ", point_dist)

	x_centered = x_coords - L_etoe / 2.0
	y_coords = height * (1 - (x_centered / (L_etoe / 2))**2)


	# Add noise to the coordinates
	if noise_level > 0:
		np.random.seed(1234)
		if noise_type == "normal":
			noise_x = np.random.normal(0, noise_level, size=x_coords.shape)  # Gaussian noise
			noise_y = np.random.normal(0, noise_level, size=y_coords.shape)  # Gaussian noise
		elif noise_type == "uniform":
			noise_x = np.random.uniform(-noise_level, noise_level, size=x_coords.shape)  # Uniform noise
			noise_y = np.random.uniform(-noise_level, noise_level, size=y_coords.shape)  # Uniform noise
		else:
			raise ValueError("Unsupported noise type: ", noise_type)

		# Apply noise to the x and y coordinates
		x_coords += noise_x
		y_coords += noise_y


	coords = np.column_stack((x_coords, y_coords))
	return coords


def plot_system(coords: np.ndarray, force: np.ndarray):

    scale_factor = 5e3
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    summed_force = np.sum(force, axis=0)
    fx = summed_force[:, 0] * scale_factor
    fy = summed_force[:, 1] * scale_factor
    force_magnitudes = np.sqrt(fx**2 + fy**2) / scale_factor
    
    plt.figure(figsize=(8, 2))
    plt.scatter(x_coords, y_coords, color='blue', label='Nodes', s=50)  # Plot nodes
    plt.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=2)
    plt.quiver(x_coords, y_coords, fx, fy, color='red', angles='xy', scale_units='xy', scale=1, label='Force')  # Plot force arrows
    
    # Display force magnitudes below each node
    for i, (x, y, magnitude) in enumerate(zip(x_coords, y_coords, force_magnitudes)):
        plt.text(x, y - 2, f'{magnitude:.2e}', ha='center', va='top', fontsize=8, color='red')


    plt.legend()
    # plt.grid(True)
    plt.ylim(-10,10)
    plt.axis('equal')  # Keep aspect ratio to 1:1


    # Show the plot
    plt.show()


def plot_parabola(coords: np.ndarray, force: np.ndarray = None, scale = 10):

    scale_factor = 1e5
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
   
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, color='blue', label='Nodes', s=50)
    plt.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=2)

    if force is not None:
	    summed_force = np.sum(force, axis=0)
	    fx = summed_force[:, 0] * scale_factor
	    fy = summed_force[:, 1] * scale_factor
	    force_magnitudes = np.sqrt(fx**2 + fy**2) / scale_factor

	    plt.quiver(x_coords, y_coords, fx, fy, color='red', angles='xy', scale_units='xy', scale=scale, label='Force')  # Plot force arrows
    
	    # # Display force magnitudes below each node
	    # for i, (x, y, magnitude) in enumerate(zip(x_coords, y_coords, force_magnitudes)):
	    #     plt.text(x, y - 2, f'{magnitude:.2e}', ha='center', va='top', fontsize=8, color='red')


    plt.legend()
    # plt.grid(True)
    plt.xlim(-60,160)
    plt.ylim(-30,70)
    # plt.axis('equal')  # Keep aspect ratio to 1:1


    # Show the plot
    plt.show()



def plot_parabola_ax(
		coords: np.ndarray, 
		force: np.ndarray, 
		ax, mesh_size, 
		force_type, scale,
		ori_coords: np.ndarray = None):
    scale_factor = 1e5
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    fx = force[:, 0] * scale_factor
    fy = force[:, 1] * scale_factor
    force_magnitudes = np.sqrt(fx**2 + fy**2) / scale_factor
    
    # Use the passed axis to plot
    ax.scatter(x_coords, y_coords, color='blue', label='Nodes', s=50)
    ax.plot(x_coords, y_coords, color='blue', linestyle='-', linewidth=2)
    ax.quiver(x_coords, y_coords, fx, fy, color='red', angles='xy', scale_units='xy', scale=scale, label='Force')  # Plot force arrows
    
    if ori_coords is not None:
        x_ori_coords = ori_coords[:, 0]
        y_ori_coords = ori_coords[:, 1]
        ax.scatter(x_ori_coords, y_ori_coords, edgecolor='black', facecolors='none', label='Original', s=50, lw=2)

    ax.legend()
    ax.set_xlim(-60, 160)
    ax.set_ylim(-30, 70)
    
    # Set title with force type and mesh size
    ax.set_title(f"{force_type} (mesh size: {mesh_size})")



def plot_norm_side_by_side(N_node_values, force_sums_bend, force_sums_surf, axes):

	positive_force_bend = [fs[0] for fs in force_sums_bend]
	negative_force_bend = [fs[1] for fs in force_sums_bend]
	net_force_bend 		= [fs[2] for fs in force_sums_bend]

	positive_force_surf = [fs[0] for fs in force_sums_surf]
	negative_force_surf = [fs[1] for fs in force_sums_surf]
	net_force_surf		= [fs[2] for fs in force_sums_surf]

	# Plot for bending forces
	axes[0].plot(N_node_values, positive_force_bend, label='Bending force (Positive)', marker='o', linestyle='--', markersize=8)
	axes[0].plot(N_node_values, negative_force_bend, label='Bending force (Negative)', marker='o', linestyle='--', markersize=8)
	axes[0].plot(N_node_values, net_force_bend, label='Bending force (Net)', marker='o', linestyle='-', markersize=8)

	axes[0].set_xlabel('Number of Vertices')
	axes[0].set_ylabel('Sum of Normal Forces')
	axes[0].set_title('Bending Forces')
	axes[0].legend()
	axes[0].grid(True)

	# Plot for surface forces
	axes[1].plot(N_node_values, positive_force_surf, label='Surface force (Positive)', marker='o', linestyle='--', markersize=8)
	axes[1].plot(N_node_values, negative_force_surf, label='Surface force (Negative)', marker='o', linestyle='--', markersize=8)
	axes[1].plot(N_node_values, net_force_surf, label='Surface force (Net)', marker='o', linestyle='-', markersize=8)
	axes[1].set_xlabel('Number of Vertices')
	axes[1].set_ylabel('Sum of Normal Forces')
	axes[1].set_title('Surface Forces')
	axes[1].legend()
	axes[1].grid(True)


	# Set symmetric y-limits around zero
	y_max_bend = max(abs(min(negative_force_bend)), max(positive_force_bend)) * 1.1
	y_max_surf = max(abs(min(negative_force_surf)), max(positive_force_surf)) * 1.1

	axes[0].set_ylim(-y_max_bend, y_max_bend)  # Bending forces y-limits
	axes[1].set_ylim(-y_max_surf, y_max_surf)  # Surface forces y-limits

	# Remove the grid on the y-axis, but keep the x-axis grid if desired
	axes[0].grid(False)  # Remove all grid lines
	axes[1].grid(False)

	# Add a bold horizontal line at y=0
	axes[0].axhline(0, color='black', linewidth=1)  # Add bold y=0 line on first plot
	axes[1].axhline(0, color='black', linewidth=1)  # Add bold y=0 line on second plot
