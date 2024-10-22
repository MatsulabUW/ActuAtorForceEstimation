import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import matplotlib.animation as animation
from automembrane.energy import ClosedPlaneCurveMaterial
from automembrane.energy import OpenPlaneCurveMaterial
from automembrane.integrator import fwd_euler_integrator
from plot import plot_coord_force, plot_frame, shift_coords
import os



# def relax(coords, Kb, Ksg, Ksl, dt, n_iter, boundary, file_stem, plot_interval=100):
#     # Instantiate material properties
#     parameters = {
#         "Kb": Kb / 4,
#         "Ksg": Ksg,
#         "Ksl": Ksl,
#         "boundary": boundary
#     }
#     # Perform energy relaxation
#     if n_iter > 0:
#         coord_log, energy_log, force_log = fwd_euler_integrator(
#             coords,
#             OpenPlaneCurveMaterial(**parameters),
#             n_steps=n_iter,
#             dt=dt,
#             save_trajectory=True,
#         )

#     for i in range(0, n_iter, plot_interval):
#         coords_at_i = coord_log[i]
#         forces_at_i = np.sum(force_log[i], axis=0)

#         plot_coord_force(
#             file_stem=file_stem,
#             coord=coords_at_i,
#             force=forces_at_i,
#             iteration=int(i / plot_interval),
#             save_image=True,
#         )



def relax(
    coords, Kb, Ksg, Ksl, dt, n_iter, boundary, file_stem, 
    plot_interval=100, output_dir="figures/bcs_test", movie_filename="simulation.mp4", 
    save_image=False, save_movie=False, save_energy=False, save_data=False
):
    # Instantiate material properties
    parameters = {
        "Kb": Kb / 4,
        "Ksg": Ksg,
        "Ksl": Ksl,
        "boundary": boundary
    }

    # Perform energy relaxation
    if n_iter > 0:
        coord_log, energy_log, force_log = fwd_euler_integrator(
            coords,
            OpenPlaneCurveMaterial(**parameters),
            n_steps=n_iter,
            dt=dt,
            save_trajectory=True,
        )

    fig, ax = plt.subplots(figsize=(5, 5))
    images = []

    # Save parameters to a text file
    save_parameters_to_file(parameters, dt, n_iter, output_dir)

    # Define consistent limits based on the data
    coord_x, coord_y = shift_coords(coords)
    xlim = (min(coord_x) - 50, max(coord_x) + 50)
    ylim = (max(coord_y) + 50, min(coord_y) - 50)

    for i in range(0, n_iter+1, plot_interval):
        coords_at_i = coord_log[i]
        forces_at_i = np.sum(force_log[i], axis=0)

        if save_image or save_movie:
            # Clear the axes for each frame
            ax.clear()
            plot_frame(ax, file_stem, coords_at_i, forces_at_i, xlim, ylim)

            if save_image:
                # Save the image if requested
                output_path = os.path.join(output_dir, f"plot_iter_{i:06d}.png")
                plt.savefig(output_path)
                print(f"Image saved at {output_path}")

            if save_movie:
                # Capture the current frame for the movie
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                images.append(image)

    if save_movie and images:
        # Create the animation if requested
        create_animation(images, output_dir, movie_filename, fig)

    if save_energy:
        print(energy_log[1,])
        plot_energy(energy_log, n_iter, output_dir)

    if save_data:
        # Save the coordinates of the last time step
        save_last_coordinates(coord_log[-1], output_dir)

    plt.close(fig)



def create_animation(images, output_dir, movie_filename, fig):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))

    def animate(i):
        ax.clear()
        ax.imshow(images[i])
        ax.set_axis_off()  # Remove axis for clean video frames

    ani = animation.FuncAnimation(fig, animate, frames=len(images), interval=100)
    movie_path = os.path.join(output_dir, movie_filename)
    ani.save(movie_path, fps=10, extra_args=['-vcodec', 'libx264'])
    print(f"Movie saved at {movie_path}")



def plot_energy(energy_log, n_iter, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    initial_values = energy_log[0, :]
    normal_energy_log = energy_log / initial_values
    # normal_energy_log = energy_log 

    # Assuming energy_log is a (n_iter+1, 3) array where each column is a component
    iterations = np.arange(n_iter+1)
    ax.plot(iterations, normal_energy_log[:, 0], label='Bending Energy', color='blue')
    ax.plot(iterations, normal_energy_log[:, 1], label='Surface Energy', color='green')
    ax.plot(iterations, normal_energy_log[:, 2], label='Regularization Energy', color='red')

    ax.set_xlabel('Iteration Step')
    ax.set_ylabel('Normalized Energy')
    ax.set_title('Energy Components Over Iteration Steps')
    ax.legend()
    # ax.set_ylim(-0.05, 1.05)

    energy_plot_path = os.path.join(output_dir, "energy_plot.png")
    plt.savefig(energy_plot_path)
    print(f"Energy plot saved at {energy_plot_path}")
    plt.close(fig)



def save_parameters_to_file(parameters, dt, n_iter, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file path
    parameter_file_path = os.path.join(output_dir, "parameters.txt")

    # Write parameters to the file
    with open(parameter_file_path, 'w') as file:
        file.write("Simulation Parameters:\n")
        file.write(f"dt: {dt}\n")
        file.write(f"n_iter: {n_iter}\n")
        for key, value in parameters.items():
            file.write(f"{key}: {value}\n")

    print(f"Parameters saved at {parameter_file_path}")


def save_last_coordinates(coords, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the file path
    coordinates_file_path = os.path.join(output_dir, "relaxed_coords.npy")

    # Save the coordinates as a NumPy array
    np.save(coordinates_file_path, coords)

    print(f"Last coordinates saved at {coordinates_file_path}")



