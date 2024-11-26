import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import numpy.typing as npt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def shift_coords(coords: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # shift coordinate so that the data matches the image

    x_shift = 765*0.6
    y_shift = 940*0.6

    x = coords[:, 0]
    y = coords[:, 1]

    x -= x_shift
    y += y_shift

    return x, y

def load_image(file_stem: str):
    image_path = os.path.join('raw_images', f'{file_stem}.png')
    image = plt.imread(image_path)
    # If the image is in RGB(A), convert it to grayscale by taking only one channel
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Assuming grayscale information is in the first channel
    return image

def resize_image(image, scale_factor):
    return ndimage.zoom(image, scale_factor)

def plot_dots(
    file_stem: str,
    original_coords: npt.NDArray[np.float64],
    relaxed_coords: npt.NDArray[np.float64],
    # xlim=(100, 600), 
    ylim=(300, 0),
    xlim=None,
    # ylim=None,
):
    fig, ax = plt.subplots(figsize=(5, 5))
    ori_x, ori_y = shift_coords(original_coords)
    rel_x, rel_y = shift_coords(relaxed_coords)
    
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6/1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.scatter(ori_x, ori_y, color='black', s=5, alpha=0.8, label='Original')
    ax.scatter(rel_x, rel_y, color='red',   s=5, alpha=0.8, label='Relaxed')    
    
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.legend()
    plt.show()


def plot_coord_force(
    file_stem: str,
    coord: npt.NDArray[np.float64],
    force: npt.NDArray[np.float64],
    xlim=None,
    ylim=(300, -50),
    iteration: int = 0,
    save_image: bool = False,
    scale_factor: float = 0.003,
    output_filename: str = None,
):
    fig, ax = plt.subplots(figsize=(5, 5))
    coord_x, coord_y = shift_coords(coord)
    
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6/1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    ax.scatter(coord_x, coord_y, color='red', s=2, alpha=0.8)


    f_mag = np.linalg.norm(force, axis=1)

    q = ax.quiver(
        coord_x,
        coord_y,
        -force[:, 0],
        -force[:, 1],
        f_mag,
        cmap=mpl.cm.viridis_r,
        angles="xy",
        # label="force",
        scale=scale_factor,
        # scale=0.003,
        # scale=0.03,
        scale_units="xy",
        width=0.005,
    )
    # cbar = fig.colorbar(
    #     q,
    #     ax=ax,
    # )
    # cbar.ax.get_yaxis().labelpad = 20
    # cbar.ax.set_ylabel(r"Force Density $\mathrm{(pN/nm^2)}$", rotation=270)

    
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    # ax.legend()


    if save_image:
        output_path = os.path.join("figures/bcs_test", f"blabla.png")
        plt.savefig(output_path)
        # print(f"Image saved at {output_path}")

    if output_filename: 
        output_path = os.path.join("figures", output_filename)
        plt.savefig(output_path)
        print(f"Image saved at {output_path}")

    plt.show()
    plt.close(fig)



def plot_coord_curve(
    file_stem: str,
    coords: npt.NDArray[np.float64],
    curves: npt.NDArray[np.float64],
    xlim=None,
    ylim=(300, -50),
    iteration: int = 0,
    save_image: bool = False,
    scale_factor: float = 0.003,
    cmap: str = 'viridis_r',
):
    fig, ax = plt.subplots(figsize=(8, 8))
    coords_x, coords_y = shift_coords(coords)
    
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6/1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


    # Prepare edges (line segments) using coordinates
    # Assuming consecutive pairs of nodes are connected by edges
    segments = [
        [(coords_x[i], coords_y[i]), (coords_x[i+1], coords_y[i+1])]
        for i in range(len(coords_x) - 1)
    ]

    # Create a LineCollection from the segments and color by curves
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=np.min(curves), vmax=np.max(curves)))
    lc.set_array(curves)  # Use the curves array for coloring the edges
    lc.set_linewidth(5)   # Set line width (adjust as necessary)
    
    # Add the LineCollection to the plot
    ax.add_collection(lc)
    
    # Add a color bar to indicate the mapping of values in `curves`
    fig.colorbar(lc, ax=ax, label='Curve Values')

    # Scatter nodes for clarity (optional)
    # ax.scatter(coords_x, coords_y, color='red', s=2, alpha=0.8)


    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    # ax.legend()


    if save_image:
        output_path = os.path.join("figures/coord_curve.png")
        plt.savefig(output_path)
        # print(f"Image saved at {output_path}")

    plt.show()
    plt.close(fig)




def plot_coord_curve_force(
    file_stem: str,
    coords: npt.NDArray[np.float64],
    curves: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    xlim=None,
    ylim=(300, -50),
    iteration: int = 0,
    scale_factor: float = 0.003,
    cmap: str = 'viridis',
    output_filename: str = None,
    force_vmin=None,
    force_vmax=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    coords_x, coords_y = shift_coords(coords)
    
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6/1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


    # Prepare edges (line segments) using coordinates
    segments = [
        [(coords_x[i], coords_y[i]), (coords_x[i+1], coords_y[i+1])]
        for i in range(len(coords_x) - 1)
    ]

    # Create a LineCollection from the segments and color by curves
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=np.min(curves), vmax=np.max(curves)))
    lc.set_array(curves)  # Use the curves array for coloring the edges
    lc.set_linewidth(5)   # Set line width (adjust as necessary)
    
    # Add the LineCollection to the plot
    ax.add_collection(lc)
    
    # fig.colorbar(lc, ax=ax, label='Curve Values')



    f_mag = np.linalg.norm(forces, axis=1)
    norm = Normalize(vmin=force_vmin if force_vmin is not None else np.min(f_mag),
                     vmax=force_vmax if force_vmax is not None else np.max(f_mag))


    q = ax.quiver(
        coords_x,
        coords_y,
        -forces[:, 0],
        -forces[:, 1],
        f_mag,
        cmap=mpl.cm.viridis_r,
        norm=norm,
        angles="xy",
        # label="force",
        scale=scale_factor,
        # scale=0.003,
        # scale=0.03,
        scale_units="xy",
        width=0.005,
    )



    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    # ax.legend()



    # if save_image:
    #     output_path = os.path.join("figures/coord_curve_force.png")
    #     plt.savefig(output_path)
        # print(f"Image saved at {output_path}")

    if output_filename: 
        output_path = os.path.join("figures", output_filename)
        plt.savefig(output_path)
        print(f"Image saved at {output_path}")

    plt.show()
    plt.close(fig)


def plot_coord_curve_force_center(
    file_stem: str,
    coords: npt.NDArray[np.float64],
    curves: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    xlim=None,
    ylim=(300, -50),
    iteration: int = 0,
    scale_factor: float = 0.003,
    cmap: str = 'viridis',
    output_filename: str = None,
    force_vmin=None,
    force_vmax=None,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    coords_x, coords_y = shift_coords(coords)
    
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6/1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


    # Prepare edges (line segments) using coordinates
    segments = [
        [(coords_x[i], coords_y[i]), (coords_x[i+1], coords_y[i+1])]
        for i in range(len(coords_x) - 1)
    ]

    # Create a LineCollection from the segments and color by curves
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(vmin=np.min(curves), vmax=np.max(curves)))
    lc.set_array(curves)  # Use the curves array for coloring the edges
    lc.set_linewidth(5)   # Set line width (adjust as necessary)
    
    # Add the LineCollection to the plot
    ax.add_collection(lc)
    
    # fig.colorbar(lc, ax=ax, label='Curve Values')



    f_mag = np.linalg.norm(forces, axis=1)
    norm = Normalize(vmin=force_vmin if force_vmin is not None else np.min(f_mag),
                     vmax=force_vmax if force_vmax is not None else np.max(f_mag))


    q = ax.quiver(
        coords_x,
        coords_y,
        -forces[:, 0],
        -forces[:, 1],
        f_mag,
        cmap=mpl.cm.viridis_r,
        norm=norm,
        angles="xy",
        # label="force",
        scale=scale_factor,
        # scale=0.003,
        # scale=0.03,
        scale_units="xy",
        width=0.005,
    )


    center_x, center_y = shift_coords(center)
    ax.scatter(center_x, center_y, c=curves, cmap=cmap, norm=plt.Normalize(vmin=np.min(curves), vmax=np.max(curves)), s=2, alpha=0.8)


    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    # ax.legend()



    # if save_image:
    #     output_path = os.path.join("figures/coord_curve_force.png")
    #     plt.savefig(output_path)
        # print(f"Image saved at {output_path}")

    if output_filename: 
        output_path = os.path.join("figures", output_filename)
        plt.savefig(output_path)
        print(f"Image saved at {output_path}")

    plt.show()
    plt.close(fig)    



def display_forces_side_by_side(
    file_stem: str,
    coords: npt.NDArray[np.float64],
    forces: npt.NDArray[np.float64],
    xlim=None,
    ylim=(300, -50),
    save_image: bool = False,
    scale_factor: float = 0.003,
    save=False, output_dir="figures", boundary=None, edge=None
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    coord_x, coord_y = shift_coords(coords)

    for i, force_label in enumerate(["Bending Force", "Tensile Force"]):
        
        image = load_image(file_stem)
        resized_image = resize_image(image, 0.6/1.1)
        
        ax = axes[i]
        ax.imshow(resized_image, cmap='gray', alpha=0.6)

        if xlim:
            ax.set_xlim(xlim)

        if ylim:
            ax.set_ylim(ylim)

        ax.scatter(coord_x, coord_y, color='red', s=2, alpha=0.8)

        # Calculate the magnitude of the force
        f_mag = np.linalg.norm(forces[i, :, :], axis=1)

        # Display the quiver for the force component
        q = ax.quiver(
            coord_x,
            coord_y,
            forces[i, :, 0],
            forces[i, :, 1],
            f_mag,
            cmap=plt.cm.viridis_r,
            angles="xy",
            scale=scale_factor,
            scale_units="xy",
            width=0.005,
        )

        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title(force_label, fontsize=20)


    # plt.tight_layout()
    plt.subplots_adjust(wspace=-0.1)

    if save:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"forces_bc_{boundary}_{edge}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Figure saved as {output_path}")


    plt.show()
    plt.close(fig)





def plot_frame(ax, file_stem, coord, force, xlim, ylim):
    coord_x, coord_y = shift_coords(coord)
    image = load_image(file_stem)
    resized_image = resize_image(image, 0.6 / 1.1)
    
    ax.imshow(resized_image, cmap='gray', alpha=0.6)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    f_mag = np.linalg.norm(force, axis=1)

    ax.scatter(coord_x, coord_y, color='red', s=2, alpha=0.8)
    ax.quiver(
        coord_x,
        coord_y,
        force[:, 0],
        force[:, 1],
        f_mag,
        cmap=plt.cm.viridis_r,
        angles="xy",
        # scale=0.001,
        scale=0.01,
        # scale=0.03,
        # scale=0.05,
        scale_units="xy",
        width=0.005,
    )



