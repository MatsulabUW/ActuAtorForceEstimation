import numpy as np
from scipy.interpolate import interp1d

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt



def align_and_compare_vectors(
    model_points,         # Array of model (x, y) coordinates, shape (N, 2)
    model_vectors,        # Array of model vector values (f_m_x, f_m_y), shape (N, 2)
    calc_points,          # Array of calculated (x', y') coordinates, shape (M, 2)
    calc_vectors,         # Array of calculated vector values (f_c_x', f_c_y'), shape (M, 2)
    num_points=100,       # Number of points to interpolate and align
    plot=False,           # Option to plot the difference
    add_weight=False
):
    """
    Aligns and compares model and calculated vectors along a parameterized curve.
    
    Parameters:
    - model_points: np.array of shape (N, 2), coordinates (x, y) for the model data.
    - model_vectors: np.array of shape (N, 2), vector values f_m(x, y) for model data.
    - calc_points: np.array of shape (M, 2), coordinates (x', y') for calculated data.
    - calc_vectors: np.array of shape (M, 2), vector values f_c(x', y') for calculated data.
    - num_points: int, number of interpolation points for alignment (default=100).
    - plot: bool, whether to plot the vector differences (default=True).
    
    Returns:
    - common_u: np.array of parameterized points.
    - differences: np.array of vector norm differences at each common_u.
    """

    # Calculate arc length parameterization for model data
    model_arc_lengths = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(model_points, axis=0)**2, axis=1))])
    model_u = model_arc_lengths / model_arc_lengths[-1]  # Normalize to [0, 1]

    # Calculate arc length parameterization for calculation data
    calc_arc_lengths = np.cumsum(np.r_[0, np.sqrt(np.sum(np.diff(calc_points, axis=0)**2, axis=1))])
    calc_u = calc_arc_lengths / calc_arc_lengths[-1]  # Normalize to [0, 1]

    # Define common parameter values
    common_u = np.linspace(0, 1, num_points)

    # Interpolate model points and vectors
    model_x_interp = interp1d(model_u, model_points[:, 0], kind='linear')(common_u)
    model_y_interp = interp1d(model_u, model_points[:, 1], kind='linear')(common_u)
    model_fm_x_interp = interp1d(model_u, model_vectors[:, 0], kind='linear')(common_u)
    model_fm_y_interp = interp1d(model_u, model_vectors[:, 1], kind='linear')(common_u)

    # Interpolate calculated points and vectors
    calc_x_interp = interp1d(calc_u, calc_points[:, 0], kind='linear')(common_u)
    calc_y_interp = interp1d(calc_u, calc_points[:, 1], kind='linear')(common_u)
    calc_fc_x_interp = interp1d(calc_u, calc_vectors[:, 0], kind='linear')(common_u)
    calc_fc_y_interp = interp1d(calc_u, calc_vectors[:, 1], kind='linear')(common_u)

    # Calculate vector differences
    differences = np.sqrt((model_fm_x_interp - calc_fc_x_interp)**2 + (model_fm_y_interp - calc_fc_y_interp)**2)

    if add_weight:
        weight = np.sqrt(model_fm_x_interp**2 + model_fm_y_interp**2)
        differences *= weight


    # Optional: Plot the vector differences along the curve
    if plot:
        plt.plot(common_u, differences, label='Vector Norm Differences')
        plt.xlabel('Normalized Arc Length')
        plt.ylabel('Difference in Vector Magnitude')
        plt.legend()
        plt.title('Comparison of Model and Calculated Vectors')
        plt.show()

    return common_u, differences


