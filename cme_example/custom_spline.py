import numpy as np
from scipy.interpolate import BSpline, splev
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sample import make_parabola

# Generate example 2D data
# np.random.seed(0)
# x = np.linspace(0, 10, 50)
# y = np.sin(x) + 0.1 * np.random.normal(size=x.size)



N_node = 100
L_etoe = 100
height = 50
noise_level = 1
ori_coords = make_parabola(N_node = N_node, L_etoe = L_etoe, height = height, point_dist = "unif", noise_level = noise_level)
x = ori_coords[:,0]
y = ori_coords[:,1]

sorted_indices = np.argsort(x)
x = x[sorted_indices]
y = y[sorted_indices]

# # Define the B-spline parameters
# degree = 4  # Set degree to 4 to allow for fourth-order derivatives
# num_knots = 8
# knots = np.linspace(x[1], x[-2], num_knots - 2 * degree)  # Place knots within data range

# # Fit the data with LSQUnivariateSpline
# spline = LSQUnivariateSpline(x, y, knots, k=degree)

# # Plot the data and the fitted B-spline
# plt.figure(figsize=(10, 6))
# plt.plot(x, y, 'o', label='Data', markersize=5)
# plt.plot(x, spline(x), '-', label='Fitted B-spline (Degree 4)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('B-spline Interpolation with LSQUnivariateSpline (Degree 4)')
# plt.legend()
# plt.show()




# # Generate example 2D data
# np.random.seed(0)
# x = np.linspace(0, 10, 50)
# y = np.sin(x) + 0.1 * np.random.normal(size=x.size)

# Define B-spline parameters
degree = 4  # Degree of the spline
num_knots = 50  # Number of internal knots
knots = np.linspace(x[1], x[-2], num_knots)  # Place knots within data range

# Augmented knot sequence (add boundary knots)
t = np.concatenate(([x[0]] * degree, knots, [x[-1]] * degree))

# Define regularization parameter
# lambda_ = 1e-3
lambda_ = 1e-1

# Define a finite difference approximation for the fourth derivative penalty
def finite_difference_penalty(spline, x, h=1e-2):
    # Fourth-order finite difference approximation for the fourth derivative
    f_xph2 = splev(x + 2 * h, spline)
    f_xph = splev(x + h, spline)
    f_xmh = splev(x - h, spline)
    f_xmh2 = splev(x - 2 * h, spline)
    f_x = splev(x, spline)

    # Fourth derivative approximation
    fourth_derivative = (f_xph2 - 4 * f_xph + 6 * f_x - 4 * f_xmh + f_xmh2) / h**4
    return np.sum(fourth_derivative**2)

# Updated objective function using finite difference for the penalty term
def objective_with_fd_penalty(coeffs):
    # Create the B-spline with current coefficients
    spline = BSpline(t, coeffs, degree)
    
    # Calculate the data fidelity term
    y_fit = splev(x, spline)
    fidelity_term = np.sum((y - y_fit) ** 2)
    
    # Smoothness penalty using finite difference approximation
    penalty_term = finite_difference_penalty(spline, x)

    # Total objective
    return fidelity_term + lambda_ * penalty_term

# Initial guess for coefficients (zero-initialized)
initial_coeffs = np.zeros(len(t) - degree - 1)

# Run optimization with the new penalty function
result = minimize(objective_with_fd_penalty, initial_coeffs)

# Get optimized coefficients and create the smooth spline
optimal_coeffs = result.x
smooth_spline = BSpline(t, optimal_coeffs, degree)

# Plot the original data and the fitted spline with penalty
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Data', markersize=5)
plt.plot(x, smooth_spline(x), '-', label='Fitted B-spline with Penalty')
plt.xlabel('x')
plt.ylabel('y')
plt.title('B-spline Fit with Fourth-Derivative Penalty (Finite Difference)')
plt.legend()
plt.show()


