"""DREDGE-MOD: A cosmology-oriented modification of the DREDGE package

Introduction:
-------------

This is a modified version of the DREDGE package [1] for geospatial ridge
estimation, which is itself an extension of the subspace-constrained mean 
shift algortihm [2] previously applied to, an extended in, cosmology [3, 4].
Additional extensions for this modified version include the application of
a maximum likelihood cross-validation to optimize the bandwidth, as well
as the use of multiprocessing capabilities for parallelizable parts.

The original package is available on PyPI (https://pypi.org/project/dredge/)
and through its own GitHub repository (https://github.com/moews/dredge).

Quickstart:
-----------
The two-dimensional set of coordinates fed into DREDGE-MOD has to be
provided in the form of a NumPy array with two columns, with the latitudes
in the first and the longitudes in the second column. Additionally, five
optional parameters can be manually set by the user:

(1) The parameter 'neighbors' specifies the number of nearest neighbors
    that should be used to calculate the optimal bandwidth if the latter
    is not provided by the user. The default number of neighbors is 10.
    
(2) The parameter 'bandwidth' provides the bandwidth that is used for the 
    kernel density estimator and Gaussian kernel evaluations. By default,
    an optimal bandwidth using the average distance to a number of neighbors
    across all points in the provided dataset is calculated, with the number
    of neighbors given by the parameter 'neighbors' explained above.
S
(3) The parameter 'convergence' specifies the convergence threshold to
    determine when to stop iterations and return the density ridge points.
    If the resulting density ridges don't follow clearly visible lines,
    this parameter can be set to a lower value. The default is 0.01.

(4) The parameter 'percentage' should be set if only density ridge points
    from high-density regions, as per a kernel density estimate of the
    provided set of coordinates, are to be returned. If, fore example, the
    parameter is set to '5', the density ridge points are evaluated via
    the kernel density estimator, and only those above the 95th percentile,
    as opposed to all of them as the default, are returned to the user.
    
(5) The parameter 'distance' can be set if a project is not dealing with 
    latitude-longitude datasets. It can can be set to either 'euclidean' or
    'haversine', with the latter being the default value.
    
(6) The parameter 'n_process' can be set to enable multiprocessing on more
    than one core in order to speed up computation times. The default is
    zero, indicating no multiprocessing, and different positive integers
    can be set to specify the number of cores that should be used.

(7) The parameter 'mesh_size' can be set to specify the number of points
    that should be used to generate the initial mesh from which the ridges
    are formed over the course of multiple iterations. By default, this is
    set to a number that enables reasonably fast computing times, but larger
    numbers of mesh points allow for more complete ridges.

A simple example for using DREDGE-MOD looks like this:

    --------------------------------------------------------------
    |  from dredge-mod import filaments                          |
    |                                                            |
    |  ridges = filaments(coordinates = your_point_coordinates)  |
    |                                                            |
    --------------------------------------------------------------

Authors:
--------

Ben Moews
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh

Andy Lawler
Dept. of Statistical Science
College of Arts and Sciences
Baylor University

Morgan A. Schmitz
CosmoStat lab
Astrophysics Dept.
CEA Paris-Saclay


References:
-----------
[1] Moews, B. et al. (2019): "Filaments of crime: Informing policing via
    thresholded ridge estimation", JQC (under review), arXiv:1907.03206
[2] Ozertem, U. and Erdogmus, D. (2011): "Locally defined principal curves 
    and surfaces", JMLR, Vol. 12, pp. 1249-1286
[3] Chen, Y. C. et al. (2015), "Cosmic web reconstruction through density 
    ridges: Method and algorithm", MNRAS, Vol. 454, pp. 1140-1156
[4] Chen, Y. C. et al. (2016), "Cosmic web reconstruction through density 
    ridges: Catalogue", MNRAS, Vol. 461, pp. 3896-3909
    
Packages and versions:
----------------------
The versions listed below were used in the development of DREDGE-MOD, but the 
exact version numbers aren't specifically required. The installation process 
via PyPI will take care of installing or updating every library to at least the
level that fulfills the requirement of providing the necessary functionality.

Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
Scikit-learn 0.19.1
statsmodels 0.10.1
"""
# Load the necessary libraries
import sys
import numpy as np
from sklearn.neighbors import BallTree
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kernel_density import EstimatorSettings
from timeit import default_timer as timer
from numba import njit, prange
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import os
from numba.core.errors import NumbaPerformanceWarning
import warnings

# Suppress Numba performance warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


_last_calculated_bandwidth = None # keep count of the bandwidth 

def make_tree(coordinates):
    coordinates = np.radians(coordinates)
    tree = BallTree(coordinates, metric='haversine')
    return tree


def query_tree(tree, points, n_process, n_neighbors=100):
    """
    Query a KD Tree to find the nearest neighbors for a set of points, 
    optionally using parallel processing.

        Parameters:
    -----------
    tree : scipy.spatial.KDTree or similar
        The KD Tree object used for querying nearest neighbors.

    points : array-like
        An array of points (in degrees) for which the nearest neighbors 
        are to be found. Each point should have the same dimensionality 
        as the KD Tree. Shape (npoint, ndim)

    n_process : int
        The number of processes to use for parallel querying. If set to 1 or less, 
        no parallel processing is used.

    n_neighbors : int, optional
        The number of nearest neighbors to query for each point. Default is 100.

    Returns:
    --------
    indices : numpy.ndarray
        An array of indices of the nearest neighbors for each input point.

    distances : numpy.ndarray
        An array of distances (in degrees) to the nearest neighbors for 
        each input point.
    """

    x = np.radians(points)

    if n_process > 1:
        with multiprocessing.Pool(n_process) as pool:
            chunks = np.array_split(x, n_process)
            results = pool.map(partial(tree.query, k=n_neighbors, return_distance=True), chunks)
        distances, indices = zip(*results)
        distances = np.concatenate(distances)
        indices = np.concatenate(indices)
    else:
        distances, indices = tree.query(x, k=n_neighbors, return_distance=True)

    # Convert the distances from radians to degrees
    distances = np.degrees(distances)
    return indices, distances

    
def cut_points_with_tree(ridges, tree, bandwidth, threshold=4):
    """
    Filters out points in the `ridges` array whose nearest neighbor, as determined
    by the provided `tree`, is farther than a specified threshold distance.
    The threshold distance is calculated as `threshold` times the angular distance
    corresponding to the given `bandwidth`.
    Parameters:
    -----------
    ridges : numpy.ndarray
        An array of points (e.g., coordinates) to be filtered, in [dec, ra] in degrees
    tree : scipy.spatial.cKDTree
        A KD-tree object used to efficiently query the nearest neighbors of points in `ridges`.
    bandwidth : float
        The bandwidth value used to calculate the angular distance threshold.
    threshold : float, optional
        A multiplier for the bandwidth to determine the maximum allowable distance
        for a point to be considered valid. Default is 4.
    Returns:
    --------
    numpy.ndarray
        A filtered array of points from `ridges` that meet the distance criteria.

    Notes:
    ------
    - The function assumes that the input `ridges` are in degrees and converts them
      to radians for distance calculations.
    """
    distances, _ = tree.query(np.radians(ridges), k=1, return_distance=True)
    keep = distances[:, 0] < threshold * np.radians(bandwidth)
    return ridges[keep]

def estimate_bandsidth(coordinates, n_process):
    defaults = EstimatorSettings()
    defaults.n_jobs = n_process
    defaults.efficient = True

    # Generate the initial density estimate
    print("Building density estimator")
    sys.stdout.flush()
    density_estimate = KDEMultivariate(data=coordinates,
                                        var_type='cc',
                                        bw='cv_ml',
                                        defaults=defaults)
    return np.mean(density_estimate.bw)


def plot_state(coordinates, ridges, plot_dir, i):
    ra = coordinates[:, 1]
    dec = coordinates[:, 0]
    ridge_ra = ridges[:, 1]
    ridge_dec = ridges[:, 0]

    plt.figure()
    plt.plot(ra, dec, 'r,')
    plt.plot(ridge_ra, ridge_dec, 'k,')
    plt.savefig(f"{plot_dir}/ridges_{i}.png")
    plt.close()





def filaments(coordinates, 
              neighbors = 10, 
              bandwidth = None, 
              convergence = 0.005,
              initial_min_percentage = None,
              distance = 'haversine',
              n_process = 0,
              mesh_size = None,
              n_neighbors = 200,
              mesh_threshold = 4.0,
              final_threshold = 0.0,
              plot_dir = None,
              ):
    """Estimate density rigdges for a user-provided dataset of coordinates.
    
    This function uses an augmented version of the subspace-constrained mean
    shift algorithm to return density ridges for a set of user-provided
    coordinates. Apart from the haversine distance to compute a more accurate
    version of a common optimal kernel bandwidth calculation in criminology,
    the code also features thresholding to avoid ridges in sparsely populated
    areas. While only the coordinate set is a required input, the user can
    override the number of nearest neighbors used to calculate the bandwidth
    and the bandwidth itself, as well as the convergence threshold used to
    assess when to terminate and the percentage indicating which top-level of
    filament points in high-density regions should be returned. If the latter
    is not chose, all filament points are returned in the output instead.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    neighbors : int, defaults to 10
        The number of neighbors used for the optimal bandwidth calculation.
    
    bandwidth : float, defaults to None
        The bandwidth used for kernel density estimates of data points.
    
    convergence : float, defaults to 0.005
        The convergence threshold for the inter-iteration update difference.
    
    initial_min_percentage : float, defaults to None
        The percentage of initial mesh points discarded, starting with the lowest
        density points, before the iterations start. If None, no initial cut is made.
    
    distance: string, defaults to 'haversine'
        The distance function to be used, can be 'haversine' or 'euclidean'.
    
    n_process : int, defaults to 0
        The number of cores to be used for multiprocess parallelization (for bandwidth)
        and thread-level parallelization (for ridge updates).
    
    mesh_size : int, defaults to None
        The number of mesh points to be used to generate ridges.

    mesh_threshold: float, defaults to 4.0
        Throw away initial mesh point more than this many bandwidths from any coordinate

    final_threshold : float, defaults to 1.0
        Throw away final points more than this many bandwidths away
        from any point
        
    Returns:
    --------
    ridges : array-like
        The coordinates for the estimated density ridges of the data.
        
    Attributes:
    -----------
    None
    """
    # Check if the inputs are valid
    parameter_check(coordinates = coordinates,
                    neighbors = neighbors,
                    bandwidth = bandwidth,
                    convergence = convergence,
                    percentage = initial_min_percentage,
                    distance = distance,
                    n_process = n_process,
                    mesh_size = mesh_size,
                    )
    print("Input parameters valid!\n")
    print("Preparing for iterations ...\n")

    if final_threshold >= 1:
        raise ValueError("final_threshold must be between 0 and 1.")

    global _last_calculated_bandwidth  # Declare it as global within the function

    # Check whether no bandwidth is provided

    if bandwidth is None:
        print("Estimating bandwidth from data")
        bandwidth = estimate_bandsidth(coordinates, n_process)


    # Set a mesh size if none is provided by the user
    if mesh_size is None:
        mesh_size = int(np.min([1e5, np.max([5e4, len(coordinates)])]))

    # Create an evenly-spaced mesh in for the provided coordinates
    ridges = mesh_generation(coordinates, mesh_size)

    print("Generated mesh.  Making tree.")
    # Make the ball tree to speed up finding nearby points
    tree = make_tree(coordinates)


    # remove any ridges that are more than mesh_threshold bandwidths from any point
    print(f"Cutting initial mesh to points within {mesh_threshold} bandwidths of a galaxy")
    sys.stdout.flush()
    ridges = cut_points_with_tree(ridges, tree, bandwidth, threshold=mesh_threshold)
    print(f"Finished cutting. {ridges.shape[0]} mesh points remain.")

    # Record the initial density of all the points to allow us to do cuts later
    initial_density = tree.query_radius(np.radians(ridges), r=np.radians(bandwidth), return_distance=False, count_only=True)

    # Intitialize the update change as larger than the convergence
    update_average = np.inf
    # Loop over the number of prescripted iterations
    iteration_number = 0

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plot_state(coordinates, ridges, plot_dir, iteration_number)

    time_taken = 0
    while not update_average < convergence:

        # Update the points in the mesh. Record the timing
        t = timer()
        updates = ridge_update(ridges, coordinates, bandwidth, tree, n_process, n_neighbors)
        time_taken = timer() - t

        # Get the update size to check convergence.
        # JZ updated this from the previous version to make it independent
        # of the mesh size. This means the appropriate convergence values are much smaller
        update_average = np.mean(updates)

        iteration_number += 1
        print(f"Iteration {iteration_number}  update change: {update_average:.2e} target: {convergence:.2e} took {time_taken:.2f} seconds")

        if plot_dir is not None:
            plot_state(coordinates, ridges, plot_dir, iteration_number)
            np.save(f"{plot_dir}/ridges_{iteration_number}.npy", ridges)


    # # Check whether a top-percentage of points should be returned
    final_density = tree.query_radius(np.radians(ridges), r=np.radians(bandwidth), return_distance=False, count_only=True)

    # Return the iteratively updated mesh as the density ridges
    print("\nDone!")
    return ridges, initial_density, final_density


def get_last_bandwidth():
    global _last_calculated_bandwidth
    return _last_calculated_bandwidth


def ridge_update(ridges, coordinates, bandwidth, tree, n_process, n_neighbors):
    all_nearby_indices, all_distances = query_tree(tree, ridges, n_process, n_neighbors)
    updates = ridge_update_inner(ridges, coordinates, bandwidth, all_nearby_indices, all_distances)
    return updates


@njit
def ridge_update_inner(ridges, coordinates, bandwidth, all_nearby_indices, all_distances):
    # Create a list to store all update values
    updates = np.zeros(ridges.shape[0])
    for i in prange(ridges.shape[0]):
        # Compute the update movements for each point
        ridge = ridges[i]
        # get all the points within the 3 sigma bandwidth
        nearby_indices = all_nearby_indices[i]
        distance = all_distances[i]
        nearby_coordinates = coordinates[nearby_indices].copy()
    
        point_updates = update_function(ridge, nearby_coordinates, bandwidth, distance)
        # Add the update movement to the respective point
        ridges[i] = ridge + point_updates
        # Store the change between updates to check convergence
        updates[i] = np.sum(np.abs(point_updates))
    return updates



def update_ridge(ridge, 
                 coordinates, 
                 bandwidth):
    """
    Update the ridge points from the mesh for the current iteration.
    
    The initial mesh of points from which the ridges are formed over the
    course of multiple iterations have to be updated separately, in each
    iteration of the SCMS algorithm. This function calculates the point
    updates, shifts the ridge points and saves the implemented change for
    the convergence check performed at the end of each iteration.
    
    Parameters:
    -----------
    ridge : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
        
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    ridge : array-like
        The updated ridge point shifted toward a new position.
        
    update_change : array-like
        The overall change in order to later check for convergence.
    """
    # Compute the update movements for each point
    point_updates = update_function(ridge, coordinates, bandwidth)
    # Add the update movement to the respective point
    ridge = ridge + point_updates
    # Store the change between updates to check convergence
    update_change = np.abs(np.mean(np.sum(point_updates)))
    # Return updated ridges and the stored change
    return ridge, update_change

def mesh_generation(coordinates, 
                    mesh_size):
    """Generate a set of uniformly-random distributed points as a mesh.
    
    The subspace-constrained mean shift algorithm operates on either a grid
    or a uniform-random set of coordinates to iteratively shift them towards
    the estimated density ridges. Due to the functionality of the code, the
    second approach is chosen, with a uniformly-random set of coordinates
    in the intervals covered by the provided dataset as a mesh. In order to
    not operate on a too-small or too-large number of mesh points, the size
    of the mesh is constrained to a lower limit of 50,000 and an upper limit
    of 100,000, with the size of the provided dataset being used if it falls
    within these limits. This is done to avoid overly long running times.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    mesh_size : int
        The number of mesh points to be used to generate ridges.
        
    Returns:
    --------
    mesh : array-like
        The set of uniform-random coordinates in the dataset's intervals.
        
    Attributes:
    -----------
    None
    """
    # Get the minimum and maximum for the latitudes
    min_latitude = np.min(coordinates[:, 0])
    max_latitude = np.max(coordinates[:, 0])
    # Get the minimum and maximum for the longitudes
    min_longitude = np.min(coordinates[:, 1])
    max_longitude = np.max(coordinates[:, 1])
    # Get the number of provided coordinates
    #size = int(np.min([1e5, np.max([5e4, len(coordinates)])]))
    # Create an array of uniform-random points as a mesh
    mesh_1 = np.random.uniform(min_latitude, max_latitude, mesh_size)
    mesh_2 = np.random.uniform(min_longitude, max_longitude, mesh_size)
    mesh = np.vstack((mesh_1.flatten(), mesh_2.flatten())).T
    # Return the evenly-spaced mesh for the coordinates
    return mesh

def threshold_function(mesh, density_estimate, threshold, n_process):
    """Calculate the cut-off threshold for mesh point deletions.
    
    This function calculates the threshold that is used to deleted mesh
    points from the initial uniformly-random set of mesh points. The
    rationale behind this approach is to avoid filaments in sparsely
    populated regions of the provided dataset, leading to a final result
    that only covers filaments in regions of a suitably high density.
    
    Parameters:
    -----------
    mesh : array-like
        The set of uniform-random coordinates in the dataset's intervals.
        
    density_estimate : scikit-learn object
        The kernel density estimator fitted on the provided dataset.

    threshold : float
        Keep only points with density below this fraction of the mean

    n_process : int
        Number of parallel processes
        
    Returns:
    --------
    mesh: the new mesh points.

    cut: the minimum density kept
        
    Attributes:
    -----------
    None
    """
    # Run KDE to get an estimate of the coordinate point density
    # at each mesh point. This can be slow so we do it in parallel.
    if n_process > 1:
        with multiprocessing.Pool(n_process) as pool:
            chunks = np.array_split(mesh, n_process)
            density_chunks = pool.map(density_estimate.pdf, chunks)
        density = np.concatenate(density_chunks)
    else:
        density = density_estimate.pdf(mesh)

    # Calculate the average of density estimates for the data
    mean_density = density.mean()
    cut = mean_density * threshold
    keep = density > cut
    return mesh[keep], cut

@njit
def update_function(point, 
                    coordinates, 
                    bandwidth,
                    distance):
    """Calculate the mean shift update for a provided mesh point.
    
    This function calculates the mean shift update for a given point of 
    the mesh at the current iteration. This is done through a spectral
    decomposition of the local inverse covariance matrix, shifting the
    respective point closer towards the nearest estimated ridge. The
    updates are provided as a tuple in the latitude-longitude space to
    be added to the point's coordinate values.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
        
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
        
    Returns:
    --------
    point_updates : float
        The tuple of latitude and longitude updates for the mesh point.
        
    Attributes:
    -----------
    None
    """
    squared_distance = distance ** 2
    # evaluate the kernel at each distance
    weights = gaussian_kernel(squared_distance, bandwidth)
    # now reweight each point
    shift = coordinates.T @ weights / np.sum(weights)
    # first, we evaluate the mean shift update
    update = shift - point
    # Calculate the local inverse covariance for the decomposition
    inverse_covariance = local_inv_cov(point, coordinates, bandwidth)
    # Compute the eigendecomposition of the local inverse covariance
    eigen_values, eigen_vectors = np.linalg.eig(inverse_covariance)
    # Align the eigenvectors with the sorted eigenvalues
    sorted_eigen_values = np.argsort(eigen_values)
    eigen_vectors = eigen_vectors[:, sorted_eigen_values]
    # Cut the eigenvectors according to the sorted eigenvalues
    cut_eigen_vectors = eigen_vectors[:, 1:]
    # Project the update to the eigenvector-spanned orthogonal subspace
    point_updates = cut_eigen_vectors.dot(cut_eigen_vectors.T).dot(update)    
    # Return the projections as the point updates
    return point_updates

@njit
def gaussian_kernel(values, 
                    bandwidth):
    """Calculate the Gaussian kernel evaluation of distance values.
    
    This function evaluates a Gaussian kernel for the squared distances
    between a mesh point and the dataset, and for a given bandwidth.
    
    Parameters:
    -----------
    values : array-like
        The squared distances between a mesh point and provided coordinates.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    kernel_value : float
        The Gaussian kernel evaluations for the given distances.
        
    Attributes:
    -----------
    None
    """
    # Compute the kernel value for the given values
    kernel_value = np.exp(-0.5 * values / bandwidth**2)
    # Return the computed kernel value
    return kernel_value

@njit
def mean1(a):
    """Calculate the mean of a 2D array along axis 1"""
    n1, n2 = a.shape
    res = np.zeros(n1)
    for i in range(n1):
        res[i] = np.sum(a[i, :]) / n2
    return res

@njit
def local_inv_cov(point, 
                  coordinates, 
                  bandwidth):
    """Compute the local inverse covariance from the gradient and Hessian.
    
    This function computes the local inverse covariance matrix for a given
    mesh point and the provided dataset, using a given bandwidth. In order
    to reach this result, the covariance matrix for the distances between
    a mesh point and the dataset is calculated. After that, the Hessian
    matrix is used to calculate the gradient at the given point's location.
    Finally, the latter is used to arrive at the local inverse covariance.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
    
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    inverse_covariance : array-like
        The local inverse covariance for the given point and coordinates.
        
    Attributes:
    -----------
    None
    """


    number_points, number_columns = coordinates.shape 

    # Calculate the squared distance between points
    squared_distance = np.sum((coordinates - point)**2, axis=1)
    # Compute the weight kernels called b_j in the paper
    weights = gaussian_kernel(squared_distance, bandwidth)
    weight_sum = np.sum(weights)
    weight_average = weight_sum / number_points

    # Compute the location differences between the point and the dataset
    mu = (coordinates - point) / bandwidth**2

    # Combine terms to get the Hessian matrix following the paper algorithm
    term1 = (weights * mu.T) @ mu / number_points
    term2 = weight_sum * np.eye(number_columns) / bandwidth**2 / number_points
    H = term1 - term2

    # This is an extra term that is not in the paper
    grad = -mean1(weights * mu.T)
    inv_cov = -H / weight_average + (grad @ grad) / weight_average**2
    return inv_cov


def parameter_check(coordinates, 
                    neighbors, 
                    bandwidth, 
                    convergence, 
                    percentage,
                    distance,
                    n_process,
                    mesh_size):
    """Check the main function inputs for unsuitable formats or values.
    
    This function checks all of the user-provided main function inputs for 
    their suitability to be used by the code. This is done right at the
    top of the main function to catch input errors early and before any
    time is spent on time-consuming computations. Each faulty input is
    identified, and a customized error message is printed for the user
    to inform about the correct inputs before the code is terminated.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    neighbors : int
        The number of neighbors used for the optimal bandwidth calculation.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
    
    convergence : float
        The convergence threshold for the inter-iteration update difference.
    
    percentage : float
        The percentage of highest-density filament points that are returned.
    
    distance: string, defaults to 'haversine'
        The distance function to be used, can be 'haversine' or 'euclidean'.
    
    n_process : int
        The number of cores to be used for multiprocess parallelization.
        
    Returns:
    --------
    None
        
    Attributes:
    -----------
    None
    """
    # Create a boolean vector to keep track of incorrect inputs
    incorrect_inputs = np.zeros(7, dtype = bool)
    # Check whether two-dimensional coordinates are provided
    if not type(coordinates) == np.ndarray:
        incorrect_inputs[0] = True
    elif not coordinates.shape[1] == 2:
        incorrect_inputs[0] = True
    # Check whether neighbors is a positive integer or float
    if not ((type(neighbors) == int and neighbors > 0)
        and not ((type(neighbors) == float) 
                 and (neighbors > 0)
                 and (neighbors.is_integer() == True))):
        incorrect_inputs[1] = True
    # Check whether bandwidth is a positive integer or float
    if not bandwidth == None:
        if not ((type(bandwidth) == int and bandwidth > 0)
            or (type(bandwidth) == float) and bandwidth > 0):
            incorrect_inputs[2] = True
    # Check whether convergence is a positive integer or float
    if not convergence == None:
        if not ((type(convergence) == int and convergence > 0)
            or (type(convergence) == float) and convergence > 0):
            incorrect_inputs[3] = True
    # Check whether percentage is a valid percentage value
    if not percentage == None:
        if not ((type(percentage) == int and percentage >= 0 
                 and percentage <= 100)
                or ((type(percentage) == float) and percentage >= 0 
                    and percentage <= 100)):
            incorrect_inputs[4] = True
    # Check whether distance is one of two allowed strings
    if not type(distance) == str:
        incorrect_inputs[5] = True
    elif not ((distance == 'haversine') or (distance == 'euclidean')):
        incorrect_inputs[5] = True
    # Check whether n_process is an applicable integer
    if not type(n_process) == int:
        incorrect_inputs[6] = True
    elif not (n_process >= 0):
        incorrect_inputs[6] = True
    # Define error messages for each parameter failing the tests
    errors = ['ERROR: coordinates: Must be a 2-column numpy.ndarray',
              'ERROR: neighbors: Must be a whole-number int or float > 0',
              'ERROR: bandwidth: Must be an int or float > 0, or None',
              'ERROR: convergence: Must be an int or float > 0, or None',
              'ERROR: percentage: Must be an int or float in [0, 100], or None',
              'ERROR: distance: Must be either "haversine" or "euclidean"',
              'ERROR: n_process: Must be an integer >= 0 or None']
    # Print eventual error messages and terminate the code
    if any(value == True for value in incorrect_inputs):
        for i in range(0, len(errors)):
            if incorrect_inputs[i] == True:
                print(errors[i])
        sys.exit()

