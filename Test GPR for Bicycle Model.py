# from matplotlib import pyplot as plt 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
import math
import csv
np.random.seed(6)               # Ensures same random var. values used each execution of the code


# arbitrary values
x_0 = 0
y_0 = 0
theta_0 = np.deg2rad(10)  # degrees
delta_0 = np.deg2rad(15)  # degrees
L = 10
dt = 1.0
v_0 = 300.0
psi_0 = 5.0

class KBM:
    def __init__(self, v, psi):
        # Define changing rate for each output of model using given input values
        self.v = v
        self.psi = psi
        self.x_dot = self.v * math.cos(delta_0 + theta_0)
        self.y_dot = self.v * math.sin(delta_0 + theta_0)
        self.theta_dot = self.v * math.sin(delta_0) / L
        self.delta_dot = self.psi
        
    def getnextstate(self, t):
        # Update output values using a discrete time model
        update_vector = np.zeros(4)
        update_vector[0] = x_0 + self.x_dot * t
        update_vector[1] = y_0 + self.y_dot * t
        update_vector[2] = theta_0 + self.theta_dot * t
        update_vector[3] = delta_0 + self.delta_dot * t
        return update_vector
    
    
class DataSet:
    def __init__(self, size_i, size_j, step_i, step_j):
        # Instantiate model and 3d array
        self.outputs = KBM(v_0, psi_0)
        self.nextstate = [x_0, y_0, theta_0, delta_0]
        self.size_i = size_i
        self.size_j = size_j
        self.step_i = step_i
        self.step_j = step_j
        self.set = np.zeros([4, int(self.size_i), int(self.size_j)])
        
        
    def update(self, v, psi):
        # Update to next position
        self.outputs = KBM(v, psi)
        self.nextstate = self.outputs.getnextstate(dt)
        
    def buildset(self):
        # Fill dataset values
        for i in range(0, self.size_i, 1):
            for j in range(0, self.size_j, 1):
                self.update((v_0+(i*self.step_i)), (psi_0+(np.deg2rad(j*self.step_j))))
                self.set[0, i, j] = self.nextstate[0]
                self.set[1, i, j] = self.nextstate[1]
                self.set[2, i, j] = self.nextstate[2]
                self.set[3, i, j] = self.nextstate[3]
"""
Inputs:
    - X_1 = velocity vector
    - X_2 = psi vector
"""           
class GPR():
    
    def __init__(self, kernel, optimizer='L-BFGS-B', noise_var=1e-8):
        self.kernel = kernel
        self.noise_var = noise_var
        self.optimizer = optimizer
        
    def _cholesky_factorise(self, y_cov):
        try:
            L = np.linalg.cholesky(y_cov)
        except np.linalg.LinAlgError as e:
            e.args = ("The kernel, %s, is not returning a" 
                      "positive definite matrix. Try increasing"
                      " the noise variance of the GP or using"
                      " a larger value for epsilon. "
                      % self.kernel,) + e.args
            raise
        return L
    
    def _sample_multivariate_gaussian(self, y_mean, y_cov, y_data, n_samples=1, epsilon=1e-10):
        y_cov[np.diag_indices_from(y_cov)] += epsilon  # for numerical stability
        L = self._cholesky_factorise(y_cov)
        print('y_mean shape = ' + str(y_mean.shape))
        print('y_data shape = ' + str(y_data.shape))
        # print(L*y_data[0, 0])
        # u = np.random.randn(y_mean.shape[0], n_samples)
        if(len(y_mean.shape) <= 2):
            z = np.dot(L, y_data) + y_mean[:, np.newaxis]
        else: # if greater than 2D mean vector
            z = np.dot(L, y_data) + y_mean
        # print('new shape of z = ' + str(z.shape))
        return z
    
    # ASSUMES X1.shape IS SAME AS X2.shape!!!
    def sample_prior(self, X1, y_data, X2 = None, n_samples=1):
        y_mean = np.zeros(X1.shape[0])                      # Get mean vector; Assume the means are all zeroes to start
        y_cov = self.kernel(X1, X2)                             # Get covariance matrix
        return self._sample_multivariate_gaussian(y_mean, y_cov, y_data, n_samples)
    
    def sample_posterior(self, X_train, y_train, X_test, y_data, n_samples=1):
        # Compute alpha (shorthand for portion of long equation)
        K = self.kernel(X_train)
        K[np.diag_indices_from(K)] += self.noise_var
        L = self._cholesky_factorise(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        print('alpha shape = ' + str(alpha.shape))
        
        # Compute posterior mean
        K_trans = self.kernel(X_test, X_train)  # K but with noise
        print('transpose of Ks shape = ' + str(K_trans.shape))
        y_mean = K_trans.dot(alpha)                                 # THINK ISSUE MIGHT BE HERE?
        
        # Compute posterior covariance
        v = np.linalg.solve(L, K_trans.T)       # L.T * K_inv * K_trans.T
        y_cov = self.kernel(X_test) - np.dot(v.T, v)
        print('y_cov shape = ' + str(y_cov.shape))
        
        return self._sample_multivariate_gaussian(y_mean, y_cov, y_data), y_mean, y_cov
    
    """
    def sample_posterior(self, X1_post, X2_post, y_train, X1_pre, X2_pre, n_samples=1):
        # compute alpha
        K = self.kernel(X_train)
        K[np.diag_indices_from(K)] += self.noise_var
        L = self._cholesky_factorise(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # Compute posterior mean
        K_trans = self.kernel(X_test, X_train)
        y_mean = K_trans.dot(alpha)
       
        # Compute posterior covariance
        v = np.linalg.solve(L, K_trans.T)  # L.T * K_inv * K_trans.T
        y_cov = self.kernel(X_test) - np.dot(v.T, v)
        
        return self._sample_multivariate_gaussian(y_mean, y_cov, n_samples), y_mean, y_cov
    """
    
# Changed self.theta to self.length from Mike O Neill's code
class SquaredExponential():
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = [length_scale]
        self.bounds = [length_scale_bounds]
    def __call__(self, X1, X2=None):
        if X2 is None:
            # K(X1, X1) is symmetric so avoid redundant computation using pdist.
            dists = pdist(X1 / self.length_scale[0], metric='sqeuclidean')
            K = np.exp(-0.5 * dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X1 / self.length_scale[0], X2 / self.length_scale[0], metric='sqeuclidean')
            K = np.exp(-0.5 * dists)
        # print('K = ' + str(K.shape))
        return K
    
class Periodic():
    def __init__(self, frequency=1.0, frequency_bounds=(1e-5, 1e5)):
        self.theta = [frequency]
        self.bounds = [frequency_bounds]
    def __call__(self, X1, X2=None):
        if X2 is None:
            # K(X1, X1) is symmetric so avoid redundant computation using pdist.
            dists = pdist(X1, lambda xi, xj: np.dot(np.sin(self.theta[0] * np.pi * (xi - xj)).T, 
                np.sin(self.theta[0] * np.pi * (xi - xj))))
            K = np.exp(-dists)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            dists = cdist(X1, X2, lambda xi, xj: np.dot(np.sin(self.theta[0] * np.pi * (xi - xj)).T, 
                np.sin(self.theta[0] * np.pi * (xi - xj))))
            K = np.exp(-dists)
        return K


# Main:

"""
# Get v, psi data
with open(".csv", 'r', newline = '') as inputDataFile:
    reader = csv.reader(inputDataFile, delimiter = ';')
    # fields = ['x_m', 'y_m', ]

    # Skip non-data containing rows in file
    next(reader)
    
    # Define arrays to store global X and Y data for optimal raceline
    x_data = []
    y_data = []
    
    for row in reader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
    
    # Get highest index of each data array (to be used later)
    # (Assuming x- and y- data sets will be of same length)
    last_point_in_path = len(x_data) - 1
    
    # Draw path
    drawPath(x_data, y_data)
"""

# np.random.randn normally generates random #s from [-1, 1]
# (np.random.randn(N) * range_size) + min_value
# range_size = max - min_value
# v = np.sort((np.random.randn(3) * 10) + 0)[:, np.newaxis]
# psi = np.sort((np.random.rand(3) * 10) + -5)[:, np.newaxis]
# v = np.sort(np.random.randn(3))[:, np.newaxis]                      # MUST BE SORTED OTHERIWSE BREAKS!!!
# psi = np.sort(np.random.randn(3))[:, np.newaxis]
    
load_set = np.loadtxt("10x10.csv", dtype=float, delimiter=',')
dataset = load_set.reshape(load_set.shape[0], load_set.shape[1]//4, 4)  
# print(dataset.shape)

low_v = 300
high_v = 309
low_psi = 0
high_psi = 9
v = np.arange(low_v, (high_v + 1), 1)
psi = np.arange(low_psi,  (high_psi + 1), 1)
inputData = np.concatenate((v[:, np.newaxis], psi[:, np.newaxis]), axis = 1)
# print('X1 = \n' + str(inputData) + '\n')


"""
MAKES A TRAINING SET OF RANDOM FLOATS SELECTED FROM THE v AND psi RANGES
# Generate n random numbs from 300-310 (velocity)
trainingInputsCol_1 = np.random.uniform(size = 10, low=low_v, high=high_v)[:, np.newaxis]
# Generate n random numbs from 0-10 (psi)
trainingInputsCol_2 = np.random.uniform(size = 10, low=low_psi, high=high_psi)[:, np.newaxis]
trainingData = np.concatenate((trainingInputsCol_1, trainingInputsCol_2), axis = 1)
"""

numberOfStates = 4   # x, y, theta, delta
trainingSetRows = 1
trainingInputsCol_1 = np.zeros(trainingSetRows)[:, np.newaxis]
trainingInputsCol_2 = np.zeros(trainingSetRows)[:, np.newaxis]
datasetSplice = np.zeros([trainingSetRows, trainingSetRows, numberOfStates])
# Generate n random numbs from 300-310 (velocity)
indexSet1 = np.random.choice((high_v-low_v + 1), size = trainingSetRows, replace=False)
indexSet2 = np.random.choice((high_psi-low_psi + 1), size = trainingSetRows, replace=False)
# trainingInputsCol_1 = np.random.choice(v, size = 4, replace = False)[:, np.newaxis]
# Generate n random numbs from 0-10 (psi)
# trainingInputsCol_2 = np.random.choice(psi, size = 4, replace = False)[:, np.newaxis]

for i in range(trainingSetRows):
    trainingInputsCol_1[i] += v[indexSet1[i]]
    trainingInputsCol_2[i] += psi[indexSet2[i]]
    # datasetSplice[i, 0] += dataset[indexSet1[i], indexSet2[i]]
    for j in range(trainingSetRows):
        datasetSplice[i, j] += dataset[indexSet1[i], indexSet2[j]]
    # print(dataset[indexSet1[i], indexSet2[i]])

# print(trainingInputsCol_1)
# print(trainingInputsCol_2)
trainingData = np.concatenate((trainingInputsCol_1, trainingInputsCol_2), axis = 1)
# print(trainingData)
# print(datasetSplice.shape) # (4,4,4)
# print('print 3 = ' + str(len(datasetSplice.shape)))
# print('X2 = \n' + str(trainingData))

# print(inputData[:, 0][:, np.newaxis])
gp = GPR(SquaredExponential(1))
y_samples = gp.sample_prior(inputData, dataset)

y_train = gp.sample_prior(trainingData, datasetSplice) # (4,4,4)

# Do we use dataset of datasetSplice for our y_data here???
# f_star_samples, f_star_mean, f_star_covar = gp.sample_posterior(X_train, y_train, X_test, n_samples=10)
f_star_samples, f_star_mean, f_star_covar = gp.sample_posterior(trainingData, y_train, inputData, dataset)



""" 
kernel = SquaredExponential()
K = kernel(v, psi)
eigvals, eigvects = np.linalg.eig(K)
# print('v = ' + str(v))
# print('psi = ' + str(psi))
# print('K = ' + str(K))
# print(eigvals)
L = np.linalg.cholesky(K)
# print(L)

gp = GPR(SquaredExponential(1))
y_samples = gp.sample_prior(v, psi, n_samples=3)
print("y_samples = " + str(y_samples))
# plt.plot(X_test, y_samples)
# plt.title('{} kernel'.format(name))
plt.show()
# print(K.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nx, ny = y_samples.shape
y = np.linspace(0, ny-1, ny)
x = np.linspace(0, nx-1, nx)
xv, yv = np.meshgrid(x, y)
dem3d = ax.plot_surface(xv, yv, y_samples)


test = DataSet(5, 5, 10, 1)
test.buildset()

# First define input test points at which to evaluate the sampled functions
X_test = test.set


Current issue: X_test is greater than a 2D matrix;
if can fix, than the code should produce a posterior mean array
aka our desired output.

Two ways of doing this:
    1. The dimensionality issue is raised at SquaredExponential's
    cdist. If we could find an alternative for finding the Euclidean distance
    |u-v|^2 then we could get the code to go as is without manipulating
    X_test.
    
    
    2. We could broadcast various sections of X_test's data into
    a bunch of 2D arrays and send them through the code.
    Pros: I think I could figure out how to do this.
    
    Cons: Will drive up the computational expense extremely as the
    only way I can think to do so is via loops :/


"""
