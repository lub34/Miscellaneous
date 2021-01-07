import scipy.linalg
import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import time

fig = plt.figure()

# Values of various properties for the bicycle model discussed in section 2 of Rajesh Rajamani's textbook.
class BicycleModel:
    # Can't set Cf or Cr to 2000 -- which is in the ball park of their actual value in the sim -- produces nan values in K
    # Define constants (All are temporary values for now, FILL IN PROPERLY LATER!!!)
    Cf = 200                               # Cornering stiffness of front tire(s) 2000 [N/deg] = 114591.6[N/rad]
    Cr = 200                               # Cornering stiffness of rear tire(s) [N/rad]
    m = 761                                # Total mass of vehicle [kg] (CHECK: Add wheel masses and fuel mass?)
    lf = 1.54                              # Length btwn vehicle's c.g. and front axle c.g. [m]
    lr = 2.554                             # Length btwn vehicle's c.g. and rear axle c.g. [m]
    Vx = 90                                # Vehicle's longitudinal velocity [m/s]
    Iz = 550                               # Vehcile's yaw moment of inertia (From ANSYS Simulation)

# Updates the vehicle's states.
def get_bicycle_func(u, model):
    def bicycle_func(x):
        Cf = model.Cf
        Cr = model.Cr
        m = model.m
        lf = model.lf
        lr = model.lr
        Vx = model.Vx
        Iz = model.Iz  
    
        y_dot = x[1]
        y_dot_dot = (-(2*Cf + 2*Cr) / (m*Vx)) * x[1] - Vx + \
            ((-2*Cf*lf - 2*Cr*lr) / (m*Vx)) * x[3] + ((2*Cf) / m) * u
        yaw_dot = x[3]
        yaw_dot_dot = ((-2*Cf*lf - 2*Cr*lr) / (Iz*Vx)) * x[1] + \
            ((-2*Cf*(lf**2) + 2*Cr*(lr**2)) / (Iz*Vx)) * x[3] + ((2*Cf*lf) / Iz) * u
        
        dx = np.array([y_dot,
                       y_dot_dot,
                       yaw_dot,
                       yaw_dot_dot])
        
        return dx
    return bicycle_func

# Performs LQR on a state space model.
# Returns the LQR gain K, the resulting matrix from solving the
# Ricatti equation, and the eigenvalues of the updated A matrix
# often referred to as A_cl = A_closed_loop = A - B*K
def lqr(A,B,Q,R):
    # Solve the Ricatti equation using our state space equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    
    # cost = integral x.T*Q*x + u.T*R*u
    # compute the LQR gain K
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K

# Perform Runge-Kutta approximation to estimate the updated state values
def step_continuous_func(dynamics_func, P_i, dt, use_runge_kutta = True):
    if use_runge_kutta:
        # Runge-Kutta Magix
        dp1 = dynamics_func(P_i)
        dp2 = dynamics_func(P_i + 0.5 * dt * dp1)
        dp3 = dynamics_func(P_i + 0.5 * dt * dp2)
        dp4 = dynamics_func(P_i + dt * dp3)
        dpf = (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4) * dt / 6.0
        return P_i + dpf, np.linalg.norm(dpf)
    else:
        dpf = dynamics_func(P_i) * dt
        return P_i + dpf, np.linalg.norm(dpf)
    
# ASK JOSH ABOUT d_t = discretization!!!
"Originally had default value for dt: d_t = discretization"
def finite_horizon_lqr(A, B, Q, R, Q_f, horizon, d_t ):
    """
    Solves a Continuous Ricatti Equation backwards in time.
    A, B, Q, R, Q_f: dynamics / cost matrices
    horizon: Time horizon in seconds
    d_t: time discretization for steps solving backwards.
    """
    # returns the discretaization period dp? CHECK
    def compute_dp(P_t):
        return -(A.T @ P_t + P_t @ A - P_t @ B @ np.linalg.inv(R) @ B.T @ P_t + Q)
    
    P_n = Q_f                       # CHECK ???
    time = horizon                  # How far out in time we want to plan for
    norm_dp = 0.0000001             # CHECK ???
    Ps = [P_n]                      # CHECK ???
    
    while math.fabs(time) > 0.00000000001:
        if time < d_t:
            d_t = time
        time -= d_t
        P_n, norm_dp = step_continuous_func(compute_dp, P_n, -d_t)
        Ps.append(P_n)
        # print("Solving P_{%3.3f}, norm: %3.9f"%(
        #     time,
        #     norm_dp
        # ))
    return np.linalg.inv(R) @ B.T @ P_n

def drawPath(X_data, Y_data):
    # Clear figure
    plt.clf()
    plt.scatter(X_data, Y_data, s = 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sample Path to Test Bicycle Controller On')
    """
    left, right = plt.xlim()  # return the current xlim
    bottom, top = plt.ylim()  # return the current ylim
    plt.xlim(left + 0.25*left, right + 0.25*right)    
    plt.ylim(bottom + 0.25*bottom, top + 0.25*top)    
    """
    return None

"""
Draws new position of the vehicle in 2D space as a red line.
Parameters:
    - state = 1D array representing current state vector x from state-space model.
              Assumes third element of 'state' is vehicle's yaw angle (index 2).
    - x_g = Global x-position of the vehicle's center of gravity
    - y_g = Global y-position of the vehicle's center of gravity
"""
def drawBicycle(yaw, X_g, Y_g, block = True, dt = 0.1):
    
    # dx_f = lf*sin(yaw); dx_f = change in x between vehicle's global CG and its front axle CG location.

    # (dx_f, dy_f) = position of front axle's CG relative to bicycle's CG.
    # (dx_r, dy_r) = position of rear axle's CG relative to bicycle's CG.
    # a is the distance from the CG to the front axle CG (not to scale -- just so visible on graph).
    # b is the distance from the CG to the rear axle CG (not to scale -- just so visible on graph).
    a = 30
    b = 50
    
    # CHANGED: As of 11/18/2020, swapped the sin and cos calls here b/c of figure 2-12 in Rajesh (do for following block too)
    dx_f = a*math.cos(yaw)
    dy_f = a*math.sin(yaw)
    
    # Potential bug: Yaw being defined from [pi,-pi] could screw up math here, messing vehicle's orientation
    dx_r = b*math.cos(math.pi + yaw)
    dy_r = b*math.sin(math.pi + yaw)
    
    # Solve for the global locations of each axle's CG
    x_f = X_g + dx_f
    y_f = Y_g + dy_f
    
    x_r = X_g + dx_r
    y_r = Y_g + dy_r
    
    # Finally, plot the bicycle's initial position and orientation:
    # x_axle_values = [x_r, x_f]
    # y_axle_values = [y_r, y_f]
    x_cg_to_frontAxle = [X_g, x_f]
    y_cg_to_frontAxle = [Y_g, y_f]
    x_cg_to_rearAxle = [X_g, x_r]
    y_cg_to_rearAxle = [Y_g, y_r]
    # plt.plot(x_axle_values, y_axle_values, 'r')
    plt.plot(x_cg_to_frontAxle, y_cg_to_frontAxle, 'k', linewidth = 5)         # Line from vehicle's CG to front axle is black
    plt.plot(x_cg_to_rearAxle, y_cg_to_rearAxle, 'r',  linewidth = 5)           # Line from vehicle's CG to rear axle is red
    plt.show(block = block)
    if not block:
        plt.pause(dt)
    return None

#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
# Code for the Controller (Simulation):

# Define variables for storing highest index of x- and y- data arrays in given path data:
# This index references the last (x,y) point in the data set.
last_point_in_path = 0

# Get optimal raceline path data
with open("optimalPathData.csv", 'r', newline = '') as optimalPathDataFile:
    reader = csv.reader(optimalPathDataFile, delimiter = ';')
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
    
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# Main:

# Get vehicle data based off vehicle model:
vehicleData = BicycleModel()

# Initialize constants and state space info  based off vehicle model
Cf = vehicleData.Cf
Cr = vehicleData.Cr
m = vehicleData.m
lf = vehicleData.lf
lr = vehicleData.lr
Vx = vehicleData.Vx
Iz = vehicleData.Iz  

# Can I add whitespace to the matrix definitions to make them prettier on the eyes?
"""
The states for the following are as follows:
- x1 = (e1) rate of change of the distance of the vehicle's CG from the desired path's center line
- x2 = (e1_dot) x1_dot
- x3 = (e2) error of the orientation error of the vehicle wrt to the road (yaw - y_goal)
- x4 = (e2_dot) x3_dot
"""
A = np.array([[0, 1,                                0,                          0],
              [0, (-(2*Cf + 2*Cr) / (m*Vx)),        ((2*Cf + 2*Cr) / m),        ((2*Cf*lf - 2*Cr*lr) / (m*Vx)) ],
              [0, 0,                                0,                          1],
              [0, (-(2*Cf*lf - 2*Cr*lr) / (Iz*Vx)), ((2*Cf*lf - 2*Cr*lr) / Iz), -((2*Cf*(lf**2) + 2*Cr*(lr**2)) / (Iz*Vx))]
              ])

B = np.array([[0,
              ((2*Cf) / m),
              0,
              ((2*Cf*lf) / Iz)]]).T

# Define Q and R matrices (guessing at values for now, tweak once running as 
# one usually does with Q and R matrices)
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Penalizes inputs (CHECK: What are our inputs???)
R = np.array([[1]])

# Define initial (x,y) point of bicycle's CG -- pt. G (will be updated in loop)
x_g = 50
y_g = 0.0005

# Initialize the state vector [e1, e1_dot, e2, e2_dot]:
x = np.array([[80,
               1,
               -0.0146114,
               -0.1]], dtype = float).T


# Initialize index of desired current and next point along path:
currentPt = 0
nextPt = currentPt + 1

# Get desired yaw angle by calculating slope between first two points of
# desired path (used to measure orientation error)
goal_yaw = math.atan((y_data[nextPt] - y_data[currentPt]) / (x_data[nextPt] - x_data[currentPt]))
        
yaw = x[2] + goal_yaw                   # yaw = e2 + desired yaw

# Draw the position and orientation of the bicycle in our plot (bicycle appears as red line):
drawBicycle(yaw, x_g, y_g, block = False)

# Use LQR to guide vehicle to satisfactory position and orientation.
dt = 0.1
# K = finite_horizon_lqr(A, B, Q, R, Q, 20.0, dt)
# print(K)
K = lqr(A, B, Q, R)
tracker = 0                                  # Escape flag for while loop. Turns on when path, assumed loop, returns to start
t_start = time.time()
old_yaw = 0

# Below you will find two input arrays. The first has roughly 10 u values that repeat and the other is a single u value repeating over and over. Comment out one and run the other to see the vehicle move.
# inputs = [0.7855, 0, 50, -0.6, -20, 0.569, 0.9033, -5, -7, 8, 7.664, 0.664, 0.993, -2, 0.7855, 0, 50, -0.6, -20, 0.569, 0.9033, -5, -7, 8, 7.664, 0.664, 0.993, -2, 0.7855, 0, 50, -0.6, -20, 0.569, 0.9033, -5, -7, 8, 7.664, 0.664, 0.993, -2]
inputs = [0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855, 0.7855]
for i in range(len(inputs)):
    u = inputs[i]
        
    # Transform state variable into terms of lateral displacement (y) and yaw
    y = x[0]
    y_dot = x[1] - Vx * x[2]                # (I think we should have + Vx * x[2] instead b/c delta = new - old = yaw_des - yaw and Vx should be mutliplied by [yaw - yaw_des])
    yaw = x[2] + goal_yaw                   # Pretty sure this is extraneous. Gonna keep to be safe. 
    yaw_dot = x[3]
    
    print("y = " + str(y))
    print("yaw = " + str(yaw))
    print("e1 = " + str(x[0]))
    print("e2_dot = " + str(x[3]))
    
    x_in_y_and_yaw_terms = np.array([y,
                                   y_dot,
                                   yaw,
                                   yaw_dot], dtype = float)
    
    # print("Shape of modified state vector: " + str(x_in_y_and_yaw_terms.shape))
    
    # Get change in state variables 
    dx = get_bicycle_func(u, vehicleData)
    
    # Update state
    # x_y_and_yaw_terms
    x_in_y_and_yaw_terms, dx_norm = step_continuous_func(dx, x_in_y_and_yaw_terms, dt)
    print(str(x_in_y_and_yaw_terms))
    
    # Update path data index trackers:
    currentPt += 1
    nextPt += 1
    
    # Convert state back to terms of error
    print(str(x_in_y_and_yaw_terms[0]))
    x[0] = float(x_in_y_and_yaw_terms[0])
    e1 = x[0]
    
    # Update actual and goal (desired) yaw angle
    yaw = x_in_y_and_yaw_terms[2]
    goal_yaw = math.atan2((y_data[nextPt] - y_data[currentPt]) , (x_data[nextPt] - x_data[currentPt]))
    
    x[2] = yaw - goal_yaw
    x[1] = x_in_y_and_yaw_terms[1] + Vx * x[2]
    x[3] = x_in_y_and_yaw_terms[3]
    
    print("e1 = {:.4f}".format(float(x[0])))
    print("e2_dot = " + str(x[3]))
    print()
    print()
    
    x_g_old = x_g
    y_g_old = y_g
    
    # Update the position of the vehicle's CG:
    x_g = x_data[currentPt] - e1*math.sin(yaw)
    y_g = y_data[currentPt] + e1*math.cos(yaw)
    
    # Get time it took to perform state update this iteration
    t_end = time.time()
    timeForIteration = 0.00001
    if ((t_end - t_start) < dt):
        timeForIteration = (dt - (t_end - t_start))
    
    # Draw path
    drawPath(x_data, y_data)
    # Draw the position and orientation of the bicycle in our plot (bicycle appears as red line):
    drawBicycle(yaw, x_g, y_g, block = False, dt = timeForIteration)
    
