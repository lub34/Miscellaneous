import scipy.linalg
import numpy as np
import math
import csv
import matplotlib.pyplot as plt

# Performs LQR on a state space model.
# Returns the LQR gain K, the resulting matrix from solving the
# Ricatti equation, and the eigenvalues of the updated A matrix
# often referred to as A_cl = A_closed_loop = A - B*K
def lqr(A,B,Q,R):
    # Thanks http://www.mwm.im/lqr-controllers-with-python/
    # Python3 didn't like the control library :(
    
    # Solve the Ricatti equation using our state space equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
    
    # cost = integral x.T*Q*x + u.T*R*u
    #compute the LQR gain K
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    return K, X, eigVals

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
    return np.flip(np.array(Ps))

class VehicleModel(object):
    Cf = ...
    # defaults.....

# Updates the bicycle's states.
def get_bicycle_func(u, model):
    def bicycle_func(x):
        # ASK JOSH HOW TO DEFINE THESE CONSTANTS ONCE AS GLOBAL VARIABLES THEN USE IN MAIN CODE AND THIS FUNCTION
        # Define constants (All are temporary values for now, FILL IN PROPERLY LATER!!!)
        Cf = 447.9                             # Cornering stiffness of front tire(s) [N/deg]
        Cr = 447.9                             # Cornering stiffness of rear tire(s) [N/deg]
        m = 761                              # Total mass of vehicle [kg] (CHECK: Add wheel masses and fuel mass?)
        lf = 1.54                             # Length btwn vehicle's c.g. and front axle c.g. [m]
        lr = 2.554                             # Length btwn vehicle's c.g. and rear axle c.g. [m]
        Vx = 80                             # Vehicle's longitudinal velocity [m/s]
        Iz = 600                             # Vehcile's yaw moment of inertia (GUESSED FOR NOW)
    
        e1_dot = x[1]
        e1_dot_dot = (-(2*Cf + 2*Cr) / (m*Vx)) * x[1] + ((2*Cf + 2*Cr) / m) * x[2] + \
            ((2*Cf*lf - 2*Cr*lr) / (m*Vx)) * x[3] + ((2*Cf) / m) * u
        e2_dot = x[3]
        e2_dot_dot = (-(2*Cf*lf - 2*Cr*lr) / (Iz*Vx)) * x[1] + ((2*Cf*lf - 2*Cr*lr) / Iz) * x[2] + \
            -((2*Cf*(lf**2) + 2*Cr*(lr**2)) / (Iz*Vx)) * x[3] + ((2*Cf*lf) / Iz) * u
        
        return [e1_dot, e1_dot_dot, e2_dot, e2_dot_dot]
    return bicycle_func

"""
Draws new position of bicycle in 2D space.
Parameters:
    - state = 1D array representing current state vector x from state-space model.
              Assumes third element of 'state' is vehicle's yaw angle (index 2).
"""
def drawBicycle(yaw, x_g, y_g):
    # dx_f = lf*sin(yaw); dx_f = change in x between bicycle CG and front axle CG location.

    # (dx_f, dy_f) = position of front axle's CG relative to bicycle's CG.
    # (dx_r, dy_r) = position of rear axle's CG relative to bicycle's CG.
    # a is the distance from the CG to the front axle CG (not to scale -- just so visible on graph).
    # b is the distance from the CG to the rear axle CG (not to scale -- just so visible on graph).
    a = 20
    b = 30
    
    dx_f = a*math.sin(yaw)
    dy_f = a*math.cos(yaw)
    
    # Potential bug: Yaw being defined from [pi,-pi] could screw up math here, messing vehicle's orientation
    dx_r = b*math.sin(math.pi + yaw)
    dy_r = b*math.cos(math.pi + yaw)
    
    # Solve for the global locations of each axle's CG
    x_f = x_g + dx_f
    y_f = y_g + dy_f
    
    x_r = x_g + dx_r
    y_r = y_g + dy_r
    
    # Finally, plot the bicycle's initial position and orientation:
    x_axle_values = [x_r, x_f]
    y_axle_values = [y_r, y_f]
    plt.plot(x_axle_values, y_axle_values, 'r')
    plt.show()
    return None

"""
Defines positional and orientation error of bicycle's current state.
Parameters:
    - state = 1D array representing current state vector x from state-space model.
              Assumes third element of 'state' is vehicle's yaw angle (index 2).
              
    - currentPt = Index of current point on desired path data set
    
    - nextPt = Index of next point on desired path data set
Returns:
    - True if negligible error
    - False if non-negligible error
"""
def negligibleError(state, currentPt, nextPt):
    # Get bicycle's normal distance from its point of rotation (y) and yaw angle 
    y = state[0]
    yawAngle = state[2]
            
    # Get desired yaw angle by calculating slope between first two points of
    # desired path (used to measure orientation error)
    goal_yaw = math.atan((y_data[nextPt] - y_data[currentPt]) / (x_data[nextPt] - x_data[currentPt]))
    
    # Define current location of bicycle's point of rotation in terms of states.
    # (Point 'O' will be known as the point of rotation and it lies at (x_0, y_0))
    x_o = x_g + y*math.cos(yawAngle)
    y_o = y_g + y*math.sin(yawAngle)
    
    # Define the two 'alpha' angles for measuring positional error:
    
    # alpha1 takes the next point on the desired path pt. P, draws a line through it that's
    # parallel to the x-axis, draws a line from said point to pt. O, then finds the angle
    # between those two lines.
    
    # alpha2 is the same angle as alpha1, except it is its value when the bicycle's CG lies
    # on pt. P.
    
    # Since this is the initial calculation of these values, assume index 0 is the point on the
    # path before pt. P, and pt. P's (x,y) data can be found at index 1 in the x-, y-data arrays:
    x_p = x_data[desiredPt]
    y_p = y_data[desiredPt]
    
    alpha1 = math.acos( (x_p - x_o) / math.sqrt((y_p - y_o)**2 + (x_p - x_o)**2) )
    alpha2 = math.acos( (x_p - x_o) / y )
    
    """
    Determine if bicycle is on path.
    Explanation: Bicycle is considered 'on' the path if two conditions are met:
        1) yawAngle ~= goal_yaw (if so, then bicycle is properly oriented in space)
        2) alpha1 ~= alpha2 (if so, then bicycle's CG is relatively on the desired path point)
    Method: Calculates percent error, using the current state variables, to measure error.
    """
    yawError = math.fabs( (yawAngle - goal_yaw) / goal_yaw ) * 100
    alphaError = math.fabs( (alpha1 - alpha2) / alpha2 ) * 100

    # If negligible error, return True; vice versa (error in %)
    if ( (yawError <= 5 ) and (alphaError <= 5) ):
        return True
    else:
        return False

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
    
    # Get highest index of each data array (to be used later)
    # (Assuming x- and y- data sets will be of same length)
    last_point_in_path = len(x_data) - 1
    
    for row in reader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))
        
    print(x_data)
    
    plt.scatter(x_data, y_data, s = 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Sample Path to Test Bicycle Controller On')
    plt.legend()
    
    """
    # Run this to see the data in the csv file
    # For isolating a single COLUMN (in this case, the first one)
    for row in reader:
        print(''.join(row[0] + ' ' + row[1]))
    """
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
# Initialize constants and state space info 

# NEED TO FIND LOCATION OF CG ALONG VEHICLE'S LONGITUDINAL AXIS
# TO GET lf AND lr!!!
# Define constants (All are temporary values for now, FILL IN PROPERLY LATER!!!)
Cf = 447.9                             # Cornering stiffness of front tire(s) [N/deg]
Cr = 447.9                             # Cornering stiffness of rear tire(s) [N/deg]
m = 761                              # Total mass of vehicle [kg] (CHECK: Add wheel masses and fuel mass?)
lf = 1.54                             # Length btwn vehicle's c.g. and front axle c.g. [m]
lr = 2.554                             # Length btwn vehicle's c.g. and rear axle c.g. [m]
Vx = 80                             # Vehicle's longitudinal velocity [m/s]
Iz = 600                             # Vehcile's yaw moment of inertia (GUESSED FOR NOW)

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
# one usually does with Q and R matrices). Edit Q matrix during tuning.
Q = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])

# Penalizes inputs (CHECK: What are our inputs???)
R = np.array([[1]])

# Define initial (x,y) point of bicycle's CG -- pt. G (will be updated in loop)
x_g = 50
y_g = 0.0005

# Initialize the state vector [y, y_dot, yaw, yaw_dot]:
# y = distance normal to bicycle between its CG and its pt. of rotation [m].
# yaw = yaw angle of bike; an orientation angle [rads].
x = np.array([[0.5,
               1,
               0.524,
               -0.1]]).T

# Initialize index of desired current and subsequent point along the path:
on_Pt = 0
upcoming_Pt = on_Pt + 1

#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

# Until there is negligible error in the vehicle's position and orientation wrt the path,
# use LQR controller to guide vehicle to satisfactory position and orientation.

K = finite_horizon_lqr(A, B, Q, R, Q, 20.0, dt)
dt = 0.1
tracker = 0;
while (tracker != last_point_in_path):
    # Reset index counters for current and next point on path if either reaches
    # last point in path (returns to start of path -- LOGIC ONLY WORKS FOR looping  paths)
    if (on_Pt == last_point_in_path):   
        on_Pt = 0
    elif(upcoming_Pt == last_point_in_path):
        upcoming_Pt = 0  
        
    # Get desired yaw angle by calculating slope between first two points of
    # desired path (used to measure orientation error)
    goal_yaw = math.atan((y_data[upcoming_Pt] - y_data[on_Pt]) / (x_data[upcoming_Pt] - x_data[on_Pt]))
    
    # For readability, define states y and yaw outside of the 1D array
    y = x[0]
    yawAngle = x[2] + goal_yaw                   # yaw = e2 + desired yaw
    
    # Draw the position and orientation of the bicycle in our plot (bicycle appears as red line):
    drawBicycle(yawAngle, x_g, y_g)
    
    # Convert yawAngle to a value between 0 and pi if not already:
    # Note: math.remainder(z,w) divides z by w. The remaider is returned with one caveat,
    # if remainder is >= w/2 then it is returned negative. Hence, the following usage of it
    # will return an angle in rads between [-pi, pi].
    yawAngle = math.remainder(yawAngle, math.pi)
    
    u = -K * x
    
    dx = get_bicycle_func(x, u)
      
    # Update state
    x = step_continuous_func(dx, x, dt)
    
    # Update path data index trackers:
    on_Pt += 1
    upcoming_Pt += 1
    
    # Update the position of the vehicle's CG:
    # x_g =
    # y_g = 
