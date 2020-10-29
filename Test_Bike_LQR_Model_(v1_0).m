clear all;
close all;
clc;

%--------------------------------------------------------------------------------------------------

% The following code is an LQR controller for a bicycle based off an
% excerpt from the textbook "Vehicle Dynamics and Control" by Rajesh
% Rajamani.The following model seemed to be for a representation of a car
% as a bike by condensing each axle's wheel to a single, idealized wheel
% creating a 'bike'.

% The ensuing state space model relies on two key ASSUMPTIONS: (1) small slip
% angle and (2) negligible bank angle. Slip angle being the angle between the
% orientation vector of the 'bike's' front wheel and said wheel's velocity
% vector. The bank angle is the angle of the road the 'bike' is traveling
% along with respect to gravity. Equations needed to account for larger
% bank angles and non-negligible bank angles are included in the textbook
% but will be omitted for now for simplicity's sake.

% Notes:
% - "c.g." = center of gravity
% - Local x-axis of model represents vehicle's longitudinal axis, y-axis
% represents its lateral axis
% There is a global X-Y elsewhere that defines the vehicle's position

%--------------------------------------------------------------------------------------------------

% Define constants (FILL IN VALUES LATER!!!)
Cf = 1;                             % Cornering stiffness of front tire(s)
Cr = 1;                             % Cornering stiffness of rear tire(s)
m = 1;                              % Total mass of vehicle
lf = 1;                             % Length btwn vehicle's c.g. and front axle c.g.
lr = 1;                             % Length btwn vehicle's c.g. and rear axle c.g.
Vx = 1;                             % Vehicle's longitudinal velocity
Iz = 1;                             % Vehcile's yaw moment of inertia

% NEED TO UPDATE INITIAL CONDITIONS!!!
% Define initial conditions
x0 = [];

% Define matrices for state space equations
A = [0      1                                       0       0;
     0      (-(2.*Cf + 2.*Cr) ./ (m.*Vx))           0       (-Vx - (2.*Cf.*lf - 2.*Cr.*lr) ./ (m.*Vx));
     0      0                                       0       1
     0      (-(2.*Cf.*lf - 2.*Cr.*lr) ./ (Iz*Vx))   0       (-(2.*Cf.*(lf^2) + 2.*Cr.*(lr^2)) ./ (Iz*Vx))];

B = [0
    ((2.*Cf) ./ m);
     0;
    ((2.*Cf.*lf) ./ Iz)];

% ADJUST C AND D MATRICES LATER! DOESN'T SEEM LIKE ANYTHING NEEDS MEASURED
% ATM
C = [1 0 0];

D = 0;


% NEED TO UPDATE Q AND R MATRICES WITH EXPERIMENTATION!!!!
Q = [4 0;
    0 1];                   % Penalize angular position and ang. veloc. error

R = 1;                      % Penalize input torque

% Get gain matrix for feedback system
K = lqr(A, B, Q, R);

% Run response to initial condition
t = 0.05;
% [y, t, x] = initial(sys, x0, t);
% initial(sys, x0);

% Will be used to terminate the state-update loop if runs for too long
flag = 0;

% Initialize state and input vectors
x = x0;
u = 0;      % Input torque

% Update system until equilibrium acheived
while ((abs(x(1)) ~= 0.0001) || (abs(x(2))) ~= 0.0001 || flag == 10 / 0.07)
    % Update system
    u = -K*x;
    
    % Will contain new theta and theta_dot values
    % b/c they are our states
    x = step_X(b, M, L, g, x, u, t);
    
    flag = flag + 1;
    theta = x(1);
    
    % Every iteration, draw a line representing the pendulum rod
    plot([0, sin(theta)], [0, cos(theta)]);
    xlim([-9 9])
    ylim([0 1])
    pause(t)
end

%--------------------------------------------------------------------------
% Function Definitions:

% From Joshua Spisak


function [dx] = compute_pendulum_dx(b, M, L, g, x, u)
    %% compute f dot for a given state & system...
    theta_dot = x(2);
    theta_dot_dot = -u - (-b/(M.*L^2)) .* x(2) + (g/L)*x(1);
    dx = [theta_dot; theta_dot_dot];
end

function [X_o] = step_X(b, M, L, g, X_i, u, dt)
    %% Use runge-kutta equations to forward sim your system...
    dx1 = compute_pendulum_dx(b, M, L, g, X_i, u);
    dx2 = compute_pendulum_dx(b, M, L, g, X_i + 0.5*dt*dx1, u);
    dx3 = compute_pendulum_dx(b, M, L, g, X_i + 0.5*dt*dx2, u);
    dx4 = compute_pendulum_dx(b, M, L, g,  X_i + dt*dx3, u);
    dxf = (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4) * dt / 6.0;
    X_o = X_i + dxf;
    
    
    % dx = compute_dendumlum_dx(x, u)
    % X_o = X_i + dx * dt
end
