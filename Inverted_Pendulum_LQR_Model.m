clear all;
close all;
clc;

% Define constants
L = 0.5;                    % Arbitrary length of pendulum rod [m]
b = 1;                      % Arbitrary value for damping constant
g = 9.81;                   % Gravitational constant [m/s^2]
M = 2;                      % Point mass on end of pendulum [kg]       

% Define initial conditions
x0 = [50;                % 5 [rad] deflection from vertical axis         
     4];                % Mass traveling 0 [rad/s] initially

% Define matrices for state space equations
A = [0              1;
    (g/L) (-b/(M*L^2))];

B = [0;
    -1];

C = [1 0];

D = 0;

Q = [4 0;
    0 1];                   % Penalize angular position and ang. veloc. error

R = 1;                      % Penalize input torque

% Get gain matrix for feedback system
K = lqr(A, B, Q, R);

% Run response to initial condition
t = 0.05;

% Will be used to terminate the state-update loop if runs for too long
flag = 0;

% Initialize state and input vectors
x = x0;
u = 0;      % Input torque
while ((abs(x(1)) ~= 0.0001) || (abs(x(2))) ~= 0.0001 || flag == 10 / 0.07)
    % Update system
    u = -K*x;
    
    % Update the state values
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
end
