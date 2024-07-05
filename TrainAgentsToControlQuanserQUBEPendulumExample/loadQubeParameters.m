% LOADQUBEPARAMETERS loads parameters for Qube Swing Up Demo

% Copyright 2021 The MathWorks, Inc.

%% Plant parameters
R_m = 21.7;         % Motor resistance [Ohm]
mu_m = 3.08e-6;     % Damping of motor shaft [Nm/rad/s]
K_t = 0.042;        % Torque constant [Nm/A]
K_b = 0.0392;       % Back EMF contant [V/rad/s]
K_b_2 = 0.182;      % Back EMF contant [V/rad/s]
J_m = 4e-6;         % Motor shaft inertia [kg*m^2]
L_m = 4.98e-3;      % Motor Inductance [H]
rod_rad = 0.003;    % Arm rod radius [m]
p_rad = 0.0045;     % Pendulum radius [m]
L_p = 0.126;        % Length of Pendulum [m]
L_r = 0.103;        % Length of Arm rod [m]
m_p = 0.024;        % Mass of Pendulum [kg]
m_r = 0.095;        % Mass of Arm rod [kg]
J_p = m_p*(p_rad^2/4+L_p^2/12) + m_p*(L_p/2)^2;     % Inertia of Pendulum [kg*m^2]
J_r = m_r*(rod_rad^2/4+L_r^2/12) + m_r*(L_r/2)^2;   % Inertia of Arm rod [kg*m^2]
D_r = 0.001;        % Damping of Arm rod [Nm/rad/s]
D_r_2 = 1.88e-04;   % Damping of Arm rod [Nm/rad/s]
D_p = 8e-6;         % Damping og Pendulum [Nm/rad/s]
n = 1;              % Gera ratio
g = 9.81;           % Gravity [m/s^2]
Volt_Limit = 12;
angle_limit = 3*pi/4;

%% Sample time
Tc = 0.005;            % Time step for feedback controller. [s]
Tf = 10;               % Simulation stop time. [s]
Ts = Tc * 4;           % Time step for Reinforcement Learning agent. [s]
ts_PID = 0.005;        % Sample time of PID controller
ts_raspi = 0.005;      % Sample time for Raspberry Pi
tf_raspi = 10;         % Simulation time for Raspberry Pi
rl_sample_rate = 4;    
init_time = 1;

%% Initial condition
theta0 = 0;
phi0 = 0;
dtheta0 = 0;
dphi0 = 0;

%% PID gains
theta_gain   = 0.162;
dtheta_gain  = 0.0356;
phi_gain     = 40;
dphi_gain    = 2;

%% Configuration
load("configs.mat")
