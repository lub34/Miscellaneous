# The finite horizon LQR function from before 11/20, when the code was working for the first time:

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
