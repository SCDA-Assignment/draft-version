import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# LQR class for Exercise 1.1

class LQR:

    # -----------------------------------------------------
    # Initialize model parameters
    # -----------------------------------------------------
    def __init__(self, H, M, C, D, R, sigma, T):

        self.H = H        # system matrix
        self.M = M        # control matrix
        self.C = C        # state cost matrix
        self.D = D        # control cost matrix
        self.R = R        # terminal cost matrix
        self.sigma = sigma
        self.T = T

        self.Dinv = np.linalg.inv(D)   # precompute D^{-1}


    # -----------------------------------------------------
    # Riccati ODE
    # -----------------------------------------------------
    def riccati_ode(self, t, S_flat):

        S = S_flat.reshape(2,2)   # reshape vector → matrix

        dS = (
            -2 * self.H.T @ S
            + S @ self.M @ self.Dinv @ self.M.T @ S
            - self.C
        )

        return dS.flatten()


    # -----------------------------------------------------
    # Solve Riccati equation on time grid
    # -----------------------------------------------------
    def solve_riccati(self, time_grid):

        sol = solve_ivp(
            self.riccati_ode,
            [self.T, 0],           # backward integration
            self.R.flatten(),            # terminal condition
            t_eval=time_grid[::-1]
        )

        S = sol.y.T.reshape(len(time_grid),2,2)[::-1]

        # interpolation function S(t)
        self.S = interp1d(time_grid, S, axis=0)


    # -----------------------------------------------------
    # Value function v(t,x) = x^T S(t) x
    # -----------------------------------------------------
    def value(self, t, x):

        S = self.S(t)

        return x.T @ S @ x


    # -----------------------------------------------------
    # Optimal control a(t,x)
    # -----------------------------------------------------
    def control(self, t, x):

        S = self.S(t)

        return - self.Dinv @ self.M.T @ S @ x