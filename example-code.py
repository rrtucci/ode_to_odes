# this comes from https://pundit.pratt.duke.edu/wiki/Python:Ordinary_Differential_Equations/Examples
# function state_plotter was written by ChatGPT
# %% Imports
import numpy as np
from scipy.integrate import solve_ivp
from ode_helpers import state_plotter

# %% Define derivative function
def f(t, y, c):
    dydt = [c[0]*np.cos(c[1]*t), c[2]*y[0]+c[3]*t]
    return dydt

# %% Define time spans, initial values, and constants
tspan = np.linspace(0, 5, 100)
yinit = [0, -3]
c = [4, 3, -2, 0.5]

# %% Solve differential equation
sol = solve_ivp(lambda t,
                y: f(t, y, c),
                [tspan[0], tspan[-1]],
                yinit,
                t_eval=tspan,
                rtol = 1e-5)
# %% Plot states
state_plotter(sol.t, sol.y, separate_axes=True)