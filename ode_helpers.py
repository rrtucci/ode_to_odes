import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.lines as mlines
import matplotlib.pyplot as plt


def state_plotter(time, states, separate_axes=False):
    """
    Plots the states of a solution as a function of time.

    Parameters:
    - time: Array-like, the time points (sol.t).
    - states: 2D array-like, the state matrix where each row corresponds to a variable (sol.y).
    - separate_axes: Bool, if True, each state is plotted on its own set of axes.
                     If False, all states are plotted on the same axes.

    Returns:
    - None
    """
    num_states = states.shape[0]  # Number of state variables (rows in sol.y)

    if separate_axes:
        # Plot each state in its own subplot
        fig, axes = plt.subplots(num_states, 1, figsize=(8, 4 * num_states),
                                 sharex=True)
        if num_states == 1:  # If there's only one state, axes won't be a list
            axes = [axes]
        for i in range(num_states):
            axes[i].plot(time, states[i], label=f'State {i + 1}',
                         color=f"C{i}")
            axes[i].set_ylabel(f'State {i + 1}')
            axes[i].grid(True)
            axes[i].legend()
        axes[-1].set_xlabel('Time')
        # fig.suptitle('State Variables vs. Time (Separate Axes)', y=0.93)
    else:
        # Plot all states on the same axes
        for i in range(num_states):
            plt.plot(time, states[i], label=f'State {i + 1}', color=f"C{i}")
        plt.xlabel('Time')
        plt.ylabel('State Variables')
        # plt.title('State Variables vs. Time (Single Axes)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_phase_plane(f,
                     g,
                     x_range=(-5, 5),
                     y_range=(-5, 5),
                     density=20,
                     scale=15,
                     trajectories=None):
    """
    Plots a phase plane diagram with a quiver plot, nullclines, and optional trajectories.

    Parameters:
        f (function): dx/dt = f(x, y), function of x and y
        g (function): dy/dt = g(x, y), function of x and y
        x_range (tuple): Range of x values (min, max)
        y_range (tuple): Range of y values (min, max)
        density (int): Number of arrows in each direction for the quiver plot
        scale (float): Scaling factor for quiver arrows (higher values make arrows shorter)
        trajectories (list of tuples): Optional, initial conditions for sample trajectories
    """

    x_vals = np.linspace(x_range[0], x_range[1], density)
    y_vals = np.linspace(y_range[0], y_range[1], density)
    X, Y = np.meshgrid(x_vals, y_vals)

    U = f(X, Y)
    V = g(X, Y)

    # Compute the magnitude for coloring
    magnitude = np.sqrt(U ** 2 + V ** 2)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Quiver plot with scaling control
    quiver = ax.quiver(X,
                       Y,
                       U,
                       V,
                       #magnitude,
                       # cmap='gray_r',
                       scale=scale,
                       scale_units='xy')

    # Add a colorbar to indicate magnitude
    # cbar = plt.colorbar(quiver, ax=ax)
    # cbar.set_label('Vector Magnitude')

    # Plot nullclines (where f(x, y) = 0 and g(x, y) = 0)
    ax.contour(X, Y, f(X, Y), levels=[0], colors='green', linewidths=2)
    ax.contour(X, Y, g(X, Y), levels=[0], colors='red', linewidths=2)

    # Legend for nullclines
    legend_x = mlines.Line2D([], [], color='green',
                             linewidth=2, label=r'$\dot{x} = 0$')
    legend_y = mlines.Line2D([], [], color='red',
                             linewidth=2, label=r'$\dot{y} = 0$')
    ax.legend(handles=[legend_x, legend_y], loc='upper right')

    # Plot sample trajectories if provided
    if trajectories:
        from scipy.integrate import solve_ivp
        for x0, y0 in trajectories:
            t_span = (0, 10)
            sol = solve_ivp(lambda t, z: [f(z[0], z[1]), g(z[0], z[1])],
                            t_span, [x0, y0], dense_output=True)
            ax.plot(sol.y[0], sol.y[1], 'r', lw=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    # ax.set_title('Phase Plane with Quiver Plot and Nullclines')

    plt.grid()
    plt.show()


if __name__ == "__main__":
    def main1():
        #  Define derivative function
        def f(t, y, c):
            ydot = [c[0] * np.cos(c[1] * t),
                    c[2] * y[0] + c[3] * t]
            return ydot

        #  Define time spans, initial values, and constants
        tspan = np.linspace(0, 5, 100)
        yinit = [0, -3]
        c = [4, 3, -2, 0.5]

        #  Solve differential equation
        sol = solve_ivp(lambda t, y: f(t, y, c),
                        [tspan[0], tspan[-1]],
                        yinit,
                        t_eval=tspan,
                        rtol=1e-5)
        #  Plot states
        state_plotter(sol.t, sol.y, separate_axes=True)


    def main2():
        def dx_dt(x, y): return y

        def dy_dt(x, y): return -x

        plot_phase_plane(dx_dt,
                         dy_dt,
                         x_range=(-5, 5),
                         y_range=(-5, 5),
                         density=10,
                         scale=5,
                         # %trajectories=[(2, 0), (-2, 1), (0, -2)]
                         )


    # main1()
    main2()
