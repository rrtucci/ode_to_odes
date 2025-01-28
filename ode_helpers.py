

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
        fig.suptitle('State Variables vs. Time (Separate Axes)', y=0.93)
    else:
        # Plot all states on the same axes
        for i in range(num_states):
            plt.plot(time, states[i], label=f'State {i + 1}', color=f"C{i}")
        plt.xlabel('Time')
        plt.ylabel('State Variables')
        plt.title('State Variables vs. Time (Single Axes)')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()