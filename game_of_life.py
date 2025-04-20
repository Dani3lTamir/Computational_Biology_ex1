import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import time
import random
from matplotlib.widgets import Button

# Global variable to control the simulation
paused = False
wraparound = False


def toggle_pause(event):
    """Toggle the paused state of the simulation."""
    global paused
    paused = not paused


# --- Matrix Creation ---
def create_random_binary_matrix(n, probability_of_one=0.5):
    """Creates an n x n NumPy array with 0s and 1s based on probability.

    Args:
        n (int): Dimension of the square matrix.
        probability_of_one (float): The probability (between 0.0 and 1.0)
                                        for a cell to be initialized as 1. Defaults to 0.5.

    Returns:
        np.ndarray: The generated n x n matrix.

    Raises:
        ValueError: If probability_of_one is not between 0.0 and 1.0.
    """
    if not 0.0 <= probability_of_one <= 1.0:
        raise ValueError("Probability must be between 0.0 and 1.0")
    # Generate random floats between 0 and 1
    random_floats = np.random.rand(n, n)
    # Where float < probability, set to 1 (True), otherwise 0 (False)
    # Then convert boolean array to integer array (True->1, False->0)
    return (random_floats < probability_of_one).astype(int)


# --- Visualization Functions ---
def visualize_matrix(matrix, title="Matrix Visualization"):
    """Initializes the visualization of the matrix."""
    fig, ax = plt.subplots()
    cmap = mcolors.ListedColormap(["white", "black"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

    # pause/continue button
    ax_button = plt.axes([0.81, 0.05, 0.1, 0.075])
    btn = Button(ax_button, "Pause")
    btn.on_clicked(toggle_pause)

    plt.ion()
    plt.show()
    return fig, ax, im, btn


def update_visualization(im, matrix):
    """Updates the existing visualization with the new matrix data."""
    im.set_data(matrix)
    plt.draw()
    # Adjust pause duration for desired animation speed
    plt.pause(0.8)  # Reduced pause for potentially faster algorithms


def block_step(matrix, iteration):
    """
    Does the algorithm described in the exercise.
    we take blocks of 2X2 starting from the bottom right corner of each 2X2

    Args:
        matrix (np.ndarray): The current state of the matrix.
        iteration (int): The current iteration number.
    Returns:
        np.ndarray: a new matrix of the next phase.
    """
    # Important: Work on a copy to avoid modifying the input matrix directly
    new_matrix = matrix.copy()
    n = new_matrix.shape[0]
    m = new_matrix.shape[1]
    if(n < 2 or m < 2):
        return new_matrix
    
    block_start = (iteration % 2) # blue lines start from (1,1) and red from (0,0)
    # Iterate over each bottom right corner in a matrix of 2x2 blocks
    for i in range(block_start, n , 2):
        for j in range(block_start, m, 2):
            if(not wraparound and ((i-1) < 0 or (j-1)<0)):
                continue
            
            grid_Cells = [(i, j), ((i-1)%n, j), (i, (j-1)%m), ((i-1)%n, (j-1)%m)]
            # Count the number of 1s in the neighborhood (4 directions)
            count = 0
            for cell in grid_Cells:
                count += new_matrix[cell]

            # Apply rules based on count
            if count == 0 or count == 1 or count == 4:  # 0, 1, or 4 ones -> flip all
                for cell in grid_Cells:
                    new_matrix[cell] = 1 - new_matrix[cell]
            elif count == 2:
                continue  # No change needed for 2 ones
            elif count == 3:  # 3 ones -> flip all and then rotate in 180 degrees
                for cell in grid_Cells:
                    new_matrix[cell] = 1 - new_matrix[cell]
                # Rotate 180 degrees
                # cell 1 and cell 4 switch
                # cell 2 and cell 3 switch
                new_matrix[grid_Cells[0]], new_matrix[grid_Cells[3]] = (
                    new_matrix[grid_Cells[3]],
                    new_matrix[grid_Cells[0]]
                )

                new_matrix[grid_Cells[1]], new_matrix[grid_Cells[2]] = (
                    new_matrix[grid_Cells[2]],
                    new_matrix[grid_Cells[1]],
                )

    return new_matrix

# --- Simulation Runner ---
def run_simulation(
    initial_matrix,
    algorithm_step_func,
    num_iterations,
    title_prefix="Life Simulation",
):
    """
    Runs the simulation loop, applying the algorithm logic and updating visualization.

    Args:
        initial_matrix (np.ndarray): The starting matrix.
        algorithm_step_func (callable): A function that takes the current matrix
                                            (np.ndarray) as input and returns the matrix
                                            (np.ndarray) for the next iteration.
        num_iterations (int): The number of steps to simulate.
        title_prefix (str): Base text for the plot title.
    """
    global paused  # Access the global paused variable
    current_matrix = initial_matrix.copy()  # Start with a copy
    matrix_size_n = current_matrix.shape[0]
    matrix_size_m = current_matrix.shape[1]

    try:
        # Set up the initial visualization
        fig, ax, im, btn = visualize_matrix(
            current_matrix, title=f"{title_prefix} - Initial State"
        )

        # Simulation Loop
        print(f"\nStarting simulation with {algorithm_step_func.__name__}...")
        # we start from odd round "1"
        for i in range(1, num_iterations + 1):
            # Check if simulation is paused
            while paused:
                plt.pause(0.1)
                if not plt.fignum_exists(fig.number):
                    print("Plot window closed manually. Stopping simulation.")
                    return

            # --- Apply the provided algorithm logic ---
            next_matrix = algorithm_step_func(current_matrix, i)
            # ------------------------------------------

            # Basic validation of the returned matrix
            if not isinstance(next_matrix, np.ndarray):
                print(
                    f"Error: Algorithm function did not return a NumPy array at iteration {i+1}. Stopping."
                )
                break
            if next_matrix.shape != (matrix_size_n, matrix_size_m):
                print(
                    f"Warning: Matrix dimensions changed from ({matrix_size_n},{matrix_size_m}) to {next_matrix.shape} at iteration {i+1}. Stopping."
                )
                break

            current_matrix = next_matrix  # Update the state for the next iteration

            # Update the title
            ax.set_title(f"{title_prefix} - Iteration {i+1}/{num_iterations}")

            # Update the visualization
            update_visualization(im, current_matrix)

            # Check if the figure window was closed manually
            if not plt.fignum_exists(fig.number):
                print("Plot window closed manually. Stopping simulation.")
                break

        print("\nSimulation finished or stopped.")
    #    print("Final Matrix:\n", current_matrix)

    except Exception as e:
        print(f"\nAn error occurred during simulation: {e}")
        import traceback

        traceback.print_exc()  # Print detailed error information
    finally:
        # Ensure interactive mode is off and final plot is shown
        # regardless of how the loop ended.
        plt.ioff()
        # Only call show if the figure potentially still exists
        if "fig" in locals() and plt.fignum_exists(fig.number):
            ax.set_title(f"{title_prefix} - Final State")  # Update title for final view
            plt.show()
        elif (
            "fig" in locals()
        ):  # Close the figure object if it exists but window is gone
            plt.close(fig)


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Parameters
    matrix_size = 4  # asked for 100 x 100
    probability_one = 0.5  # e.g., 50% chance for a cell to start as 1
    num_iterations = 100  # Number of simulation steps

    # if got "wraparound" in cli do the wraparound version, otherwise no
    if(len(sys.argv) > 1):
        if sys.argv[1] == "wraparound":
            wraparound = True
    
    if(wraparound):
        title = "with wraparound"
    else:
        title = "without wraparound"
    # ----------------------------------

    # 3. Create the initial matrix using the specified probability
    try:
        initial_matrix = create_random_binary_matrix(matrix_size, probability_one)
        print(
            f"Initial Matrix ({matrix_size}x{matrix_size}) with P(1)={probability_one}\n"
        )

        # 4. Run the simulation, passing the chosen algorithm function
        run_simulation(
            initial_matrix=initial_matrix,
            algorithm_step_func=block_step,
            num_iterations=num_iterations,
            title_prefix=f"{matrix_size}x{matrix_size} Matrix ({title})",
        )
    except ValueError as ve:
        print(f"Error setting up simulation: {ve}")
