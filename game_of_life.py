import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
import random


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
    im = ax.imshow(
        matrix, cmap=cmap, norm=norm, interpolation="nearest"
    )
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.ion()
    plt.show()
    return fig, ax, im


def update_visualization(im, matrix):
    """Updates the existing visualization with the new matrix data."""
    im.set_data(matrix)
    plt.draw()
    # Adjust pause duration for desired animation speed
    plt.pause(0.5)  # Reduced pause for potentially faster algorithms


# --- Algorithm Logic Examples ---
def random_flip_step(matrix):
    """
    Example Algorithm Function: Flips a single random cell in the matrix.

    Args:
      matrix (np.ndarray): The current state of the matrix.

    Returns:
      np.ndarray: The next state of the matrix.
    """
    # Important: Work on a copy to avoid modifying the input matrix directly
    # if the caller might reuse the old state.
    new_matrix = matrix.copy()
    n = new_matrix.shape[0]
    if n > 0:  # Avoid error on empty matrix
        row = random.randint(0, n - 1)
        col = random.randint(0, n - 1)
        new_matrix[row, col] = 1 - new_matrix[row, col]  # Flip 0 to 1 or 1 to 0
    return new_matrix


def do_nothing_step(matrix):
    """
    Example Algorithm Function: Returns the matrix unchanged. Useful for testing setup.

    Args:
      matrix (np.ndarray): The current state of the matrix.

    Returns:
      np.ndarray: The same matrix.
    """
    # Can return the same object if algorithm guarantees no side effects,
    # otherwise return matrix.copy()
    return matrix


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
    current_matrix = initial_matrix.copy()  # Start with a copy
    matrix_size_n = current_matrix.shape[0]
    matrix_size_m = current_matrix.shape[1]

    try:
        # Set up the initial visualization
        fig, ax, im = visualize_matrix(
            current_matrix, title=f"{title_prefix} - Initial State"
        )

        # Simulation Loop
        print(f"\nStarting simulation with {algorithm_step_func.__name__}...")
        for i in range(num_iterations):
            # --- Apply the provided algorithm logic ---
            next_matrix = algorithm_step_func(current_matrix)
            # ------------------------------------------

            # Basic validation (optional but good practice)
            if not isinstance(next_matrix, np.ndarray):
                print(
                    f"Error: Algorithm function did not return a NumPy array at iteration {i+1}. Stopping."
                )
                break
            if next_matrix.shape != (matrix_size_n, matrix_size_m):
                print(
                    f"Warning: Matrix dimensions changed from ({matrix_size_n},{matrix_size_m}) to {next_matrix.shape} at iteration {i+1}. Stopping."
                )
                # You might want different handling, e.g., trying to resize the plot
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
        print("Final Matrix:\n", current_matrix)

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
    matrix_size = 100  # asked for 100 x 100
    probability_one = 0.5  # e.g., 50% chance for a cell to start as 1
    num_iterations = 100  # Number of simulation steps

    # --- CHOOSE YOUR ALGORITHM HERE ---
    # Assign the function name of the algorithm you want to run
    algorithm_to_run = random_flip_step
    # algorithm_to_run = do_nothing_step
    # ----------------------------------

    # 3. Create the initial matrix using the specified probability
    try:
        initial_matrix = create_random_binary_matrix(matrix_size, probability_one)
        print(
            f"Initial Matrix ({matrix_size}x{matrix_size}) with P(1)={probability_one}:\n",
            initial_matrix,
        )

        # 4. Run the simulation, passing the chosen algorithm function
        run_simulation(
            initial_matrix=initial_matrix,
            algorithm_step_func=algorithm_to_run,
            num_iterations=num_iterations,
            title_prefix=f"{matrix_size}x{matrix_size} Matrix ({algorithm_to_run.__name__})",
        )
    except ValueError as ve:
        print(f"Error setting up simulation: {ve}")
