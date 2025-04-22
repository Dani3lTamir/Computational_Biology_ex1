import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import time
import random
from matplotlib.widgets import Button, TextBox, CheckButtons, RadioButtons

# Global variables to control the simulation
paused = True  # Start paused to allow configuration
wraparound = False
reset_requested = False
current_iteration = 0
stability = 0
max_iterations = 250
matrix_size = 100  # Default size
probability_one = 0.5
initial_pattern = 'random'  

# Initialize with default values that will be overwritten
fig, ax, im = None, None, None
initial_matrix = None
current_matrix = None
size_text = None
pattern_radio = None



def toggle_pause(event):
    """Toggle the paused state of the simulation."""
    global paused
    paused = not paused
    if paused:
        pause_button.label.set_text("Start/Continue")
    else:
        pause_button.label.set_text("Pause")
    plt.draw()


def toggle_wraparound(label):
    """Toggle the wraparound state of the simulation."""
    global wraparound
    wraparound = not wraparound
    update_title()
    plt.draw()

def change_pattern(label):
    """Change the initial pattern type."""
    global initial_pattern, reset_requested
    initial_pattern = label.lower()
    reset_requested = True
    plt.draw()

def reset_simulation(event):
    """Reset the simulation to its initial state."""
    global reset_requested, current_iteration, current_matrix, initial_matrix, paused, stability
    reset_requested = True
    paused = True  # Pause the simulation
    pause_button.label.set_text("Start/Continue")
    current_iteration = 0
    stability = 0
    initial_matrix = create_random_binary_matrix(matrix_size, probability_one)
    current_matrix = initial_matrix.copy()
    im.set_data(current_matrix)
    im.set_extent(
        [-0.5, matrix_size - 0.5, matrix_size - 0.5, -0.5]
    )  # Update extent for new size
    update_title()
    plt.draw()


def update_max_iterations(text):
    """Update the maximum number of iterations."""
    global max_iterations
    try:
        max_iterations = int(text)
        update_title()
        plt.draw()
    except ValueError:
        pass  # Ignore invalid input


def update_matrix_size(text):
    """Update the matrix size and reset the simulation."""
    global matrix_size, reset_requested
    try:
        new_size = int(text)
        if new_size > 7 and new_size % 2 == 0: # Ensure size is even and at least 8x8
            matrix_size = new_size
            reset_requested = True
            update_title()
            plt.draw()
    except ValueError:
        pass  # Ignore invalid input


def update_title():
    """Update the plot title with current parameters."""
    ax.set_title(
        f"{matrix_size}x{matrix_size} Matrix, P(1)={probability_one} "
        f"{'with' if wraparound else 'without'} wraparound - "
        f"Iteration {current_iteration}/{max_iterations}"
        f" - Stability: {stability}"
    )
    
def stability_calculator(old_matrix, new_matrix):
    """Calculates the stability of the matrix. summing the differences between the two matrices.
    Args:
        old_matrix (np.ndarray): The previous state of the matrix.
        new_matrix (np.ndarray): The current state of the matrix.
    Returns:
        float: Stability value rounded to 4 decimal places
    """
    stability = 1 - ((np.sum(np.abs(old_matrix - new_matrix))) / matrix_size**2)
    return np.around(stability, 4)


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
    return (np.random.rand(n, n) < probability_of_one).astype(int)



def create_horizontal_glider(n):
    """Creates an n x n matrix with a horizontal glider pattern."""
    matrix = np.ones((n, n), dtype=int)
    center = n // 2
    # Glider pattern (adjust positions if needed)
    matrix[center, center] = 0
    matrix[center+1, center+1] = 0
    matrix[center+1, center+2] = 0
    matrix[center, center+3] = 0
    return matrix


# --- Visualization Functions ---
def setup_visualization(matrix, title="Matrix Visualization"):
    """Initializes the visualization of the matrix."""
    global fig, ax, im, pause_button, reset_button, wrap_check, iter_text, size_text, pattern_radio

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)  # Increased bottom margin for additional controls

    # Create colormap for visualization
    cmap = mcolors.ListedColormap(["white", "black"])
    bounds = [-0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Display the matrix
    im = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="grey", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    update_title()

    # Create control buttons - arranged in two rows
    # Size and iteration and pattern inputs
    ax_size = plt.axes([0.28, 0.12, 0.15, 0.05])
    size_text = TextBox(ax_size, "Size (n):", initial=str(matrix_size))
    size_text.on_submit(update_matrix_size)

    ax_iter = plt.axes([0.63, 0.12, 0.15, 0.05])
    iter_text = TextBox(ax_iter, "Generations:", initial=str(max_iterations))
    iter_text.on_submit(update_max_iterations)
    
    ax_pattern = plt.axes([0.8, 0.15, 0.2, 0.1])
    pattern_radio = RadioButtons(ax_pattern, ('Random', 'Glider'), active=0)
    pattern_radio.on_clicked(change_pattern)


    # Control buttons
    ax_reset = plt.axes([0.1, 0.05, 0.15, 0.05])
    reset_button = Button(ax_reset, "Reset")
    reset_button.on_clicked(reset_simulation)

    ax_pause = plt.axes([0.45, 0.05, 0.15, 0.05])
    pause_button = Button(ax_pause, "Start/Continue")
    pause_button.on_clicked(toggle_pause)
    
    # Add wraparound checkbox
    ax_wrap = plt.axes([0.8, 0.05, 0.15, 0.05])
    wrap_check = CheckButtons(ax_wrap, ["Wraparound"], [wraparound])
    wrap_check.on_clicked(toggle_wraparound)

    plt.ion()
    plt.show()
    return fig, ax, im


def update_visualization():
    """Updates the existing visualization with the current matrix data."""
    im.set_data(current_matrix)
    update_title()
    plt.draw()
    plt.pause(0.8)


# --- Simulation Logic ---
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
    new_matrix = matrix.copy()
    n, m = new_matrix.shape

    if n < 2 or m < 2:
        return new_matrix

    block_start = iteration % 2  # Alternates between 0 and 1

    for i in range(block_start, n, 2):
        for j in range(block_start, m, 2):
            if not wraparound and ((i - 1) < 0 or (j - 1) < 0):
                continue

            grid_cells = [
                (i, j),
                ((i - 1) % n, j),
                (i, (j - 1) % m),
                ((i - 1) % n, (j - 1) % m),
            ]
            count = sum(matrix[cell] for cell in grid_cells)

            if count in {0, 1, 4}:  # Flip all
                for cell in grid_cells:
                    new_matrix[cell] = 1 - matrix[cell]
                    
            elif count == 2:
                continue  # No change needed for 2 ones

            elif count == 3:  # Flip all and rotate
                for cell in grid_cells:
                    new_matrix[cell] = 1 - matrix[cell]
                # Rotate 180 degrees
                new_matrix[grid_cells[0]], new_matrix[grid_cells[3]] = (
                    new_matrix[grid_cells[3]],
                    new_matrix[grid_cells[0]],
                )
                new_matrix[grid_cells[1]], new_matrix[grid_cells[2]] = (
                    new_matrix[grid_cells[2]],
                    new_matrix[grid_cells[1]],
                )

    return new_matrix


# --- Simulation Runner ---
def run_simulation():
    """Runs the simulation loop, applying the algorithm logic and updating visualization."""
    global current_matrix, initial_matrix, reset_requested, current_iteration, stability

    while True:
        # Handle reset if requested
        if reset_requested:
            current_iteration = 0
            
            if initial_pattern == 'random':
                initial_matrix = create_random_binary_matrix(matrix_size, probability_one)
            else:
                initial_matrix = create_horizontal_glider(matrix_size)
                
            current_matrix = initial_matrix.copy()
            im.set_data(current_matrix)
            im.set_extent([-0.5, matrix_size - 0.5, matrix_size - 0.5, -0.5])
            reset_requested = False
            update_visualization()

        # Run simulation when not paused
        if not paused and current_iteration < max_iterations:
            old_matrix = current_matrix.copy()
            current_matrix = block_step(current_matrix, current_iteration + 1)
            current_iteration += 1
            stability = stability_calculator(old_matrix, current_matrix)
            update_visualization()
        else:
            plt.pause(0.1)

        # Check if window is closed
        if not plt.fignum_exists(fig.number):
            break


# --- Main Execution ---
if __name__ == "__main__":
    # Handle command line argument for wraparound
    if len(sys.argv) > 1 and sys.argv[1] == "wraparound":
        wraparound = True

    # Create initial matrix
    try:
        initial_matrix = create_random_binary_matrix(matrix_size, probability_one)
        current_matrix = initial_matrix.copy()

        # Setup visualization
        fig, ax, im = setup_visualization(current_matrix)

        # Start the simulation loop
        run_simulation()

    except ValueError as ve:
        print(f"Error setting up simulation: {ve}")
    finally:
        plt.ioff()
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.show()
