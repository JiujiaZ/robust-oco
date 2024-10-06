from src.learners import *
from src.datasets import *
from src.adversarial import *
import matplotlib.pyplot as plt


def initialization(num_samples=50, input_dim=2, G=None):

    dataset = DatasetGenerator(num_samples=num_samples)
    learner = ParameterFree(input_dim=input_dim)
    adversarial = Adversarial(G=G)
    return dataset, learner, adversarial


def corrupt_and_train(dataset, learner, adversarial, K=5):

    # Corrupt K labels
    labels = np.array([y for _, y in dataset])
    corrupted_labels, corrupted_indices = corruption(labels, K)

    for idx, (x, y) in enumerate(dataset):
        y_corrupted = corrupted_labels[idx]  # Use the corrupted label
        w = learner.play()
        grad = adversarial.feedback(w, x, y_corrupted)
        learner.update(grad)

    return corrupted_indices, learner.w


def visualize_results(dataset, learner_weights, corrupted_indices):

    x_vals, y_vals = zip(*dataset.dataset)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals[y_vals == 1][:, 0], x_vals[y_vals == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(x_vals[y_vals == -1][:, 0], x_vals[y_vals == -1][:, 1], color='red', label='Class -1')

    # Step 4: Highlight corrupted labels
    plt.scatter(x_vals[corrupted_indices, 0], x_vals[corrupted_indices, 1],
                edgecolor='k', facecolor='none', s=100, label='Corrupted Labels')

    # Plot decision boundary
    x_boundary = np.linspace(min(x_vals[:, 0]), max(x_vals[:, 0]), 100)
    y_boundary = -(learner_weights[0] / learner_weights[1]) * x_boundary
    plt.plot(x_boundary, y_boundary, color='green', linestyle='--', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('Visualization of Binary Classification and Decision Boundary with Corrupted Labels')
    plt.grid(True)
    plt.show()


def run_experiment(num_samples=50, input_dim=2, G=None, K=5):
    """
    Main function to run the entire experiment.
    """
    # Initialize experiment components
    dataset, learner, adversarial = initialization(num_samples, input_dim, G)

    # Corrupt labels and train
    corrupted_indices, final_weights = corrupt_and_train(dataset, learner, adversarial, K)

    # Visualize the results
    visualize_results(dataset, final_weights, corrupted_indices)


# Run the experiment
if __name__ == '__main__':
    run_experiment(num_samples=500, input_dim=2, G=None, K=100)