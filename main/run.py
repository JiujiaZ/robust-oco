from src.learners import *
from src.datasets import *
from src.adversaries import *
import matplotlib.pyplot as plt


def train(num_samples=50, input_dim=2, G=None, K = 0, name = 'cluster_split', learner_methods = ['fake_cb']):

    # initialize:
    dataset = DatasetGenerator(num_samples=num_samples, name = name)
    adversarial = Adversarial(G=G)

    # Corrupt K labels
    labels = np.array([y for _, y in dataset])
    corrupted_labels, corrupted_indices = corruption(labels, K)

    # results holder for each method - iterates
    res = {method: list() for method in learner_methods}

    # iterate over leaner_names:
    for method in learner_methods:
        if method == 'fake_cb':
            learner = FakeCoinMeta(input_dim=input_dim)
        elif method == 'zycp':
            learner = ZYCPMeta(input_dim=input_dim, epsilon = 1, alpha = 1, h = G)
        elif method == 'robust':
            learner = RobustMetaWithG(input_dim=input_dim, epsilon=1, h=G, c=K*G, T=len(dataset.dataset), K=K)
        else:
            raise ValueError(f'method: {method} , not implemented')

        iterates = list()
        for idx, (x, y) in enumerate(dataset):
            y_corrupted = corrupted_labels[idx]  # Use the corrupted label
            w = learner.play()
            grad = adversarial.feedback(w, x, y_corrupted)
            # corrupt = True if idx in corrupted_indices else False
            # grad = adversarial.feedback(w, x, y, corrupt)
            if method != 'fake_cb':
               learner.update(grad, G)
            else:
               learner.update(grad)
            iterates.append(w)
        res[method] = np.array(iterates)

    return dataset, corrupted_indices, res

def visualize_results(dataset, iterates, corrupted_indices):

    x_vals, y_vals = zip(*dataset.dataset)
    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    plt.figure(figsize=(8, 6))
    plt.scatter(x_vals[y_vals == 1][:, 0], x_vals[y_vals == 1][:, 1], color='blue', label='Class 1')
    plt.scatter(x_vals[y_vals == -1][:, 0], x_vals[y_vals == -1][:, 1], color='red', label='Class -1')

    # highlight corrupted labels
    plt.scatter(x_vals[corrupted_indices, 0], x_vals[corrupted_indices, 1],
                edgecolor='k', facecolor='none', s=100, label='Corrupted Labels')
    # Plot generators boundary
    x_boundary = np.linspace(min(x_vals[:, 0]), max(x_vals[:, 0]), 100)
    if dataset.linear_boundary is not None:
        y_boundary = -(dataset.linear_boundary[0] / dataset.linear_boundary[1]) * x_boundary
        plt.plot(x_boundary, y_boundary, color='black', linestyle='--', label='Ground Truth')


    # Plot learners decision boundary
    for method in iterates.keys():
        # simple last iterate
        learner_weights = iterates[method][-1]
        y_boundary = -(learner_weights[0] / learner_weights[1]) * x_boundary
        plt.plot(x_boundary, y_boundary, linestyle='--', label=method)


    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(f'# examples: {len(y_vals)} / # corruption: {len(corrupted_indices)}')
    plt.grid(True)
    plt.show()


def run(num_samples=50, input_dim=2, G=None, K=5, name = 'cluster_split'):

    learner_methods = ['fake_cb', 'zycp', 'robust']
    dataset, corrupted_indices, res = train(num_samples=num_samples, input_dim=input_dim,
                                            G= G, K = K, name = name, learner_methods = learner_methods)

    # Visualize the results
    visualize_results(dataset, res, corrupted_indices)

    return res

# Run the experiment
if __name__ == '__main__':
    res = run(num_samples=500, input_dim=2, G=2, K=30, name = 'cluster_split')

