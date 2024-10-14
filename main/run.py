from src.learners import *
from src.datasets import *
from src.adversaries import *
import matplotlib.pyplot as plt


def train(num_samples=50, input_dim=2, G=None, K = 0, name = 'cluster_split', learner_methods = ['fake_cb'], loss_fn = 'absolute'):

    # initialize:
    dataset = DatasetGenerator(num_samples=num_samples, name = name)
    adversary = Adversary(loss_fn=loss_fn)

    # results holder for each method - iterates
    res = {method: dict() for method in learner_methods}

    # iterate over leaner_names:
    for method in learner_methods:
        if method == 'fake_cb':
            learner = FakeCoinMeta(input_dim=input_dim, h = G)
        elif method == 'zycp':
            learner = ZYCPMeta(input_dim=input_dim, epsilon = 1, alpha = 1, h = G)
        elif method == 'robust':
            learner = RobustMetaWithG(input_dim=input_dim, epsilon=1, h=G, c=K*G, T=len(dataset.dataset), K=K)
        else:
            raise ValueError(f'method: {method} , not implemented')

        iterates = list()
        loss_val = list()
        lin_loss_val = list()
        for epoch in range(10):
            # corrupted rounds:
            corrupted_indices = corruption(len(dataset.dataset), 10)
            for idx, (x, y) in enumerate(dataset):

                w = learner.play()

                grad, _, _ = adversary.feedback(x, w, y, w, y)
                if idx in set(corrupted_indices):
                    x_corrupted = x + grad
                else:
                    x_corrupted = x

                grad, loss, lin_loss = adversary.feedback(w, x, y, x_corrupted, y)

                learner.update(grad, G)

                iterates.append(w)
                loss_val.append(loss)
                lin_loss_val.append(lin_loss)
        res[method]['iter'] = np.array(iterates)
        res[method]['loss'] = np.array(loss_val)
        res[method]['lin_loss'] = np.array(lin_loss_val)

    return dataset, corrupted_indices, res

def visualize_results(dataset, res, corrupted_indices):

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
    for method in res.keys():
        # simple last iterate
        learner_weights = res[method]['iter'][-1]
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
    loss_fn = 'square'
    # learner_methods = [ 'robust']
    dataset, corrupted_indices, res = train(num_samples=num_samples, input_dim=input_dim,
                                            G= G, K = K, name = name, learner_methods = learner_methods,
                                            loss_fn = loss_fn)

    # Visualize the results
    # visualize_results(dataset, res, corrupted_indices)

    return res

# Run the experiment
if __name__ == '__main__':
    res = run(num_samples=500, input_dim=1, G=10, K=1, name = 'mean')

    plt.figure(figsize=(8, 6))
    for method in res.keys():
        w = res[method]['loss']
        # w_mag = np.linalg.norm(w, axis=1)
        w_mag = np.cumsum(w)
        # w_mag = w_mag / np.arange(1, len(w_mag) + 1)
        plt.plot(w_mag, label=method)

    plt.legend()
    # plt.ylim([0, 2])
    plt.show()

    plt.figure(figsize=(8, 6))
    for method in res.keys():
        w = res[method]['iter']
        w_mag = np.linalg.norm(w, axis=1)
        plt.plot(w_mag, label=method)

    plt.legend()
    # plt.ylim([0, 1])
    plt.show()


