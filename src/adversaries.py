import numpy as np


def corruption(labels, K):
    n = len(labels)
    corrupted_indices = np.random.choice(n, K, replace=False)
    # corrupted_indices = np.arange(n-K,n)
    corrupted_labels = labels.copy()
    corrupted_labels[corrupted_indices] = corrupted_labels[corrupted_indices] * -1

    return corrupted_labels, corrupted_indices

class Adversarial:
    def __init__(self, G = None):
        self.clip = np.inf if G is None else G

    def feedback(self, w, x, y, corrupt = False):
        error = y - np.dot(w,x)
        grad = -x * error

        if corrupt:
            grad = -1 * grad
        # truncate
        return grad if np.linalg.norm(grad) <= self.clip else grad / np.linalg.norm(grad)



