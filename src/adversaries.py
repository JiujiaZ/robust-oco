import numpy as np


def corruption(N, K):
    # n = len(labels)
    corrupted_indices = np.random.choice(N, K, replace=False)
    return corrupted_indices
    # corrupted_indices = np.arange(n-K,n)
    # corrupted_labels = labels.copy()
    # corrupted_labels[corrupted_indices] = corrupted_labels[corrupted_indices] * -100

    # return corrupted_labels, corrupted_indices


class Adversary:
    def __init__(self, loss_fn='square'):

        self.loss_fn = loss_fn

        if loss_fn == 'square':
            self.compute_grad_loss = self._squared_loss
        elif loss_fn == 'absolute':
            self.compute_grad_loss = self._absolute_loss
        elif loss_fn == 'hinge':
            self.compute_grad_loss = self._hinge_loss
        else:
            raise ValueError(f"loss function: {loss_fn} not implemented")

    def feedback(self, w, x, y, x_corruted, y_corruted):

        grad, loss, lin_loss = self.compute_grad_loss(w, x, y, x_corruted, y_corruted)

        # return grad if np.linalg.norm(grad) <= self.clip else grad / np.linalg.norm(grad)

        # grad for corrupted feedback, later twos no corruption for benchmarking
        return grad, loss, lin_loss

    def _squared_loss(self, w, x, y, x_corruted, y_corruted):

        corrupted_grad = x_corruted * (np.dot(w, x_corruted) - y_corruted)

        loss = 0.5 * ((np.dot(w, x) - y)**2)
        lin_loss = w * (np.dot(w, x) - y)

        return corrupted_grad, loss, lin_loss

    def _hinge_loss(self, w, x, y,x_corruted, y_corruted):

        error_corrupted = 1 - y_corruted * np.dot(w, x_corruted)
        corrupted_grad = -y_corruted * x if error_corrupted > 0 else np.zeros_like(x)

        error = 1 - y * np.dot(w, x)
        loss = max(np.array([0]), error).item()
        grad = -y * x if error > 0 else np.zeros_like(x)
        lin_loss = np.dot(grad, w)

        return corrupted_grad, loss, lin_loss

    def _absolute_loss(self, w, x, y, x_corruted, y_corruted):

        error_corrupted = y_corruted - np.dot(w, x_corruted)
        corrupted_grad = -x_corruted * np.sign(error_corrupted)

        error = y - np.dot(w, x)
        grad = -x * np.sign(error)
        loss = np.abs(error)
        lin_loss = np.dot(grad, w)

        return corrupted_grad, loss, lin_loss





