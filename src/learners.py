import copy
import numpy as np

class FakeCoinBetting:
    def __init__(self):
        self.w = 0
        self.grad_cumsum = 0
        self.grad_cumsquaresum = 0
    def play(self):
        return self.w

    def update(self, grad):
        self.grad_cumsum += grad
        self.grad_cumsquaresum += grad**2
        self.w = -1 * ( np.sign(self.grad_cumsum) / np.sqrt(self.grad_cumsquaresum)) * np.exp( np.abs(self.grad_cumsum)/np.sqrt(self.grad_cumsquaresum) )

class OSDBall:
    def __init__(self,input_dim=2):
        self.w = np.random.randn(input_dim)
        self.grad_cumsquaresum = 0

    def play(self):
        return self.w

    def update(self, grad):
        self.grad_cumsquaresum += np.linalg.norm(grad)**2
        lr = self.grad_cumsquaresum ** (-1/2)

        # update in grad:
        self.w -= lr * grad
        # projection:
        self.w = self.w if np.linalg.norm(self.w) <= 1 else self.w / np.linalg.norm(self.w)

class ParameterFree():
    def __init__(self,input_dim=2):
        self.learner_magnitude = FakeCoinBetting()
        self.learner_direction = OSDBall(input_dim)

        self.w = self.learner_magnitude.play() * self.learner_direction.play()
    def play(self):

        return self.w

    def update(self, grad):
        self.learner_magnitude.update(np.dot(grad, self.learner_direction.play()))
        self.learner_direction.update(grad)
        self.w = self.learner_magnitude.play() * self.learner_direction.play()






#
# class CenteredMirrorDescent:
#     def __init__(self, input_dim):
#         self.w_1 = np.random.randn(input_dim)
#         self.w = copy.deepcopy(self.w_1)
#         self.t = 1
#     def play(self, x):
#         return self.w
#
#     def update(self, grad):
#         Delta = self._bregman(w)
#
#         V_t = 4 * h_t ** 2 + np.linalg.norm(self.weights) ** 2
#         alpha_t = self.learning_rate / (np.sqrt(B_t) * np.log2(B_t))
#         mirror_gradient = gradient + alpha_t * self._gradient_regularizer(self.weights, V_t)
#         self.weights -= alpha_t * mirror_gradient
#
#     def _regularizer(self, x):
#         return 1
#     def _grad_regularizer(self, x):
#         return 1
#     def _penalty(self):
#         return 1
#     def _bregman(self, x, y):
#         return self._regularizer(x) - self._regularizer(y) - np.dot( self._grad_regularizer(y), x-y)
