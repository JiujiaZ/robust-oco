import copy
import numpy as np
from scipy.special import erfi

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
        self.grad_cumsquaresum = 1

    def play(self):
        return self.w

    def update(self, grad):
        self.grad_cumsquaresum += np.linalg.norm(grad)**2
        lr = self.grad_cumsquaresum ** (-1/2)

        # update in grad:
        self.w -= lr * grad
        # projection:
        self.w = self.w if np.linalg.norm(self.w) <= 1 else self.w / np.linalg.norm(self.w)

class FakeCoinMeta():
    def __init__(self,input_dim=2, h = 1):
        self.learner_magnitude = FakeCoinBetting()
        self.learner_direction = OSDBall(input_dim)

        self.w = self.learner_magnitude.play() * self.learner_direction.play()
    def play(self):

        return self.w

    def update(self, grad, h):

        grad = grad if np.linalg.norm(grad) <= h else grad / np.linalg.norm(grad) * h

        self.learner_magnitude.update(np.dot(grad, self.learner_direction.play()))
        self.learner_direction.update(grad)
        self.w = self.learner_magnitude.play() * self.learner_direction.play()

class ZYCPMagnitude:
    def __init__(self, epsilon = 1, alpha = 1, h = 1):
        self.alpha = alpha
        self.epsilon = epsilon
        self.h = h

        self.w = 0
        self.V = 0
        self.S = 0

    def play(self):

        return self._d2Phi()

    def update(self, grad, h):
        self.V += grad**2
        self.S -= grad
        self.h = h

    def _dphi_dy(self, x, y):
        def scaled_erfi(u):
            return np.sqrt( np.pi / 2) * erfi(u)

        coef = np.sqrt(4 * self.alpha * x)
        return self.epsilon * scaled_erfi(y / coef)

    def _d2Phi(self):
        # z ~ h^2, k ~ h
        k = 2 * self.h
        z = (12 * self.alpha + 4) / (2 * self.alpha - 1) * self.h**2
        return self._dphi_dy(self.V + z + k * self.S, self.S)

class ZYCPMeta:
    def __init__(self,input_dim=2, epsilon = 1, alpha = 1, h = 1):
        self.learner_magnitude = ZYCPMagnitude(epsilon, alpha, h)
        self.learner_direction = OSDBall(input_dim)

        self.w = max(self.learner_magnitude.play(),0) * self.learner_direction.play()

    def play(self):

        return self.w

    def update(self, grad, h):
        y_tilde = self.learner_magnitude.play()
        y = max(self.learner_magnitude.play(), 0)
        x = self.learner_direction.play()


        grad = grad if np.linalg.norm(grad) <= h else grad / np.linalg.norm(grad) * h
        self.learner_direction.update(grad)
        l = np.dot(grad, x)
        l_tilde = l if l * y_tilde >= l * y else 0
        self.learner_magnitude.update(l_tilde, h)

        self.w = max(self.learner_magnitude.play(), 0) * self.learner_direction.play()

class RobustMagnitudeWithG:
    def __init__(self, epsilon = 1, alpha = 1, h = 1, c = 0, T = 10, K = 0):

        self.c = c
        self.alpha = epsilon / self.c if self.c > 0 else epsilon
        # self.alpha = epsilon * 10
        self.T = T
        self.K = K
        self.denominator_sum = self.alpha ** np.log(self.T)
        self.w = 0

        self.learner_rough = ZYCPMeta(input_dim=1, epsilon = epsilon, alpha = alpha, h = h)
        self.learner_fine = ZYCPMeta(input_dim=1, epsilon=epsilon, alpha=alpha, h=h)


    def play(self):
        return self.w

    def update(self, grad, h):

        grad = grad if np.abs(grad) <= h else np.sign(grad) * h

        composite_grad = grad+self._nabla_r(self.w)
        composite_h = (1 + self.K * np.log(self.T)) * h
        self.learner_rough.update(composite_grad, composite_h)
        self.learner_fine.update(-composite_grad*self._nabla_r(self.w), composite_h**2)
        # self.learner_rough.update(composite_grad, h)
        # self.learner_fine.update(-composite_grad * self._nabla_r(self.w), h ** 2)

        x = self.learner_rough.play()
        y = self.learner_fine.play()
        self.w = self._optimism(x, y)
        self.denominator_sum += np.abs(self.w) ** np.log(self.T)

    def _nabla_r(self, u):
        numerator = self.c * np.log(self.T) * (np.abs(u)**(np.log(self.T)-1)) * np.sign(u)
        denominator = (self.denominator_sum + np.abs(u) ** np.log(self.T))**(1 - 1 / np.log(self.T))
        return numerator / denominator

    def _optimism(self, x, y, eps = 1e-8 ):
        def h(u):
            return u - x + y * self._nabla_r(u)

        if x >= 0:
            lb = x - y * self.c * np.log(self.T)
            ub = x
        else:
            lb = x
            ub = x + y * self.c * np.log(self.T)

        while (ub - lb) / 2 > eps:
            u = (ub + lb) / 2

            if h(u) == 0:
                return u
            elif h(lb) * h(u) < 0:
                ub = u
            else:
                lb = u

        u = (ub + lb) / 2

        # print('opt:', u, h(u))

        return u


class RobustMetaWithG:
    def __init__(self, input_dim=2, epsilon=1, h=1, c=0, T=10, K=0):

        self.learner_magnitude = RobustMagnitudeWithG(epsilon=epsilon, h=h, c=c, T=T, K=K)
        self.learner_direction = OSDBall(input_dim)

        self.w = self.learner_magnitude.play() * self.learner_direction.play()

    def play(self):
        return self.w

    def update(self, grad, h):
        self.learner_magnitude.update(np.dot(grad, self.learner_direction.play()), h)
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
