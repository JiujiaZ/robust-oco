import copy
import numpy as np
from scipy.special import erfi

class FakeCoinBetting:
    def __init__(self):
        self.w = 0
        self.grad_cumsum = 1
        self.grad_cumsquaresum = 1
    def play(self):
        return self.w

    def update(self, grad):
        self.grad_cumsum += grad
        self.grad_cumsquaresum += grad**2
        self.w = -1 * ( np.sign(self.grad_cumsum) /  np.sqrt(self.grad_cumsquaresum)) * np.exp( np.abs(self.grad_cumsum)/np.sqrt(self.grad_cumsquaresum))

class OSDBall:
    def __init__(self,input_dim=2):
        self.w = np.random.randn(input_dim)*0
        self.grad_cumsquaresum = 1

    def play(self):
        return self.w

    def update(self, grad):
        self.grad_cumsquaresum += np.linalg.norm(grad)**2
        lr = self.grad_cumsquaresum ** (-1/2)

        # update in grad:
        w = copy.deepcopy(self.w - lr * grad)
        # projection:
        self.w = w if np.linalg.norm(w) <= 1 else w / np.linalg.norm(w)

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

class CoinBetting():
    def __init__(self):
        self.w = 0
        self.grad_cumsum = 1
        self.wealth = 1
        self.t = 1
    def play(self):
        return self.w

    def update(self, grad, h):
        self.grad_cumsum += grad
        self.wealth -= grad * self.w
        self.w = -self.grad_cumsum / (self.t) * self.wealth

        self.t += 1




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
        # print(grad)
        # print(self.V, self.S)

    def _dphi_dy(self, x, y):
        def scaled_erfi(u):
            return np.sqrt( np.pi / 2) * erfi(u)

        coef = np.sqrt(4 * self.alpha * x)

        # print(y / coef)
        return self.epsilon * scaled_erfi(y / coef)

    def _d2Phi(self):
        # z ~ h^2, k ~ h
        # k = 2 * self.h
        # z = (12 * self.alpha + 4) / (2 * self.alpha - 1) * self.h**2
        k =  self.h
        z =  self.h ** 2
        # print(k, z)
        return self._dphi_dy(self.V + z + k * self.S, self.S)

class ZYCPMeta:
    def __init__(self,input_dim=2, epsilon = 1, alpha = 1, h = 1):
        self.learner_magnitude = ZYCPMagnitude(epsilon, alpha, h)
        self.learner_direction = OSDBall(input_dim)

        self.x = self.learner_direction.play()
        self.y_tilde = self.learner_magnitude.play()
        self.y = max(self.y_tilde,0)

        self.w = self.y * self.x

    def play(self):

        return self.w

    def update(self, grad, h):
        grad = grad if np.linalg.norm(grad) <= h else grad / np.linalg.norm(grad) * h
        l = np.dot(grad, self.x)
        l_tilde = l if l * self.y_tilde >= l * self.y else 0


        self.learner_direction.update(grad)
        self.learner_magnitude.update(l_tilde, h)


        self.x = self.learner_direction.play()
        self.y_tilde = self.learner_magnitude.play()
        self.y = max(self.y_tilde, 0)

        self.w = self.y * self.x

class NaiveCenteredMirrorDescent:

    def __init__(self,  input_dim, h = 1, epsilon = 1):
        self.V = 4 * h**2
        self.w = np.zeros(input_dim)
        self.theta = np.zeros(input_dim)
        self.epsilon = epsilon

    def play(self):
        return self.w

    def update(self, grad, h):
        self.theta -= grad
        self.V += np.linalg.norm(grad)**2
        alpha = self.epsilon * h / (np.sqrt(self.V) * np.log( self.V / h) ** 2)

        if np.linalg.norm(self.theta) <= 6* self.V / h:
            f = np.linalg.norm(self.theta)**2 / (36 * self.V)
        else:
            f = np.linalg.norm(self.theta) / (3 * h) - self.V / h**2
        self.w = alpha * self.theta / np.linalg.norm(self.theta) * (np.exp(f) - 1)



class CenteredMirrorDescent:
    def __init__(self, input_dim, T, alpha, c, epsilon = 1, k = 3, h =1):
        # k >= 3
        self.w = np.zeros(input_dim)
        self.epsilon = epsilon
        self.k = k
        self.h = h

        # only pad grad squares
        self.V = 0

        # for B_t
        self.B_inner = 0
        self.B_outer = 0
        self.B = 16

        self.alpha = self.epsilon / (np.sqrt(self.B) * np.log(self.B) ** 2)

        self.branch = 0 # change based on w

        # regularization:
        self.p = np.log(T)
        self.S = alpha ** self.p
        self.c = c

        self.t = 1


    def play(self):
        return self.w

    def update(self, grad, h):

        theta = self._nabla_psi(self.w, self.V+ self.h**2, self.alpha, self.h, self.branch) - grad
        A = copy.deepcopy(theta)

        # update next step
        self.V += np.linalg.norm(grad)**2
        self.h = h
        V_t = 4 * h ** 2 + self.V
        x_ast = self.alpha * ( np.exp(V_t / self.h**2 ) - 1 )
        if self.c == 0:
            threshold = 2 * self.k * V_t / h
        else:
            threshold = 2 * self.k * V_t / h + self.c * self.p / self.S ** (1 - 1/self.p) * x_ast ** (self.p - 1)
        # print(A, x_ast, self.S, threshold)
        if A <= threshold:
            self.branch = 0
        else:
            self.branch = 1

        self.B_inner += np.linalg.norm(grad) / h**2
        self.B_outer += copy.deepcopy(self.B_inner)
        self.B_outer += 4
        self.B = 4 * self.B_outer
        # print(grad, self.B_inner, self.B_outer, self.B, 16*self.t + 4 * self.t**2)
        self.alpha = self.epsilon / ( np.sqrt(self.B) * np.log(self.B) ** 2 )

        if self.branch == 0:
            a = 4 * self.k**2 * V_t
            b = self.alpha
            c = self.c * self.p / self.S ** (1 - 1 /self.p )
            p = self.p - 1
            w = self._solve_for_x_0(a, b, c, p, A)
        else:
            a = self.h * self.k
            b = self.alpha
            c = self.c * self.p / self.S ** (1 - 1 / self.p)
            d = self.k * V_t
            p = self.p - 1
            w = self._solve_for_x_1(a, b, c, d, p, A)
        # print(self.alpha, self.branch, w)
        self.w = w * A / np.linalg.norm(A)
        self.S += np.linalg.norm(self.w) ** self.p
        self.t += 1

    def _F(self, x, a):
        return np.log(1 + x / a)

    def _F_inv(self, y, a):
        return a * ( np.exp(y) - 1 )

    def _nabla_psi(self, x, V, alpha, h, branch = 0):

        if branch == 0:
            return 2 * self.k * np.sqrt( V * self._F(x, alpha) )
        else:
            return h * self._F(x, alpha)+ V / h

    def _solve_for_x_0(self, a, b, c, p, A, eps = 1e-8 ):

        # sqrt (a ln (1 + x /b)) + c x ^p = A
        def h(u):
            return np.sqrt(a * self._F(u, b)) + c * u ** p

        lb = 0

        if c == 0:
            lb = b * (np.exp(A ** 2 / a) - 1)
            ub = lb
        else:
            if A <= np.sqrt(a * np.log(2)) + c * b ** p:
                ub = b
            else:
                ub = ((A - np.sqrt(a * np.log(2))) / c) ** (1 / p)

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

    def _solve_for_x_1(self, a, b, c, d, p, A, eps=1e-8):

        # a ln (1 + x /b) + d / a  + c x ^p = A
        def h(u):
            return a * self._F(u, b) + d / a + c * u ** p

        lb = 0
        if c == 0:
            lb = b * (np.exp((A - d / a) / a) - 1)
            ub = lb
        else:
            if A <= a * np.log(2) + d / a + c * b**p :
                ub = b
            else:
                ub = ((A - a * np.log(2) - d/a ) / c)**(1/p)

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


class Filter:
    def __init__(self, h = 1, K = 0, gamma = 1):

        self.h_ini = h
        self.h_current = copy.deepcopy(self.h_ini)
        self.h_next = copy.deepcopy(self.h_ini)
        self.h_checkpoint = copy.deepcopy(self.h_ini)
        self.grad = 0
        self.K = K
        self.n = 0
        self.gamma = gamma

        self.denominator_sum = 1


    def play(self):
        return self.grad, self.denominator_sum, self.h_current, self.h_next

    def update(self, grad):
        self.denominator_sum += (self.h_next - self.h_current) / self.h_next

        self.h_current = copy.deepcopy(self.h_next)

        if np.linalg.norm(grad) > self.h_checkpoint:
            self.grad = (grad / np.linalg.norm(grad))*self.h_checkpoint
            self.n += 1
            self.h_next += 1/(self.K+1) * self.h_checkpoint

            if self.n == self.K:
                self.h_checkpoint = copy.deepcopy(self.h_next)
        else:
            self.grad = grad
    def reset(self):
        self.h_current = copy.deepcopy(self.h_ini)
        self.h_next = copy.deepcopy(self.h_ini)
        self.h_checkpoint = copy.deepcopy(self.h_ini)
        self.n = 0


class Tracker:
    def __init__(self, z=1, gamma = 1):
        self.z_ini = z
        self.z_current = copy.deepcopy(self.z_ini)
        self.z_next = copy.deepcopy(self.z_ini)
        self.gamma = gamma

        self.denominator_sum = 1

    def play(self):
        return  self.denominator_sum, self.z_current, self.z_next

    def update(self, z):
        self.denominator_sum += (self.z_next - self.z_current) / self.z_next
        self.z_current = copy.deepcopy(self.z_next)
        if np.linalg.norm(z) > self.z_current:
            self.z_next = 2 * self.z_next

    def reset(self):
        self.z_current = copy.deepcopy(self.z_ini)
        self.z_next = copy.deepcopy(self.z_ini)


class RobustMagnitudeWithoutG:
    def __init__(self, epsilon = 1, alpha = 1, h = 1, z = 1, c = 0, T = 10, K = 0):

        self.c = c
        self.alpha = epsilon / self.c if self.c > 0 else epsilon
        self.T = T
        self.K = K
        # self.denominator_sum = self.alpha ** np.log(self.T)
        self.w = 0
        self.y = 0
        self.hat_w = 0
        self.hat_y = 0

        self.learner_w = ZYCPMeta(input_dim=1, epsilon = epsilon, alpha = alpha, h = h)
        self.learner_y = ZYCPMeta(input_dim=1, epsilon=epsilon, alpha=alpha, h=h)
        self.filter = Filter(h = h, K = self.K, gamma = (self.K+1)/2 )
        self.tracker = Tracker(z = z, gamma = (self.K+1)/2 )

    def play(self):
        return self.w

    def update(self, grad):

        self.filter.update(grad)
        grad, denominator_sum_h, h_current, h_next = self.filter.play()
        self.tracker.update(self.w)
        denominator_sum_z, z_current, z_next = self.tracker.play()

        h_adjust = (h_next - h_current)/h_next
        alpha = self.filter.gamma * h_adjust / (denominator_sum_h + h_adjust)
        z_adjust = (z_next - z_current) / z_next
        beta = self.tracker.gamma * h_adjust / (denominator_sum_z + z_adjust)
        a = alpha + beta

        dual_norm = self._dual_norm(h_current, self.K+1, grad, a)
        grad_w, grad_y = self._subgrad(h_current, self.K+1, self.w, self.y, self.hat_w, self.hat_y)
        grad_w = (grad + grad_w * dual_norm)/2
        grad_y = (a + grad_y * dual_norm)/2

        self.learner_w.update(grad_w, h_next / 2)
        self.learner_y.update(grad_y, 1.5 * (self.K+1))

        self.hat_w, self.hat_y = self.learner_w.play(), self.learner_y.play()
        # check constrained
        if self.hat_y >= self.hat_w**2:
            self.w, self.y = self.hat_w, self.hat_y
        else:
            self.w, self.y = self._project(h_next, self.K+1, self.hat_w, self.hat_y)


        # composite_grad = grad+self._nabla_r(self.w)
        # composite_h = (1 + self.K * np.log(self.T)) * h
        # self.learner_rough.update(composite_grad, composite_h)
        # self.learner_fine.update(-composite_grad*self._nabla_r(self.w), composite_h**2)
        # # self.learner_rough.update(composite_grad, h)
        # # self.learner_fine.update(-composite_grad * self._nabla_r(self.w), h ** 2)
        #
        # x = self.learner_rough.play()
        # y = self.learner_fine.play()
        # self.w = self._optimism(x, y)
        # self.denominator_sum += np.abs(self.w) ** np.log(self.T)

    def _nabla_r(self, u):
        numerator = self.c * np.log(self.T) * (np.abs(u)**(np.log(self.T)-1)) * np.sign(u)
        denominator = (self.denominator_sum + np.abs(u) ** np.log(self.T))**(1 - 1 / np.log(self.T))
        return numerator / denominator


    def _norm(self, h, gamma, w, y):
        return w ** 2 * h ** 2 + y ** 2 * gamma ** 2

    def _dual_norm(self, h, gamma, w, y):

        return w**2 / h**2 + y**2 / gamma**2

    def _subgrad(self, h, gamma, w, y, hat_w, hat_y):
        # eval with reference of hat_w, hat_y

        bottom = self._norm(h, gamma, w-hat_w, y - hat_y)
        if bottom == 0:
            bottom+=1

        return h**2 * (hat_w-w) / bottom, gamma**2 * (hat_y-y) / bottom

    def _project(self, h, gamma, hat_w, hat_y):
        # through KKT -> third order polynomial Aw^3 + 0w^2 + Bw + C = 0
        A = 2 * gamma ** 2
        B = h ** 2 - 2 * gamma ** 2 * hat_y
        C = -h ** 2 * hat_w

        coefficients = [A, 0, B.item(), C.item()]
        roots = np.roots(coefficients)
        real_roots = [r.real for r in roots if np.isclose(r.imag, 0)]

        min_distance = np.inf
        closest_root = None
        for w in real_roots:
            y = w**2
            distance = self._norm(h, gamma, w - hat_w, y - hat_y)

            if distance<min_distance:
                min_distance = distance
                closest_root = (w, y)

        return closest_root


class RobustMetaWithoutG:
    def __init__(self, input_dim=2, epsilon=1, h=1, z = 1, c=0, T=10, K=0):
        self.learner_magnitude = RobustMagnitudeWithoutG(epsilon=epsilon, h=h, z = z,  c=c, T=T, K=K)
        self.learner_direction = OSDBall(input_dim)

        self.w = self.learner_magnitude.play() * self.learner_direction.play()

    def play(self):
        return self.w

    def update(self, grad, h):
        self.learner_magnitude.update(np.dot(grad, self.learner_direction.play()))
        self.learner_direction.update(grad)
        self.w = self.learner_magnitude.play() * self.learner_direction.play()

