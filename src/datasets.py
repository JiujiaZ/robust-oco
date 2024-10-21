import numpy as np

class DatasetGenerator:
    def __init__(self, num_samples=100, name = 'two_cluster'):
        self.num_samples = num_samples
        self.linear_boundary = None
        self.dataset = self.generate(name)
        self.index = 0

    def generate(self, name):

        if name == 'two_cluster':
            # Generate a simple dataset with two clusters
            x_class_1 = np.random.randn(self.num_samples, 2) + np.array([2, 2])
            y_class_1 = np.ones(self.num_samples)

            x_class_2 = np.random.randn(self.num_samples, 2) + np.array([-2, -2])
            y_class_2 = -1 * np.ones(self.num_samples)

            X = np.vstack((x_class_1, x_class_2))
            Y = np.concatenate((y_class_1, y_class_2))

        elif name == 'cluster_split':
            X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=self.num_samples*2)
            self.linear_boundary = np.array([1, -1])
            Y = np.where(np.dot(X, self.linear_boundary) >= 0, 1, -1)

        elif name == 'mean':
            # learn mean estimation problem
            # suspect robust olo can only produce small iterate

            Y = np.random.normal(loc=10, scale=1, size=self.num_samples*2)
            # Y = X.mean() * np.ones_like(X)
            X = np.ones_like(Y)

        elif name == 'greedy':
            # l(w) = |y-wx| = |y-1|
            Y = np.ones(self.num_samples*2)
            X = np.ones_like(Y)


        dataset = list(zip(X, Y))
        np.random.shuffle(dataset)
        return dataset


    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            self.index = 0
            raise StopIteration
        data = self.dataset[self.index]
        self.index += 1
        return data