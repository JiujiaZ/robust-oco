import numpy as np

class DatasetGenerator:
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        self.dataset = self.generate()
        self.index = 0

    def generate(self):
        # Generate a simple dataset with two clusters
        x_class_1 = np.random.randn(self.num_samples, 2) + np.array([2, 2])
        y_class_1 = np.ones(self.num_samples)

        x_class_2 = np.random.randn(self.num_samples, 2) + np.array([-2, -2])
        y_class_2 = -1 * np.ones(self.num_samples)

        X = np.vstack((x_class_1, x_class_2))
        Y = np.concatenate((y_class_1, y_class_2))
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