import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def load_dataset(dataset, num_dimensions=2):
    X = dataset.data[:, :num_dimensions]  # Use only the first two features
    y = dataset.target
    return X, y

# train a k-NN classifier with k neighbors
def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def test_knn(knn, X_test, y_test):
    return knn.score(X_test, y_test)

# perform PCA on the dataset for any number of dimensions
def perform_pca(X, num_dimensions=2):
    pca = PCA(n_components=num_dimensions)
    X_pca = pca.fit_transform(X)
    return X_pca


def plot_decision_boundaries(X, y, knn, dataset_name, feature_labels, num_dimensions=2, class_names=None):
    if num_dimensions == 1:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        xx = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
        Z = knn.predict(xx)
        
        plt.plot(xx, Z, label=f'{dataset_name}, KNN=3')
        for class_label in np.unique(y):
            plt.scatter(X[y == class_label, 0], y[y == class_label], label=class_names[class_label])
        plt.xlabel(feature_labels[0])
        plt.ylabel('Target')
        plt.title(f'{dataset_name} - 1D Decision Boundaries')
        
    elif num_dimensions == 2:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.3)
        for class_label in np.unique(y):
            plt.scatter(X[y == class_label, 0], X[y == class_label, 1], label=class_names[class_label])
        plt.xlabel(feature_labels[0])
        plt.ylabel(feature_labels[1])
        plt.title(f'{dataset_name} - 2D Decision Boundaries')
        
    elif num_dimensions == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for class_label in np.unique(y):
            ax.scatter(X[y == class_label, 0], X[y == class_label, 1], X[y == class_label, 2], label=class_names[class_label])
        ax.set_xlabel(feature_labels[0])
        ax.set_ylabel(feature_labels[1])
        ax.set_zlabel(feature_labels[2])
        ax.set_title(f'{dataset_name} - 3D Scatter Plot')
    plt.legend()
    plt.show()

def plot_test_decisions(X, y, knn, dataset_name, feature_labels, num_dimensions=2):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        
    y_pred = knn.predict(X)
    correct_indices = y_pred == y
    incorrect_indices = y_pred != y

    plt.scatter(X[correct_indices, 0], X[correct_indices, 1], c='green', marker='o', label='Correct')
    plt.scatter(X[incorrect_indices, 0], X[incorrect_indices, 1], c='red', marker='x', label='Incorrect')

    plt.xlabel(feature_labels[0])
    plt.ylabel(feature_labels[1])
    plt.title(f'{dataset_name} - 2D Decision Boundaries with Correct/Incorrect Classifications')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load data
    iris = load_iris()
    num_dim = 2
    num_pca_components = 2
    num_neighbors = 10
    X, y = load_dataset(iris, num_dimensions=num_dim)
    feature_labels = iris.feature_names[:num_dim]
    
    # Non-PCA Code
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = train_knn(X_train, y_train, k=num_neighbors)
    accuracy = test_knn(knn, X_test, y_test)
    print(f'Accuracy: {accuracy}')
    plot_decision_boundaries(X, y, knn, dataset_name='Iris', feature_labels=feature_labels, num_dimensions=num_dim, class_names=iris.target_names)
    # plot_test_decisions(X_test, y_test, knn, dataset_name='Iris', feature_labels=feature_labels, num_dimensions=2)
    
    # PCA Code
    # X_pca = perform_pca(X, num_dimensions=num_pca_components)
    # X_train, X_test, y_train, y_test = train_test_split(X_pca, y, random_state=0)
    # knn = train_knn(X_train, y_train, k=num_neighbors)
    # accuracy = test_knn(knn, X_test, y_test)
    # print(f'Accuracy: {accuracy}')
    # plot_decision_boundaries(X_pca, y, knn, dataset_name='Iris', feature_labels=["Principal Component 1", "Principal Component 2"], num_dimensions=num_pca_components, class_names=iris.target_names)
    # plot_test_decisions(X_test, y_test, knn, dataset_name='Iris', feature_labels=["Principal Component 1", "Principal Component 2"], num_dimensions=2)
