# Import libraries
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR
from sklearn.metrics import accuracy_score

# Load data
def load_mnist():
    train_images = idx2numpy.convert_from_file('train-images-idx3-ubyte')/255
    train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
    test_images = idx2numpy.convert_from_file('t10k-images-idx3-ubyte')/255
    test_labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
    return train_images, train_labels, test_images, test_labels

def gauss_noise(dataset, sigma=0.2, mu=0):
    noise = np.random.normal(mu, sigma, (dataset.shape))
    dataset = dataset + noise
    return dataset

# Flatten data. ie convert Nxdxd image to Nx(d^2)x1 vector
def n_vectorize(dataset):
    flat_data = dataset.reshape(dataset.shape[0], -1)
    #print(np.shape(flat_data))
    return flat_data

# mean vector
def mean_vector(dataset):
    meanVector = (np.mean(dataset, axis = 0))
    #print(np.shape(meanVector))
    return meanVector


def pca(train_images_vectorize, energy_boundary):
# Eigenvectors of dataset
    train_images_vectorize = train_images_vectorize - mean_vector(train_images_vectorize)
    covariance_matrix = np.matmul(train_images_vectorize.T, train_images_vectorize)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    print("eigenvalues shape", np.shape(eigenvalues), "eigenvectors shape", np.shape(eigenvectors))

    index_for_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index_for_sort]
    eigenvectors = eigenvectors[:,index_for_sort]   

    last_important_eigen_value_index = eigen_energy(eigenvalues, energy_boundary)

    # Encode Training Data

    encoded_data =  np.matmul(train_images_vectorize, eigenvectors[:,:last_important_eigen_value_index])

    print("encoded data shape ", np.shape(encoded_data))

    return encoded_data, eigenvectors[:,:last_important_eigen_value_index]
    #return eigenvectors

def pca_reconstruct(encoded_data, eigenvectors):
    return np.matmul(encoded_data, eigenvectors)

def eigen_energy(eigenvalues, energy_boundary):
    total_energy = eigenvalues.sum()
    ratio_val = 0
    i = 0

    while (i<np.shape(eigenvalues)[0]) and ratio_val<energy_boundary:
        ratio_val += eigenvalues[i]/total_energy
        i+=1

    print("Ratio", ratio_val)

    return i

train_images, train_labels, test_images, test_labels = load_mnist()
train_images_vectorize = n_vectorize(train_images)
train_images_vectorize = train_images_vectorize - mean_vector(train_images_vectorize)


for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_images_vectorize[i].reshape((28,28)), cmap=plt.get_cmap('gray'))

plt.show()

train_images_vectorize_gauss_noise = gauss_noise(train_images_vectorize)
train_images_vectorize_gauss_noise = train_images_vectorize_gauss_noise - mean_vector(train_images_vectorize_gauss_noise)

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(train_images_vectorize_gauss_noise[i].reshape((28,28)), cmap=plt.get_cmap('gray'))
    
plt.show()


encoded_data, eigenvectors = pca(train_images_vectorize_gauss_noise, 0.80)
reconstructed_data = pca_reconstruct(encoded_data, eigenvectors.T)
print(np.shape(reconstructed_data))

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(reconstructed_data[i].reshape((28,28)), cmap=plt.get_cmap('gray'))
    
plt.show()

# first_train_image = train_images_vectorize[3]
# pixels = first_train_image.reshape((28, 28))
# plt.imshow(pixels, cmap='gray')
# plt.show()
# plt.close