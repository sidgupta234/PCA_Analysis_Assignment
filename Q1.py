# Import libraries
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def load_mnist():
    train_images = (idx2numpy.convert_from_file('train-images-idx3-ubyte'))/255
    train_labels = idx2numpy.convert_from_file('train-labels-idx1-ubyte')
    test_images = (idx2numpy.convert_from_file('t10k-images-idx3-ubyte'))/255
    test_labels = idx2numpy.convert_from_file('t10k-labels-idx1-ubyte')
    
    return train_images, train_labels, test_images, test_labels

# Flatten data. ie convert Nxdxd image to Nx(d^2)x1 vector

def n_vectorize(dataset):
    flat_data = dataset.reshape(dataset.shape[0], -1)
    #print(np.shape(flat_data))
    return flat_data

# Compute mean
def mean_vector(dataset):
    meanVector = (np.mean(dataset, axis = 0))
    #print(np.shape(meanVector))
    return meanVector

# Compute global mean
def global_mean(dataset):
    globalMean = np.mean(dataset)
    return globalMean

def eigen_energy(eigenvalues, energy_boundary):
    eigenvalues = np.real(eigenvalues)
    total_energy = eigenvalues.sum()
    ratio_val = 0
    i = 0

    while (i<np.shape(eigenvalues)[0]) and ratio_val<energy_boundary:
        ratio_val += eigenvalues[i]/total_energy
        i+=1

    print("Ratio", ratio_val)

    return i

def pca_ndim(train_images_vectorize, pca_ndim):
    # Eigenvectors of dataset
    train_images_vectorize = train_images_vectorize - mean_vector(train_images_vectorize)
    covariance_matrix = np.matmul(train_images_vectorize.T, train_images_vectorize)
    #print(np.shape(train_images_vectorize[0]))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #print("eigenvectors", eigenvectors)
    index_for_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index_for_sort]
    eigenvectors = eigenvectors[:,index_for_sort]   
    #print("eigenvalues ",(eigenvalues), "eigenvectors shape", (eigenvectors[:,:pca_ndim].shape))
    
    last_important_eigen_value_index = pca_ndim

    # Encode Training Data
    
    encoded_data = np.matmul(train_images_vectorize, eigenvectors[:,:last_important_eigen_value_index])
    #print(eigenvalues[0:10])
    
    #print("Shape of encoded data", np.shape(encoded_data))

    return encoded_data
    #return eigenvectors

#Ans 2
def pca(train_images_vectorize, energy_boundary):
    # Eigenvectors of dataset
    train_images_vectorize = train_images_vectorize - mean_vector(train_images_vectorize)
    covariance_matrix = np.matmul(train_images_vectorize.T, train_images_vectorize)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    index_for_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index_for_sort]
    eigenvectors = eigenvectors[:,index_for_sort]   
    
    last_important_eigen_value_index = eigen_energy(eigenvalues, energy_boundary)

    print("eigenvalues shape", np.shape(eigenvalues), "eigenvectors shape", np.shape(eigenvectors[:,:last_important_eigen_value_index]))
    
    # Encode Training Data

    encoded_data =  np.matmul(train_images_vectorize, eigenvectors[:,:last_important_eigen_value_index])
    
    print("encoded data shape ", np.shape(encoded_data))

    return encoded_data, eigenvectors[:,:last_important_eigen_value_index]
    #return eigenvectors

def within_class_covariance(train_images_vectorize, train_labels):
    dict_fda = {}
    covar_i = []
    
    for i, j in zip(train_labels, train_images_vectorize):
        if(i in dict_fda):
            dict_fda[i].append(j)
            
        else:
            dict_fda[i] = [j]
    
    for label in np.unique(train_labels):
        dict_fda[label] = np.asarray(dict_fda[label])
        dict_fda[label] -= np.mean(dict_fda[label], axis=0)
        
    for i in dict_fda:
        covar_i.append(np.matmul(dict_fda[i].T, dict_fda[i]))

    print(np.shape((np.sum(covar_i, axis=0))))
    return np.sum(covar_i, axis=0)

    # cov_class_i[i] = cov_class_i[i] - mean_vector(cov_class_i[i])
# cov_class_i[i] = np.matmul(cov_class_i[i].T, cov_class_i[i])

#     cov_class_i = [[] for i in range(10)]
    
#     print("train_labels shape, ", np.shape(train_labels), "train_image shape ", np.shape(train_images_vectorize[1]))

#     for i in range(len(train_images_vectorize)):
#         cov_class_i[train_labels[i]].append(train_images_vectorize[i])
    
#     print("np shape", np.shape(np.array(cov_class_i)))
#     print("cov_class_shape ", np.shape(cov_class_i))
    
#     for i in range(10):
#         print("shape of ", i, " class", np.shape(cov_class_i[i]))
#         cov_class_i[i] = np.array(cov_class_i[i])
#         cov_class_i[i] = cov_class_i[i] - mean_vector(cov_class_i[i])
#         cov_class_i[i] = np.matmul(cov_class_i[i].T, cov_class_i[i])
        
#     #sum_dim = np.shape(np.sum(cov_class_i, axis=0))
#     #print("dimensions of cov sum ", sum_dim)


def between_class_covariance(within_class_covariance, total_covariance):
    return total_covariance - within_class_covariance

# Ans 2
def fda(train_images_vectorize, train_labels):
    train_images_vectorize -= mean_vector(train_images_vectorize)
    within_class_covariance_val = within_class_covariance(train_images_vectorize, train_labels)
    total_covariance = np.cov(train_images_vectorize.T)
    between_class_covariance_val = between_class_covariance(within_class_covariance_val, total_covariance)

    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(np.linalg.pinv(within_class_covariance_val), between_class_covariance_val ))
    
    index_for_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index_for_sort]
    eigenvectors = eigenvectors[:,index_for_sort]   
    print(eigenvalues[:5])
    last_important_eigen_value_index = 9

    encoded_data = np.matmul(train_images_vectorize, eigenvectors[:,:last_important_eigen_value_index])
    
    print("eigenvalues shape", np.shape(eigenvalues), "eigenvectors shape", np.shape(eigenvectors[:,:last_important_eigen_value_index]))
    
    #print(np.shape(encoded_data))

    return encoded_data, eigenvectors[:,:last_important_eigen_value_index]

# Ans 4
def lda(train_images_vectorize, train_labels):
    dict_lda = {}
    prior_probabilities = {}
    mu_vals = {}
    covar = {}
    cov_det = {}
    cov_inv = {}

    for i, j in zip(train_labels, train_images_vectorize):
        if(i in dict_lda):
            dict_lda[i].append(j)
            
        else:
            dict_lda[i] = [j]

    for label in np.unique(train_labels):
        dict_lda[label] = np.asarray(dict_lda[label])
        
    for i in dict_lda:
        prior_probabilities[i] = np.shape(dict_lda[i])[0]/np.shape(train_images_vectorize)[0]
    
    for i in dict_lda:
        mu_vals[i] = np.mean(dict_lda[i], axis = 0)
    
    for i in dict_lda:
        covar[i] = np.cov(dict_lda[i].T)
        cov_det[i] = np.linalg.slogdet(covar[i])
        cov_inv[i] = np.linalg.pinv(covar[i])
    return prior_probabilities, mu_vals, covar, cov_det, cov_inv

def lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test):
    discriminant_max = -100000
    predict_class = -1
    

    for i in range(10):
        #print(test.shape)
        #disc = (-1/2)*np.log(cov_det[i]) - (1/2) * np.matmul(np.matmul((test - mu_vals[i]).T, cov_inv[i]), test - mu_vals[i]) + np.log(prior_probab[i])
        disc = (-1/2)*cov_det[i][1] - (1/2) * np.matmul(np.matmul((test - mu_vals[i]).T, cov_inv[i]), test - mu_vals[i]) + np.log(prior_probab[i])
        
        if (disc>discriminant_max):
            discriminant_max = disc
            predict_class = i

    return predict_class


def fda_ndim(train_images_vectorize, train_labels, dim):
    within_class_covariance_val = within_class_covariance(train_images_vectorize, train_labels)
    total_covariance = np.cov(train_images_vectorize.T)
    between_class_covariance_val = between_class_covariance(within_class_covariance_val, total_covariance)

    eigenvalues, eigenvectors = np.linalg.eig(np.matmul(np.linalg.pinv(within_class_covariance_val), between_class_covariance_val ))
    
    print("eigenvalues shape", np.shape(eigenvalues), "eigenvectors shape", np.shape(eigenvectors))
    
    index_for_sort = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index_for_sort]
    eigenvectors = eigenvectors[:,index_for_sort]   
    
    last_important_eigen_value_index = 2

    encoded_data = np.matmul(train_images_vectorize, eigenvectors[:last_important_eigen_value_index].T)
    
    #print(np.shape(encoded_data))

    return encoded_data

train_images, train_labels, test_images, test_labels = load_mnist()
#print(mean_flattened(train_images))

# Ans 1 a.) GLOBAL MEAN
train_images_global_mean = global_mean(train_images)
print("mean global", train_images_global_mean)
## print("Global mean of train_images dataset is: ",train_images_global_mean)

# VECTORIZE DATA: Convert 60K x 28 x 28 to 60K X 784
train_images_vectorize = n_vectorize(train_images)
#train_images_vectorize = StandardScaler().fit_transform(train_images_vectorize)

# Ans 1 a.) MEAN VECTOR
#print((mean_flattened(train_images_vectorize)))
#print("global mean ", np.mean(train_images_vectorize))
#print(np.shape(mean_vector(train_images_vectorize)))
## print("Mean vector of train_images dataset is: ", mean_vector(train_images))

# Ans 1 a.) Covariance
#train_images_covariance = np.cov(train_images_vectorize.T)

##print("Covariance matrix for train_images dataset is: ", train_images_covariance )

#print(np.shape(train_images))
#mean_flattened(train_images)
#mean_vector(train_images)

#n_vectorize(train_images)

## Ans 1 b.) Implementation of PCA

data_after_pca, eigenvectors = pca(train_images_vectorize, energy_boundary = 0.95)
#data_after_fda = fda(train_images_vectorize, train_labels)
print(np.shape(data_after_pca))

#print(np.shape(data_after_fda))

## Ans 3 PCA

data_after_pca, eigenvectors = pca(train_images_vectorize, 0.95)
#features = data_after_pca.T
print(np.shape(data_after_pca))

plt.scatter(data_after_pca[:, 0], data_after_pca[:, 1],
            c=train_labels[:], edgecolor='none', alpha=1,
            cmap=plt.cm.get_cmap('Spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.figure(figsize=(30,20))
plt.show()
## Ans 3 FDA

data_after_fda, eigenvectors = fda(data_after_pca, train_labels)
print(np.shape(data_after_fda))

plt.scatter(data_after_fda[:, 0], data_after_fda[:, 1],
            c=train_labels[:], edgecolor='none', alpha=0.3,
            cmap=plt.cm.get_cmap('Dark2', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()
plt.show()

#prior_probab, mu_vals, covar = lda(train_images_vectorize, train_labels)
## Ans e 

# Apply PCA then LDA on training

data_after_pca, eigenvectors = pca(train_images_vectorize, 0.95)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_pca, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_proba`b, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after 0.95 eigen PCA on test data", (((correct)/n_test)*100))


#eigenvectors = eigenvectors.reshape()

# first_train_image = eigenvectors.T

# pixels = first_train_image[17].reshape((28,28))

# plt.imshow(pixels, cmap='gray')
# plt.show()
# plt.close

data_after_pca, eigenvectors = pca(train_images_vectorize, 0.70)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_pca, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_probab, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after 0.70 eigen PCA on test data", (((correct)/n_test)*100))

data_after_pca, eigenvectors = pca(train_images_vectorize, 0.90)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_pca, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_probab, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after 0.90 eigen PCA on test data", (((correct)/n_test)*100))

data_after_pca, eigenvectors = pca(train_images_vectorize, 0.99)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_pca, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_probab, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after 0.99 eigen PCA on test data", (((correct)/n_test)*100))

####FDA

data_after_fda, eigenvectors = fda(train_images_vectorize, train_labels)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_fda, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_probab, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after FDA on test data", (((correct)/n_test)*100))

####FDA then PCA

data_after_fda, eigenvectors = fda(data_after_pca, train_labels)
data_after_pca, eigenvectors = pca(train_images_vectorize, 0.95)
prior_probab, mu_vals, covar, cov_det, cov_inv = lda(data_after_pca, train_labels)

test_images_vectorize = n_vectorize(test_images)
test_images_vectorize = test_images_vectorize - mean_vector(test_images_vectorize)
test_images_vectorize_projected = np.matmul(test_images_vectorize, eigenvectors)

correct = 0

n_test = np.shape(test_images_vectorize_projected)[0]

for i in range(n_test):
    if(lda_predict(prior_probab, mu_vals, covar, cov_det, cov_inv, test_images_vectorize_projected[i]) == test_labels[i]):
        correct+=1
    #print(train_labels[i], lda_predict(prior_probab, mu_vals, train_images_vectorize[i]))

print("Accuracy on test data after applying FDA then PCA", (((correct)/n_test)*100))

###