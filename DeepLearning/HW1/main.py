import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *
#import wandb


data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    idx_1 = np.where((y==1))
    idx_ow = np.where((y==-1))
    
    X_1_0 = X[idx_1 , 0]
    X_1_1 = X[idx_1 , 1]
    
    X_ow_0 = X[idx_ow , 0]
    X_ow_1 = X[idx_ow , 1]
    
    plt.scatter(X_1_0, X_1_1, c='blue' , label = 'Y = 1')
    plt.scatter(X_ow_0, X_ow_1, c='green' , label = 'Y = 2')
    plt.legend()
    plt.title('')
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.savefig('train_features.png')


def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
        
    '''
    plt.figure(facecolor='white')
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = np.dot(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], W)
    Z = np.where(Z >= 0, 1, -1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4, colors=['lightblue', 'lightgreen'])
    plt.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, 'green', 'blue'), marker='o')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Sigmoid Model Decision Boundary')
    plt.savefig('train_result_sigmoid.png')
    plt.show()

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 
    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    plt.figure(facecolor='white')
    x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
    y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = np.dot(np.c_[np.ones((xx.ravel().shape[0], 1)), xx.ravel(), yy.ravel()], W)
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, marker='o', edgecolor='black')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Softmax Model Decision Boundary')
 
    plt.savefig('train_result_softmax.png')
    plt.show()

	### END YOUR CODE

def main():
    #wandb.login()
    #run = wandb.init(project="DL_HW1")
	# ------------Data Preprocessing------------
	# Read data for training.
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. ##WHY????
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)
    Q3 = 1
    if Q3:
       # ------------Logistic Regression Sigmoid Case------------

       ##### Check BGD, SGD, miniBGD
        logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)
        print('For Training Set: ')
        print('For BGD: ')
        logisticR_classifier.fit_BGD(train_X, train_y)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(equal to BGD): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For SGD: ')
        logisticR_classifier.fit_SGD(train_X, train_y)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(equal to SGD): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(batch_size = 10): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))


        # Explore different hyper-parameters.
        ### YOUR CODE HERE
        logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=4000)
        print('After Fine-tuning the Hyperarameter: ')
        print('For Training Set: ')
        print('For BGD: ')
        logisticR_classifier.fit_BGD(train_X, train_y)
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(equal to BGD): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For SGD: ')
        logisticR_classifier.fit_SGD(train_X, train_y)
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(equal to SGD): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))

        print('For miniBGD(batch_size = 32): ')
        logisticR_classifier.fit_miniBGD(train_X, train_y, 32)
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))
        ### END YOUR CODE

        # Visualize the your 'best' model after training.
        ### YOUR CODE HERE
        best_logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=4000)
        best_logisticR_classifier.fit_miniBGD(train_X, train_y, 32)
        visualize_result(train_X[:, 1:3], train_y, best_logisticR_classifier.get_params())
        ### END YOUR CODE
        # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
        ### YOUR CODE HERE
        raw_test, label_test = load_data(os.path.join(data_dir, test_filename))
        test_X_all = prepare_X(raw_test)
        test_y_all, test_idx = prepare_y(label_test)
        test_X = test_X_all[test_idx]
        test_y = test_y_all[test_idx]
        test_y[np.where(test_y==2)] = -1

        print('Score on Test Set: ' , best_logisticR_classifier.score(test_X, test_y))
        ### END YOUR CODE
   
    Q4 = 1
    if Q4:

        # ------------Logistic Regression Multiple-class case, let k= 3------------
        ###### Use all data from '0' '1' '2' for training
        train_X = train_X_all
        train_y = train_y_all
        valid_X = valid_X_all
        valid_y = valid_y_all

        #########  miniBGD for multiclass Logistic Regression
        print('For Multi-Class Problem: ')
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k=3)
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
        print('Weights: ' , logisticR_classifier_multiclass.get_params())
        print('Score on Training Set: ', logisticR_classifier_multiclass.score(train_X, train_y))
        print('Score on Validation Set: ', logisticR_classifier_multiclass.score(valid_X, valid_y))


        # Explore different hyper-parameters.
        ### YOUR CODE HERE
        print('For Multi-Class Problem after Fine-tuning: ')
        best_logistic_multi_R = logistic_regression_multiclass(learning_rate=0.1, max_iter=10000,  k=3)
        best_logistic_multi_R.fit_miniBGD(train_X, train_y, 64)
        print('Weights: ' , best_logistic_multi_R.get_params())
        print('Score on Training Set: ', best_logistic_multi_R.score(train_X, train_y))
        print('Score on Validation Set: ', best_logistic_multi_R.score(valid_X, valid_y))
        ### END YOUR CODE

        # Visualize the your 'best' model after training.
        visualize_result_multi(train_X[:, 1:3], train_y, best_logistic_multi_R.get_params())
        # Use the 'best' model above to do testing.
        ### YOUR CODE HERE
        raw_test, labels_test = load_data(os.path.join(data_dir, test_filename))
        test_X = prepare_X(raw_test)
        test_y, _ = prepare_y(labels_test)
        print('Score on Test Set: ' , best_logistic_multi_R.score(test_X, test_y))
        ### END YOUR CODE
    Q5 = 1
    if Q5:
        # ------------Connection between sigmoid and softmax------------
        ############ Now set k=2, only use data from '1' and '2' 

        #####  set labels to 0,1 for softmax classifer
        train_X = train_X_all[train_idx]
        train_y = train_y_all[train_idx]
        train_X = train_X[0:1350]
        train_y = train_y[0:1350]
        valid_X = valid_X_all[val_idx]
        valid_y = valid_y_all[val_idx] 
        train_y[np.where(train_y==2)] = 0
        valid_y[np.where(valid_y==2)] = 0  

        ###### First, fit softmax classifer until convergence, and evaluate 
        ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
        ### YOUR CODE HERE
        print('Fitting Softmax Classifer: ')
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=100000,  k=2)
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 64)
        print('Weights: ' , logisticR_classifier_multiclass.get_params())
        print('Score on Training Set: ' , logisticR_classifier_multiclass.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier_multiclass.score(valid_X, valid_y))
        ### END YOUR CODE
        train_X = train_X_all[train_idx]
        train_y = train_y_all[train_idx]
        train_X = train_X[0:1350]
        train_y = train_y[0:1350]
        valid_X = valid_X_all[val_idx]
        valid_y = valid_y_all[val_idx] 
        #####       set lables to -1 and 1 for sigmoid classifer
        train_y[np.where(train_y==2)] = -1
        valid_y[np.where(valid_y==2)] = -1   

        ###### Next, fit sigmoid classifer until convergence, and evaluate
        ### YOUR CODE HERE
        print('Fitting Sigmoid Classifer: ')
        logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=100000)
        logisticR_classifier.fit_miniBGD(train_X, train_y, 64)
        print('Weights: ' , logisticR_classifier.get_params())
        print('Score on Training Set: ' , logisticR_classifier.score(train_X, train_y))
        print('Score on Validation Set: ' , logisticR_classifier.score(valid_X, valid_y))
        ### END YOUR CODE
        ################Compare and report the observations/prediction accuracy
        ### YOUR CODE HERE
        data_shape= train_y.shape[0] 
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=1,  k=2)
        logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, data_shape)
        print('Weights for SF: ' , logisticR_classifier_multiclass.get_params())
        
        logisticR_classifier = logistic_regression(learning_rate=0.1, max_iter=1)
        logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
        print('Weights for Sig: ' , logisticR_classifier.get_params())
        ### END YOUR CODE

    # ------------End------------
    
    
if __name__ == '__main__':
	main()
    
    
