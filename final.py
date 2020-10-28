import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import locale
import csv
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def main(datapath):
    # importing data set
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    Brooklyn = []
    Manhatten = []
    William = []
    Queen = []
    Total = []
    HighT = []
    LowT = []
    Precipitation = []

    f = open(datapath, newline='')
    csv_reader = csv.reader(f)
    next(csv_reader)
    for i in csv_reader:
        Brooklyn.append(locale.atof(i[5]))
        Manhatten.append(locale.atof(i[6]))
        William.append(locale.atof(i[7]))
        Queen.append(locale.atof(i[8]))
        Total.append(locale.atof(i[9]))
        HighT.append(locale.atof(i[2]))
        LowT.append(locale.atof(i[3]))
        Precipitation.append(i[4])

    # calculating mse for each bridge
    bmw = []
    bmq = []
    bwq = []
    mwq = []
    total_o = Total.copy()
    ## step 1
    # calculate the sum of four different combination
    B_mean, B_std = mean_std(Brooklyn)
    M_mean, M_std = mean_std(Manhatten)
    W_mean, W_std = mean_std(William)
    Q_mean, Q_std = mean_std(Queen)
    T_mean, T_std = mean_std(Total)

    Brooklyn = norm_list(Brooklyn, B_mean, B_std)
    Manhatten = norm_list(Manhatten, M_mean, M_std)
    William = norm_list(William, W_mean, W_std)
    Queen = norm_list(Queen, Q_mean, Q_std)
    Total = norm_list(Total, T_mean, T_std)

    for i in range(len(Total)):
        bmw.append((Brooklyn[i] + Manhatten[i] + William[i])/3)
        bmq.append((Brooklyn[i] + Manhatten[i] + Queen[i])/3)
        bwq.append((Brooklyn[i] + William[i] + Queen[i])/3)
        mwq.append((Manhatten[i] + William[i] + Queen[i])/3)

    # calculate the mse for each combo
    BMW = MSE_score(bmw, Total)
    BMQ = MSE_score(bmq, Total)
    BWQ = MSE_score(bwq, Total)
    MWQ = MSE_score(mwq, Total)

    print("the mean square error between BMW and total {}".format(BMW))
    print("the mean square error between BMQ and total {}".format(BMQ))
    print("the mean square error between BWQ and total {}".format(BWQ))
    print("the mean square error between MWQ and total {}".format(MWQ))

    ## step2
    # modify the Precipitation data from string to float
    for i in range(len(Precipitation)):
        if Precipitation[i] == 'T':
            Precipitation[i] = 0
        elif 'S' in Precipitation[i]:
            Precipitation[i] = 0.47
        else:
            Precipitation[i] = float(Precipitation[i])

    # feature and target matrices
    X = np.array([HighT, LowT, Precipitation])
    X = X.transpose()
    y = np.array(Total)
    y = y.transpose()

    # Training and testing split, with 25% of the data reserved as the test set
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    # Normalizing training and testing data
    [X_train_n, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    # Define the range of lambda to test
    lmbda = np.logspace(-1, 2, num=101)  # fill in

    MODEL = []
    MSE = []
    for l in lmbda:
        # Train the regression model using a regularization parameter of l
        model = train_model(X_train_n, y_train, l)

        # Evaluate the MSE on the test set
        mse = error(X_test, y_test, model)
        # Store the model and mse in lists for further processing

        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda

    plt.plot(lmbda, MSE, color='orange') #fill in
    plt.xlabel('Regularization Parameter Lambda')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Lambda')
    plt.show()

    ind = MSE.index(min(MSE))
    [lmda_best, MSE_best, model_best] = [lmbda[ind], MSE[ind], MODEL[ind]]
    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))

    total_test = model_best.predict(X_test)

    #r^2
    r2 = r2_score(y_test, total_test)
    print("coefficient of determination: ", r2)

    # question 3
    prec_binary = [int(x!=0) for x in Precipitation]
    X3 = np.array([total_o])
    X3 = X3.transpose()
    y3 = np.array(prec_binary)
    y3 = y3.transpose()
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.4, random_state=1)
    mnb = MultinomialNB()
    mnb.fit(X_train3, y_train3)
    y_pred3 = mnb.predict(X_test3)

    print("Multinomial Naive Bayes model accuracy (in %):", accuracy_score(y_test3,y_pred3)*100)


    return model_best

def MSE_score(x, total):
    sum_mse = 0
    for i in range(len(x)):
        sum_mse += math.pow((total[i] - x[i]), 2)

    return sum_mse

def error(X,y,model):
    y_hat = model.predict(X)
    mse = mean_squared_error(y_hat, y)

    return mse

def train_model(X,y,l):
    model = Ridge(alpha = l, fit_intercept=True)
    model.fit(X,y)

    return model

def normalize_test(X_test, trn_mean, trn_std):
    X = (X_test - trn_mean) / trn_std

    return X

def normalize_train(X_train):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X = (X_train - mean) / std


    return X, mean, std

def mean_std(test_list):
    mean = sum(test_list) / len(test_list)
    variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list)
    res = variance ** 0.5
    return mean, res

def norm_list(list,mean, std):
    X = [(list[i] - mean) / std for i in range(len(list))]
    return X

def feature_matrix(x, d):
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    length = len(x)
    X = [[np.power(x[i], j) for j in range(d, -1, -1)] for i in range(length)]
    return X

def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.linalg.inv(X.T @ X) @ X.T @ y

    return B.tolist()



if __name__ == '__main__':
    model_best = main("NYC_Bicycle_Counts_2016_Corrected.csv")
    print(model_best.coef_)
    print(model_best.intercept_)