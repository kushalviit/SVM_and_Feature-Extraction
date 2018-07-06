import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der,fmin_cg
from sklearn import svm


#def svmTrain(X, Y, C, kernelFunction,tol, max_passes):

def plotData(X,y):
    pos_index=np.where(y==1)[0]
    neg_index=np.where(y==0)[0]
    pos_X=X[pos_index,:]
    neg_X=X[neg_index,:]
    plt.figure()
    plt.scatter(pos_X[:,0],pos_X[:,1],marker='^')
    plt.scatter(neg_X[:,0],neg_X[:,1],marker='o')
    plt.show()


def decisionPlot(X,y,clf):
    #max_x_0=np.amax(X[:,0])
    #min_x_0=np.amin(X[:,0])
    #w_0=clf.coef_[0][0]
    #w_1=clf.coef_[0][1]
    #b=clf.intercept_[0]
    #x_p=np.arange(min_x_0,max_x_0,0.2)
    #y_p=(-1*(np.multiply(w_0,x_p)+b))/w_1
    
    plt.figure()
    plt.clf()
    plt.scatter(X[:,0],X[:,1], c=y, zorder=10, cmap=plt.cm.Paired,edgecolor='k', s=20)
    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
    #plt.plot(x_p,y_p)
    plt.show()


def gaussianKernel(x1, x2, sigma):
    x1=np.matrix(x1.reshape((x1.size,1)))
    x2=np.matrix(x2.reshape((x2.size,1)))
    diff=x1-x2
    sim=np.exp((-1*np.matmul(diff.T,diff))/(2*sigma*sigma))
    return sim[0][0]


def  dataset3Params(X, y, Xval, yval):
     C_set=np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
     sigma_set=np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
     prediction_error=[]
     index_pair=[]
     for i in range(C_set.size):
         for j in range(sigma_set.size):
             clf=svm.SVC(C=C_set[i],kernel='rbf',gamma=(1/sigma_set[j]),tol=1e-6)
             clf.fit(X,y)
             pred=clf.predict(Xval)
             pe=np.mean(pred!=yval)
             prediction_error.append(pe)
             index_pair.append([i,j])    
     min_index=np.argmin(np.array(prediction_error))
     [im,jm]=index_pair[min_index]
     return [C_set[im],sigma_set[jm]]
             


file_name="ex6data1.mat"
data_content=sio.loadmat(file_name)
X=data_content['X']
y1=data_content['y']
y=np.squeeze(np.asarray(y1))


plotData(X,y1)
c = 1
clf = svm.LinearSVC(C=c)
clf.fit(X, y)
decisionPlot(X,y1,clf)
x1 =np.matrix(np.array([1,2,1])).T
x2 =np.matrix(np.array([0,4,-1])).T
sigma = 2
si=gaussianKernel(x1,x2,sigma)
print("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1](for sigma = 2, this value should be about 0.324652) :")
print(si)

del file_name,data_content,X,y1,y,clf
file_name="ex6data2.mat"
data_content=sio.loadmat(file_name)
X=data_content['X']
y1=data_content['y']
y=np.squeeze(np.asarray(y1))

plotData(X,y1)

c = 1; sigma = 0.1

clf=svm.SVC(C=c,kernel='rbf',gamma=(1/sigma),tol=1e-6)
clf.fit(X,y)
decisionPlot(X,y1,clf)


del file_name,data_content,X,y1,y,clf
file_name="ex6data3.mat"
data_content=sio.loadmat(file_name)
X=data_content['X']
y1=data_content['y']
Xval=data_content['Xval']
yval=data_content['yval']
y=np.squeeze(np.asarray(y1))


[C, sigma] = dataset3Params(X, y, Xval, yval)

clf=svm.SVC(C=C,kernel='rbf',gamma=(1/sigma),tol=1e-6)
clf.fit(X,y)
decisionPlot(X,y1,clf)








