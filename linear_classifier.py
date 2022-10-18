import numpy as np
from linear_svm import *
from softmax import *
import torch
import torchvision.transforms as trans
from sklearn import datasets,cross_validation
from scipy.spatial import distance
import pandas as pd

class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def tracin_get(self, a, b):
        return sum([torch.dot(at.flatten(), bt.flatten()) for at, bt in zip(a, b)])

    def cal_distance(self,tracin_dis,train_dis):
        #包括欧氏距离，余弦距离，曼哈顿城市距离，切比雪夫距离，相关因素距离，马氏距离
        eu_dist=distance.cdist(tracin_dis,train_dis,'euclidean')
        cos_dist=distance.cdist(tracin_dis,train_dis,'cosine')
        city_dist=distance.cdist(tracin_dis,train_dis,'cityblock')
        cheby_dist=distance.cdist(tracin_dis,train_dis,'chebyshev')
        corre_dist=distance.cdist(tracin_dis,train_dis,'correlation')
        #maha_dist=distance.cdist(tracin_dis,train_dis,'mahalanobis')

        return eu_dist,cos_dist,city_dist,cheby_dist,corre_dist

    def tracin(self,grad_z_test,test_idx,reg,learning_rate):
        data = datasets.load_iris()
        X_train = data.data
        y_train = data.target
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_train, y_train, test_size=0.25,
                                                                             random_state=0, stratify=y_train)
        num_training = 100
        num_validation = 12
        num_test = 38
        num_dev = 50
        #验证集12个
        mask = range(num_training, num_training + num_validation)
        X_tracin = X_train[mask]
        y_tracin = y_train[mask]
        #训练集100个
        mask_train=range(num_training)
        X_training=X_train[mask_train]
        y_training=y_train[mask_train]
        #这里先保留两个图片的距离，先不插入辅助列
        X_tracin_dis=X_tracin
        X_train_dis=X_training

        #处理数据
        mean_image = np.mean(X_train, axis=0)
        X_tracin -= mean_image
        X_tracin = np.hstack([X_tracin, np.ones((X_tracin.shape[0], 1))])
        num_tracin, dim = X_tracin.shape
        #print('this dim is {}'.format(dim))
        num_classes = np.max(y_tracin) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)
        batchs = np.random.choice(num_tracin, num_tracin, replace=False)
        #设置一个csv
        #df=pd.DataFrame()
        #开始统计距离
        eu=0
        cos=0
        city=0
        cheby=0
        corre=0

        tracin_score=0
        #开始内循环
        for it in range(num_tracin):
            X_batch = None
            y_batch = None
            #print(it)
            batch_idx = np.array(batchs[it]).reshape(1) #np.random.choice(num_tracin, 1, replace=False)  # 这里是随机选一个，考虑锚点的话应该改成顺序选择
            #print("this point is Point_{}".format(batch_idx))
            X_batch = X_tracin[batch_idx]
            y_batch = y_tracin[batch_idx]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            grad_z_train = grad  # self.get_grad(grad, batch_idx)

            t = trans.ToTensor()
            grad_test = t(grad_z_test)
            grad_train = t(grad_z_train)
            score = self.tracin_get(grad_test, grad_train)
            #这里计算两者的物理距离
            #print('X_tracin_dis is {}'.format(X_tracin_dis[batch_idx]))
            #print('X_train_dis is {}'.format(X_train_dis[test_idx]))
            eu_dist,cos_dist,city_dist,cheby_dist,corre_dist=self.cal_distance(X_tracin_dis[batch_idx],X_train_dis[test_idx])
            #先全部相加
            eu  =eu + eu_dist
            cos = cos+cos_dist
            city = city + city_dist
            cheby = cheby+cheby_dist
            corre = corre + corre_dist
            #maha = maha + maha_dist
            tracin_score=tracin_score+score
            # perform parameter update

            self.W += - learning_rate * grad
            print('{} is score between train_sample {} & test_sample {} '.format(score,test_idx,batch_idx))
        #这里先取平均值
        #df['tracin_score']=tracin_score
        euclidean=eu/num_tracin
        cosine=cos/num_tracin
        cityblock=city/num_tracin
        chebyshev=cheby/num_tracin
        correlation=corre/num_tracin
        #mahalanobis=maha/num_tracin
        #df.to_csv()
        return tracin_score,euclidean,cosine,cityblock,chebyshev,correlation#,mahalanobis
            #print(score)

    def train(self, X, y, learning_rate=1e-3, reg=1e-5,epochs=10,
              batch_size=100, verbose=False):
        """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Outputs:
    A list containing the value of the loss function at each training iteration.
    """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_all = []
        grad_all = []
        idx_all=[]
        #设置一个可控外循环
        batchs = np.random.choice(num_train, num_train, replace=False)
        for epoch in range(epochs):
          #按照train_size加载数据
          print('This is epoch {}'.format(epoch))
          loss_history = []
          grad_history = []
          idx = []
          #print(num_train)
          #batchs=np.random.choice(num_train, num_train, replace=False)
          #print(batchs)
          #开一些空列表存储距离
          df = pd.DataFrame()
          score=[]
          eu=[]
          cos=[]
          city=[]
          cheby=[]
          corre=[]
          maha=[]
          label=[]

          for it in range(num_train):
              X_batch = None
              y_batch = None
              #print(it)
              batch_idx =np.array(batchs[it]).reshape(1)#np.random.choice(num_train, batch_size, replace=False)#只能在trainset里挑且不能重复

              #print("Now is sample {}".format(batch_idx))
              idx.append(batch_idx)
              X_batch = X[batch_idx]
              y_batch = y[batch_idx]
              label.append(int(y_batch))


              # evaluate loss and gradient
              loss, grad = self.loss(X_batch, y_batch, reg)
              loss_history.append(loss)
              grad_history.append(grad)
              grad_z_test=grad
              tracin_score,euclidean,cosine,cityblock,chebyshev,correlation=self.tracin(grad_z_test,batch_idx,reg,learning_rate)
              score.append(float(tracin_score))

              eu.append(float(euclidean))
              cos.append(float(cosine))
              city.append(float(cityblock))
              cheby.append(float(chebyshev))
              corre.append(float(correlation))
              #maha.append(mahalanobis)

              # perform parameter update

              self.W += - learning_rate * grad



              #if verbose and it % 1 == 0:
              print('epoch: %d/%d sample: %d loss: %f' % (epoch,epochs, batch_idx, loss))
              #print('grad is : {}'.format(grad))
              #print('now W is {}'.format(self.W))
         # for i in loss_history:

          loss_all.append(loss_history)
          grad_all.append(grad_history)
          idx_all.append(idx)
          # 输入进csv文件
          df['sample']=batchs
          df['tracin_score'] = score
          df['euclidean'] = eu
          df['cosine'] = cos
          df['cityblock'] = city
          df['chebyshev'] = cheby
          df['correlation'] = corre
          df['loss']=loss_history
          df['label']=label
          #df['mahalanobis'] = mahalanobis
          df.to_csv('D:/svm_tracin/12_val_epoch_'+str(epoch)+'.csv')

        return loss_all, grad_all,idx_all

    def predict(self, X):
        """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - X: D x N array of training data. Each column is a D-dimensional point.

    Returns:
    - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
      array of length N, and each element is an integer giving the predicted
      class.
    """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)
        # pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

