import csv
from google.colab import drive 
import io
from sklearn.svm import SVC
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
drive.mount('/content/gdrive')
file = open('gdrive/My Drive/Training_Data1.csv', mode='r', encoding='utf-8-sig')
csvreader = csv.reader(file)
rows = [row for row in csvreader]
x = list((map(itemgetter(0,1),rows)))
y = [row[2] for row in rows]
X=np.array(x).astype(np.float)
Y=np.array(y).astype(np.float)
sv=SVC(random_state=43)
sv.fit(X,Y)
file.close()
file = open('gdrive/My Drive/Testing_Data1.csv', mode='r', encoding='utf-8-sig')
csvreader2 = csv.reader(file)
rows2 = [row for row in csvreader2]
x2 = list((map(itemgetter(0,1),rows2)))
y2 = [row[2] for row in rows2]
test1_X=np.array(x2).astype(np.float)
test1_Y=np.array(y2).astype(np.float)
file.close()
pred_label_1=sv.predict(test1_X) 
acc1=np.mean(pred_label_1==test1_Y)

print('Accuracy on test dataset 1 is ',acc1)
h = .02 
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(test1_X,test1_Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(test1_X, test1_Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(test1_X, test1_Y)
lin_svc = svm.LinearSVC(C=C).fit(test1_X, test1_Y)


x_min, x_max = test1_X[:, 0].min() -2, test1_X[:, 0].max() + 1
y_min, y_max = test1_X[:, 1].min() -2, test1_X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

  
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,s=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
plt.clf()
file = open('gdrive/My Drive/Training_Data2.csv', mode='r', encoding='utf-8-sig')
csvreader4 = csv.reader(file)
rows4 = [row for row in csvreader4]
x4 = list((map(itemgetter(0,1),rows4)))
y4 = [row[2] for row in rows4]
X4=np.array(x4).astype(np.float)
Y4=np.array(y4).astype(np.float)
sv=SVC(random_state=43)
sv.fit(X4,Y4)
file.close()
file = open('gdrive/My Drive/Testing_Data2.csv', mode='r', encoding='utf-8-sig')
csvreader3 = csv.reader(file)
rows3 = [row for row in csvreader3]
x3 = list((map(itemgetter(0,1),rows3)))
y3 = [row[2] for row in rows3]
test2_X=np.array(x3).astype(np.float)
test2_Y=np.array(y3).astype(np.float)
file.close()
X=np.array(x4).astype(np.float)
Y=np.array(y4).astype(np.float)
pred_label_2=sv.predict(test2_X)
acc2=np.mean(pred_label_2==test2_Y)

print('Accuracy on test dataset 2 is ',acc2)
h = .02 
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(test2_X,test2_Y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(test2_X, test2_Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(test2_X, test2_Y)
lin_svc = svm.LinearSVC(C=C).fit(test2_X, test2_Y)


x_min, x_max = test2_X[:, 0].min() , test2_X[:, 0].max() 
y_min, y_max = test2_X[:, 1].min() , test2_X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

plt.clf()
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):

    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    plt.scatter(X[:, 0], X[:, 1], c=y4, cmap=plt.cm.coolwarm,s=5)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()
