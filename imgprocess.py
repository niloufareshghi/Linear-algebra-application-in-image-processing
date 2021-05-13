from PIL import Image, ExifTags
from numpy import asarray
import numpy as np

# باز کردن عکس
image = Image.open('skyandtree.jpg')

# ذخیره عکس در قالب ماتریس
data = asarray(image)
print("فرم ماتریسی: ")
print(data)

# رسم ابرداده
exif_data = image._getexif()
exif = {
    ExifTags.TAGS[k]: v
    for k, v in image._getexif().items()
    if k in ExifTags.TAGS
}
print()
print("ابر داده: ")
for key, value in exif.items():
    print(key, ' : ', value)


X = data.transpose(2, 0, 1).reshape(3, -1)

# میانگین
meanX = np.mean(X, axis=0)
print()
print("میانگین داده")
print(meanX)

# کوواریانس
covariance = np.cov(X)
print()
print("کوواریانس")
print(covariance)
lenCov = covariance.shape[0]

# واریانس
print()
print("واریانس Xj ها")
variances = []
for i in range(lenCov):
    variances.append(covariance[i, i])
    print("Variance of X{} is = {} ".format(i, covariance[i, i]))
trace = covariance.trace()
print()
print("واریانس کل داده")
print(trace)

# هم بستگی
for i in range(lenCov):
    for j in range(lenCov):
        if i != j:
            print("Covariance of X{} and X{} is = {}".format(i, j, covariance[i, j]))
            if abs(covariance[i, j]) < 0.001:
                print("X{} and X{} are uncorrelated".format(i, j))
            else:
                print("X{} and X{} are correlated".format(i, j))

# کاهش بعد
explained_variances = []
eigenVal, eigenVec = np.linalg.eig(covariance)
D = np.array([[eigenVal[0], 0, 0], [0, eigenVal[1], 0], [0, 0, eigenVal[2]]])
print()
print("ماتریس کوواریانس برای داده های تبدیل شده")
print(D)
traceD = D.trace()
for i in range(lenCov):
    temp = eigenVal[i]/traceD
    explained_variances.append(temp)
print()
print("واریانس های توضیح داده شده توسط Yj ها")
explained_variances = np.sort(explained_variances)
print(explained_variances)


def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1]
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num


k = Percentage2n(eigenVal, explained_variances[2])
eigenvalue = np.argsort(eigenVal)
K_eigenValue = eigenvalue[-1:-(k+1):-1]
K_eigenVector = eigenVec[:, K_eigenValue]

meanX.reshape(-1, 1)

reducedMat = X.T.dot(K_eigenVector)

# ایجاد تصویر جدید به روش PCA

reconstructedMat = (reducedMat.dot(K_eigenVector.T)).T + meanX

newData = reconstructedMat.T.reshape(data.shape[0], data.shape[1], data.shape[2])
newData = newData.astype(np.uint8)
print()
print("فرم آرایه ایی تصویر جدید")
print(newData)
img = Image.fromarray(newData).save("out.jpeg")
