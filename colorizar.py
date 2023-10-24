import numpy as np
import argparse
import os
import cv2

DIR = r"C:\development\pypro\colorization"
PROTOTXT = os.path.join(DIR,r"models\colorization_deploy_v2.prototxt")
MODEL = os.path.join(DIR,r"models\colorization_release_v2.caffemodel")
POINTS = os.path.join(DIR,r"models\pts_in_hull.npy")
parser = argparse.ArgumentParser()
parser.add_argument("-i",type=str, required=True, help="Name image")
args = parser.parse_args()
print(args.i)

print("Cargando modelos espero un momentito.....")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
print("Modelo cargados correctamente 😉")

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")

pts = pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313],2.606, dtype="float32")]

img = cv2.imread(args.i)
scaled = img.astype("float32")/255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab, (224,224))
L = cv2.split(resized)[0]
L -= 50

print("Procesando imagen.....")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1,2,0))

ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:,:,np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0,1)

colorized = (255 * colorized).astype("uint8")
cv2.imshow("Original", img)
cv2.imshow("Colorized", colorized)
cv2.waitKey(0)

cv2.imwrite("colorized.jpg", colorized)

print("Imagen procesada correctamente 😎")