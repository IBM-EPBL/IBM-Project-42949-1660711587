{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Calibri;}}
{\*\generator Msftedit 5.41.21.2510;}\viewkind4\uc1\pard\sa200\sl276\slmult1\lang9\f0\fs22 Assignment-3 Problem Statement :- Build CNN Model for Classification Of Flowers\par
\u9679? Download the Dataset\par
\u9679? Image Augmentation\par
import tensorflow \par
from tensorflow.keras.preprocessing.image import ImageDataGenerator\par
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)\par
x_train = train_datagen.flow_from_directory(r"C:\\Users\\mstof\\Downloads\\Flowers-Dataset\\flowers",target_size=(64,64),batch_size=32,class_mode="categorical")\par
Found 4317 images belonging to 5 classes.\par
\u9679? Create Model\par
from tensorflow.keras.layers import Convolution2D\par
from tensorflow.keras.layers import MaxPooling2D\par
from tensorflow.keras.layers import Flatten\par
from tensorflow.keras.layers import Dense\par
from tensorflow.keras.models import Sequential\par
model = Sequential()\par
\u9679? Add Layers (Convolution,MaxPooling,Flatten,Dense-(HiddenLayers),Output)\par
model.add(Convolution2D(32,(3,3),input_shape=(64,64,3),activation="relu"))\par
model.add(MaxPooling2D(pool_size=(2,2)))\par
model.add(Flatten())\par
model.add(Dense(units=300,kernel_initializer="random_uniform",activation="relu"))\par
model.add(Dense(units=200,kernel_initializer="random_uniform",activation="relu"))\par
model.add(Dense(units=5,kernel_initializer="random_uniform",activation="softmax"))\par
\u9679? Compile The Model\par
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])\par
\u9679? Fit The Model\par
model.fit(x_train,steps_per_epoch=135,epochs=50)\par
Epoch 1/50\par
135/135 [==============================] - 23s 163ms/step - loss: 1.2656 - accuracy: 0.4617\par
Epoch 2/50\par
135/135 [==============================] - 20s 149ms/step - loss: 0.9821 - accuracy: 0.6034\par
Epoch 3/50\par
135/135 [==============================] - 19s 141ms/step - loss: 0.6702 - accuracy: 0.7464\par
Epoch 4/50\par
135/135 [==============================] - 21s 152ms/step - loss: 0.4049 - accuracy: 0.8559\par
Epoch 5/50\par
135/135 [==============================] - 23s 172ms/step - loss: 0.1788 - accuracy: 0.9460\par
Epoch 6/50\par
135/135 [==============================] - 22s 160ms/step - loss: 0.0907 - accuracy: 0.9761\par
Epoch 7/50\par
135/135 [==============================] - 21s 155ms/step - loss: 0.0373 - accuracy: 0.9910\par
Epoch 8/50\par
135/135 [==============================] - 20s 147ms/step - loss: 0.0549 - accuracy: 0.9873\par
Epoch 9/50\par
135/135 [==============================] - 19s 138ms/step - loss: 0.0509 - accuracy: 0.9886\par
Epoch 10/50\par
135/135 [==============================] - 19s 139ms/step - loss: 0.0399 - accuracy: 0.9884\par
Epoch 11/50\par
135/135 [==============================] - 19s 138ms/step - loss: 0.0585 - accuracy: 0.9856\par
Epoch 12/50\par
135/135 [==============================] - 19s 142ms/step - loss: 0.0526 - accuracy: 0.9875\par
Epoch 13/50\par
135/135 [==============================] - 20s 145ms/step - loss: 0.0247 - accuracy: 0.9931\par
Epoch 14/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0504 - accuracy: 0.9854\par
Epoch 15/50\par
135/135 [==============================] - 18s 134ms/step - loss: 0.0543 - accuracy: 0.9845\par
Epoch 16/50\par
135/135 [==============================] - 18s 133ms/step - loss: 0.0201 - accuracy: 0.9949\par
Epoch 17/50\par
135/135 [==============================] - 18s 136ms/step - loss: 0.0088 - accuracy: 0.9986\par
Epoch 18/50\par
135/135 [==============================] - 18s 135ms/step - loss: 0.0045 - accuracy: 0.9988\par
Epoch 19/50\par
135/135 [==============================] - 18s 135ms/step - loss: 0.0045 - accuracy: 0.9984\par
Epoch 20/50\par
135/135 [==============================] - 18s 133ms/step - loss: 0.0059 - accuracy: 0.9986\par
Epoch 21/50\par
135/135 [==============================] - 18s 134ms/step - loss: 0.0047 - accuracy: 0.9991\par
Epoch 22/50\par
135/135 [==============================] - 19s 141ms/step - loss: 0.0047 - accuracy: 0.9988\par
Epoch 23/50\par
135/135 [==============================] - 19s 142ms/step - loss: 0.0031 - accuracy: 0.9988\par
Epoch 24/50\par
135/135 [==============================] - 19s 138ms/step - loss: 0.0025 - accuracy: 0.9986\par
Epoch 25/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0022 - accuracy: 0.9988\par
Epoch 26/50\par
135/135 [==============================] - 19s 137ms/step - loss: 0.0020 - accuracy: 0.9984\par
Epoch 27/50\par
135/135 [==============================] - 18s 134ms/step - loss: 0.0043 - accuracy: 0.9986\par
Epoch 28/50\par
135/135 [==============================] - 20s 145ms/step - loss: 0.0030 - accuracy: 0.9988\par
Epoch 29/50\par
135/135 [==============================] - 23s 170ms/step - loss: 0.0022 - accuracy: 0.9986\par
Epoch 30/50\par
135/135 [==============================] - 22s 163ms/step - loss: 0.0016 - accuracy: 0.9988\par
Epoch 31/50\par
135/135 [==============================] - 21s 154ms/step - loss: 0.0015 - accuracy: 0.9986\par
Epoch 32/50\par
135/135 [==============================] - 22s 165ms/step - loss: 0.0015 - accuracy: 0.9991\par
Epoch 33/50\par
135/135 [==============================] - 20s 148ms/step - loss: 0.0022 - accuracy: 0.9986\par
Epoch 34/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0070 - accuracy: 0.9984\par
Epoch 35/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.2664 - accuracy: 0.9187\par
Epoch 36/50\par
135/135 [==============================] - 17s 129ms/step - loss: 0.0823 - accuracy: 0.9754\par
Epoch 37/50\par
135/135 [==============================] - 17s 129ms/step - loss: 0.0344 - accuracy: 0.9905\par
Epoch 38/50\par
135/135 [==============================] - 18s 130ms/step - loss: 0.0134 - accuracy: 0.9947\par
Epoch 39/50\par
135/135 [==============================] - 17s 129ms/step - loss: 0.0078 - accuracy: 0.9975\par
Epoch 40/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0035 - accuracy: 0.9981\par
Epoch 41/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0056 - accuracy: 0.9991\par
Epoch 42/50\par
135/135 [==============================] - 18s 132ms/step - loss: 0.0024 - accuracy: 0.9991\par
Epoch 43/50\par
135/135 [==============================] - 18s 134ms/step - loss: 0.0019 - accuracy: 0.9986\par
Epoch 44/50\par
135/135 [==============================] - 19s 140ms/step - loss: 0.0018 - accuracy: 0.9991\par
Epoch 45/50\par
135/135 [==============================] - 20s 148ms/step - loss: 0.0017 - accuracy: 0.9988\par
Epoch 46/50\par
135/135 [==============================] - 21s 155ms/step - loss: 0.0016 - accuracy: 0.9986\par
Epoch 47/50\par
135/135 [==============================] - 24s 180ms/step - loss: 0.0015 - accuracy: 0.9988\par
Epoch 48/50\par
135/135 [==============================] - 21s 155ms/step - loss: 0.0015 - accuracy: 0.9993\par
Epoch 49/50\par
135/135 [==============================] - 21s 155ms/step - loss: 0.0014 - accuracy: 0.9991\par
Epoch 50/50\par
135/135 [==============================] - 23s 169ms/step - loss: 0.0015 - accuracy: 0.9991\par
\u9679? Save The Model\par
model.save("Flowers.h5")\par
\u9679? Test The Model\par
from tensorflow.keras.models import load_model\par
from tensorflow.keras.preprocessing import image\par
model = load_model("Flowers.h5")\par
img = image.load_img("jbskk.jpg",target_size=(64,64))\par
img\par
\par
type(img)\par
PIL.Image.Image\par
x=image.img_to_array(img)\par
x\par
array([[[118.,  81.,  63.],\par
        [131.,  83.,  71.],\par
        [127.,  79.,  67.],\par
        ...,\par
        [140.,  98.,  76.],\par
        [146., 104.,  82.],\par
        [157., 114.,  95.]],\par
\par
       [[147.,  98.,  84.],\par
        [161.,  99.,  88.],\par
        [166., 100.,  88.],\par
        ...,\par
        [151., 109.,  87.],\par
        [158., 115.,  96.],\par
        [162., 124., 105.]],\par
\par
       [[162., 110.,  97.],\par
        [174., 104.,  94.],\par
        [172.,  98.,  85.],\par
        ...,\par
        [146., 108.,  89.],\par
        [157., 120., 101.],\par
        [163., 127., 111.]],\par
\par
       ...,\par
\par
       [[126.,  94.,  83.],\par
        [114.,  84.,  73.],\par
        [161., 120., 114.],\par
        ...,\par
        [ 88.,  66.,  53.],\par
        [ 89.,  67.,  54.],\par
        [ 95.,  71.,  59.]],\par
\par
       [[ 13.,  13.,  13.],\par
        [ 44.,  18.,  17.],\par
        [ 99.,  61.,  52.],\par
        ...,\par
        [100.,  76.,  64.],\par
        [108.,  84.,  72.],\par
        [106.,  84.,  71.]],\par
\par
       [[ 13.,  13.,  15.],\par
        [ 12.,  13.,  15.],\par
        [ 12.,  13.,  15.],\par
        ...,\par
        [108.,  86.,  73.],\par
        [105.,  85.,  74.],\par
        [113.,  90.,  82.]]], dtype=float32)\par
x.shape\par
(64, 64, 3)\par
import numpy as np\par
x=np.expand_dims(x,axis=0)\par
x.shape\par
(1, 64, 64, 3)\par
pred_prob=model.predict(x)\par
1/1 [==============================] - 0s 41ms/step\par
pred_prob\par
array([[0., 0., 1., 0., 0.]], dtype=float32)\par
class_name=np.array(["daisy","dandelion","rose","sunflower","tulip"])\par
pred_id=pred_prob.argmax(axis=1)\par
pred_id\par
array([2], dtype=int64)\par
print("The pedicted flower is",str(class_name[pred_id]))\par
The pedicted flower is ['rose']\par
END OF THE ASSIGNMENT\par
}
