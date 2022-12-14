{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "2.Image  Augmentation\n",
        "3.Createmodel\n",
        "\n"
      ],
      "metadata": {
        "id": "8SvG_xmw1G1u"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator"
      ],
      "metadata": {
        "id": "LQBpfQjf1KPe"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "train_datagen = ImageDataGenerator(rescale=1./255,\n",
        "                                   zoom_range=0.2,\n",
        "                                   horizontal_flip=True)"
      ],
      "metadata": {
        "id": "WdmdbPtyTdd2"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "test_datagen = ImageDataGenerator(rescale=1./255)"
      ],
      "metadata": {
        "id": "H4tWQZpmUFu-"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "xtrain = train_datagen.flow_from_directory('/content/drive/MyDrive/Classroom/flowers',\n",
        "                                           target_size=(64,64),\n",
        "                                           class_mode='categorical',\n",
        "                                           batch_size=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Y7gbVhZBVRF0",
        "outputId": "573eb3d6-e7c3-41b2-fb04-6d1537373d36"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 3160 images belonging to 5 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "xtest = train_datagen.flow_from_directory('/content/drive/MyDrive/Classroom/flowers',\n",
        "                                           target_size=(64,64),\n",
        "                                           class_mode='categorical',\n",
        "                                           batch_size=10)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6119b245-f479-46dd-aa9f-c49fb25b63d4",
        "id": "VANPFB9GTjBw"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Found 3160 images belonging to 5 classes.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense"
      ],
      "metadata": {
        "id": "FtRKhJ5SZIav"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "4.Add Layers"
      ],
      "metadata": {
        "id": "EgE8tYm21j4Q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential()\n",
        "model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(64,64,3))) \n",
        "model.add(MaxPooling2D(pool_size=(2,2))) \n",
        "model.add(Flatten()) \n",
        "model.add(Dense(300,activation='relu'))\n",
        "model.add(Dense(150,activation='relu'))\n",
        "model.add(Dense(4,activation='softmax')) "
      ],
      "metadata": {
        "id": "9mo3PnKic8DT"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "5.Compile the model"
      ],
      "metadata": {
        "id": "vsFlCUuv1uX1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "5RoaxAhpgFHM"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "6.Fit the model"
      ],
      "metadata": {
        "id": "E8H-SsqO5a2Q"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from keras.callbacks import EarlyStopping,ReduceLROnPlateau"
      ],
      "metadata": {
        "id": "xT-8tKBrUEcZ"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "early_stopping=EarlyStopping(monitor='val_accuracy',\n",
        "                             patience=5)\n",
        "reduce_lr=ReduceLROnPlateau(monitor='val_accuracy',\n",
        "                            patience=5,\n",
        "                            factor=0,min_lr=0.00001)\n",
        "callback= [reduce_lr,early_stopping] "
      ],
      "metadata": {
        "id": "0eUipagKUX69"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit_generator(xtrain,\n",
        "                    steps_per_epoch=len(xtrain),\n",
        "                    epochs=10,\n",
        "                    validation_data=xtest,\n",
        "                    validation_steps=len(xtest))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uoXy9JNbgnKU",
        "outputId": "24d04e62-9f6d-4ab0-c03c-871774b04553"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:7: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.\n",
            "  import sys\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "13/13 [==============================] - 17s 483ms/step - loss: 2.8257 - accuracy: 0.3021 - val_loss: 1.3794 - val_accuracy: 0.2423\n",
            "Epoch 2/10\n",
            "13/13 [==============================] - 6s 471ms/step - loss: 1.2803 - accuracy: 0.4515 - val_loss: 1.2042 - val_accuracy: 0.4939\n",
            "Epoch 3/10\n",
            "13/13 [==============================] - 6s 455ms/step - loss: 1.1233 - accuracy: 0.5703 - val_loss: 0.9858 - val_accuracy: 0.6104\n",
            "Epoch 4/10\n",
            "13/13 [==============================] - 6s 455ms/step - loss: 0.9527 - accuracy: 0.6357 - val_loss: 0.7862 - val_accuracy: 0.7270\n",
            "Epoch 5/10\n",
            "13/13 [==============================] - 6s 454ms/step - loss: 0.8818 - accuracy: 0.6470 - val_loss: 0.7306 - val_accuracy: 0.7362\n",
            "Epoch 6/10\n",
            "13/13 [==============================] - 6s 464ms/step - loss: 0.7774 - accuracy: 0.6922 - val_loss: 0.7415 - val_accuracy: 0.6902\n",
            "Epoch 7/10\n",
            "13/13 [==============================] - 6s 456ms/step - loss: 0.7191 - accuracy: 0.7068 - val_loss: 0.6056 - val_accuracy: 0.7914\n",
            "Epoch 8/10\n",
            "13/13 [==============================] - 6s 455ms/step - loss: 0.6875 - accuracy: 0.7367 - val_loss: 0.7401 - val_accuracy: 0.7178\n",
            "Epoch 9/10\n",
            "13/13 [==============================] - 6s 462ms/step - loss: 0.6907 - accuracy: 0.7407 - val_loss: 0.5864 - val_accuracy: 0.8006\n",
            "Epoch 10/10\n",
            "13/13 [==============================] - 6s 496ms/step - loss: 0.6405 - accuracy: 0.7472 - val_loss: 0.7157 - val_accuracy: 0.7577\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f11ea689ed0>"
            ]
          },
          "metadata": {},
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "7.Save the model"
      ],
      "metadata": {
        "id": "PRXsjbjy1C2k"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model.save('rose.h6')"
      ],
      "metadata": {
        "id": "PRj46wxMhN30"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "8.Test the model"
      ],
      "metadata": {
        "id": "OZ2tB34r4e-N"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "from tensorflow.keras.preprocessing import image"
      ],
      "metadata": {
        "id": "HHEOVXQqhtQ0"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "img = image.load_img('/content/drive/MyDrive/Classroom/flowers/dandelion/10477378514_9ffbcec4cf_m.jpg',target_size=(94,94))"
      ],
      "metadata": {
        "id": "EfBznYqfigu2"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 111
        },
        "id": "qJpetzZci3Fk",
        "outputId": "e3537931-2b32-4f8e-c301-97873e40741f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=94x94 at 0x7FC48B024990>"
            ],
            "image/png": "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\n"
          },
          "metadata": {},
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img=image.load_img('/content/drive/MyDrive/Classroom/flowers/sunflower/1008566138_6927679c8a.jpg',target_size=(64,64))\n",
        "x=image.img_to_array(img)\n",
        "x=np.expand_dims(x,axis=0)\n",
        "pred=np.argmax(model.predict(x))\n",
        "op=['dandelion','rose','sunflower','tulip']\n",
        "op[pred]\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "3svgMA4FZXzx",
        "outputId": "cb27b847-29cd-4b2c-a331-b876207853a3"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'rose'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 26
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "0MZjyRxUZQsw",
        "outputId": "54bb0e66-0381-4817-f1dd-7341fbe871ad"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=64x64 at 0x7FC61CE4A7D0>"
            ],
            "image/png": "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\n"
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img=image.load_img('/content/drive/MyDrive/Classroom/flowers/dandelion/33869330174_b259025135_n.jpg',target_size=(64,64))\n",
        "x=image.img_to_array(img)\n",
        "x=np.expand_dims(x,axis=0)\n",
        "pred=np.argmax(model.predict(x))\n",
        "op=['dandelion','rose','sunflower','tulip']\n",
        "op[pred]\n"
      ],
      "metadata": {
        "outputId": "9c36b473-9271-4a62-ea52-6b8854c95f57",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "VQVsGe-wbMBv"
      },
      "execution_count": 29,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'sunflower'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "8_uHNijobhqb",
        "outputId": "82e98a62-6d6a-49b1-95bd-f5a0a1343ddc"
      },
      "execution_count": 30,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=64x64 at 0x7FC61D611450>"
            ],
            "image/png": "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\n"
          },
          "metadata": {},
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img=image.load_img('/content/drive/MyDrive/Classroom/flowers/tulip/10791227_7168491604.jpg',target_size=(64,64))\n",
        "x=image.img_to_array(img)\n",
        "x=np.expand_dims(x,axis=0)\n",
        "pred=np.argmax(model.predict(x))\n",
        "op=['dandelion','rose','sunflower','tulip']\n",
        "op[pred]\n"
      ],
      "metadata": {
        "outputId": "06c3013c-2b36-4e74-e4c1-872ad665c720",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "id": "2TyDvcgFbMsM"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'tulip'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "img"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "diN2U59IbmL9",
        "outputId": "4518e359-0d9d-4e4e-98a4-155b310f9807"
      },
      "execution_count": 33,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<PIL.Image.Image image mode=RGB size=64x64 at 0x7FC61CD60510>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAZHklEQVR4nKV6a5Be2VXdWvvce79Hf91qqbullkYazdge5mHj14wHGxuTEGLAUASDgSqSQEjFqRSPCiQ/IClSJIRKkQflCpgKlaJ4VUGSAoJJAgYTJw4wwfgBg2c8I0aaGT1aUnerH9/7vs7ZKz9amoekGY3Dqe/HV/2dc3uvs/bZe529L/HqI8O/9+6X/2p83cOdJ+6v3V1EKwdAMhcBiJAk0gAKB3+h4FABTDO97fNZXIj9Xr73mM79UJydswpeOiJtwbwwdA+ht46FwyE77DHCv6DyIluhgQC21BzYEbaAXYaxfN9sVzgkPEZldwCg7K8sVynH2VF0t5t+TAQAAxIhuZNGSnJJRC3rknnkua+OK48i/0i7+pXo/sdi/+cXnvzl/YYMUldc/w5ky6JZey6lCTELswuxBiO8AZysoBkwAubGSAQPNb1y9uAA7wQghPaDoGL3Y4fHGAYySQDcSMmhg1kV4WAQAx2EhLkwhbryJVi8ivKj2Pkd3HsOxf3N2o80D74Lz/yE1v5e6B9O3TdgYZXDM/JN+qUwfTxGsqTXbg4nfAYMYTP6lIQ8UY2H0lIHIXm6A4CQaiwU5tmZH9s/ArZyAxNN7gAP5rRU66ypABYCyQSVxEiqwBbeF3pAHrHzDTj86xgcDse/lsWbIwOKQV4MkrunDLFGGkWn1XIHAHeiEWpgBi9puTvJykIdkwHBDnz8VUcSmr5zl4daRMgBgyIgAlIiAVRAQzaQEx3BJBFjYAhkCBPEw2CCgpB9/tDm/aO7fg3H35IOnYbB6hR7vTC8LP88F4KGV+QHzxc8MCXW9FoYA04QDA4YClhJ7+k1AHhfCPXcl+ahInIx0Wo4oAiAoDQHaqKW10ALdWgZ1ApjYEi0ijlZCjUp0DnuA+NvzbIPUN9Ssdf63UAvzb9gvJw0AWomUEQkYuIBktpIMXMAFD0KKc9CahrnnQG8o+2411FtEBw6MNSASEoEvDQMHSUBsQFMHkin5sAUNiczqIacDO4BIAjWWx+F/SYmKxh8B/bX1H5evRz+hDkkegTcAwDBI1ALFb0Qc8iIxtM8eYCNKehOAO5DG2bebKMyumtEVQBplbtTQWGuNII1ckAtYTABUZoRc3hJFq4akBBBB1oKYg5Lch9mex9Jchs8HPOe6UrWsmkPzhXUEBVUyVqJRiMLceg+B0patyh2mvJODASsulK0NvreYXX2MAZH1ExOhtIcQmM2TQAMxpRSgp9E+Aqgp95u3n7C42VyDtVAQ7jkglF9SECbWgl9eHenwEZTqW2NB1BBdzAJERJl6XqKiUAVOE/e1mVFAq+eB4TsZGx2Idnrv1/n/oUNhX1oJjRIShxZUkJJmOjJa8OHfPl+Gx5+Wz7aKo9ftr7xl2h7Si3gUEH2qMIheAZEAYCB2mhLR000rhZIsAhr4C0RpZYIogGEStIdtWmATB7vwIAB3QWWh+XueH51jN0xbBeYk3OwlebI58Gjy6g8yz98xN/+O+PPPpxv/XnMZB35KjBguugoTe5WECZlZAK6QAZ25ABqt1qpAmqwghyKVBIiUYPu17NNumFYIU4ZD76/GgA346qlSWQBYVhBc6YpMAdLYkyOTWVMLdAz/KgXb/6l+WMPr13DtTyiAxcwTdgFKrPCUQUO3QkUUgAE5FQPcGPpqQFqoCFbKVERCgCdydACmZCBDTSWxhZcPnMD/E5nQKFtPOtZmTw2gcKQmhF7rgZ2mbGMAhhod3n6in9dLrN4fqCtiNVafVkBf9Z4BqjlfTMmb4DWCKgACx1kEkRXBdREIyuVWjJJfuDuxlogEWiRahIactcVTY30gpu8sgulaGZK3iGc7amlOJH24GPDhmIlROjg82MGP8TJPP+6P7u2egI7sh1wy3iG2rUQhYm0T82AXfEauC9VYCIqWAVWYCnOmUqgphoLJdDiumpMLgCZY24shZZq/EUjM1yn4jYjh2JHKaElELW0XswnKYIzooFFCAdaKLOHvt/ve3vIT08/+557n3pu8zmgU5RvzWE1BjFNkQleQQ3VoSgcCzZN3gMifGqsHAmqBQcjrHWBcJjkDsFAoaWm1K5hLtYvMTL7W4ZPOK4aKOoGLwcj5LZ31ZdWwRYx2vLRzv4zky2hRGjgEEKGo7H/M+3Juz90iYeqnaN472cuvNechbXDcOV8/rl3tt+G/gezKZIcnIC1XORYcqAWg0FSQzrQCi0UpTl9IIsEyQilwFJikpQP2dYvN9LuC/ZV5PrtSJDU7bsEzxGaBHSnQAJKXp+dCX8V8/H9G2m9bHItJHT6DF1Lcu+mw29o3v9E0XYn7xYP7gkJXgr70hhqYQ00EeZiTavFGdnA5nAAM1OiKgpA3krSFBgrTT3c7Oen2z9+WA9/0LJTMHv5iehJ3ZxwwBE6ePKP9mdAyReI6vxED7/y24MP/K95N0O/38kddaV3D/xQzw6tH+XKD+8+x/s+vPCP1H1zBgMPPC6CI7EiatoQGJIz1xRoiTm8EisxulohOZ1IZo1rJmzDK96809myf1kOUPgq4LcMY38x3MYUcqVUhXwhpRByxJJoAAAk1zN8y98/WS1tNDWWzq9untvZ/Sz6Hj7bdIU562sn+eFfz8s3//ji4g+vfOyrz//++/1DshiDp3YvmCcKmpkgFpABM6EhHYzwjtgRImSykgoWqpQ2MyjqJgC2SCwAq8IDwNe7LZMv/FaDdYkYnW2Blqff12tfWCZ9pF0evG8jW+8vHeF8cTarYevwPvxLVg7m/AJKtDj3T8utnzifH8XXf5qf/Bt9WgJQuyeopSbCkBoGjshSmBL78qgQKQCJSARcY/cSNou3iZmWK/TN+sQx4EGG90h2A4IQvQQjmnHbWkolW1z3nwR8jLtn/iUu/VZTb/WmdVnTQhHyJT2xcgldADjybcce+cHDp345jo6h00WR86F/Uj3+7QW6mMEAb4WRbOoqXWNqQkxcDVAakqwiQGulFkjiHqzlzdsPwFqK7kHoEguKX8LsAV4H6kAI8MA2yUaq917mf7/BcHGAnU4sJ9nWp5Eu9kKe5h7rBYQKANa+aTz8jv29Zdz9zYXNCnN5E9a+u/qjD6xEoSLmZtfMR8SImDkmwhiYG0UmQyIi5FIERtQ2063WA7BSPgFLsnJ0pBXFR13rAMnIcG3POuumJVTi5NmXxl8MVc9/D1d+PTz54XH1KdSX6tFluGz91KIBZnb44erQ8QJlaAfN3lu6NurVTR2BYx/a/yF0xrQx9Lxw0WwirsIKYEROEUZKY9dIKGUToMqzLXIk3BTlrwOoyYqqoAahFBw4DD5IGwCBabTreYHFtdBZxbx9WaJQwFNFZ/qkhp/G+CxCFbhtnRJX6tgiixPGbdv88+aNF5amG73Un0/HZdazEAL3/Gve7nvGWpoBm8nvg98feDc0k0pPEyECpTCTT8Fx60Mo4jbWA7CGSmIrK5EiIKCAr8jvI92xCVS154veW7eYXuZCjPl649mup7OoPmPDq3V3IUfOU7+a3XdETz2G3bNsPstPHtl/wxsX5u8q0oNCjqaEOVb/bn1SYR9qhS8HH+3zge+Jp4m1EKZkKw6pNmQjakbbNe3cUtF5CQOO+kB4ECUwARqiMC7IT5tdhk1+JzQ5suXwx/nLlB/ZXuj4fzfUGWb7np7slJete5UnVsqP/UDa/4PI55AKHTl7/9U0X8jnCzJrQ6oRpwgBfy01NbiW4Rv7eOtP2ol3YfUR3pXSRD6ERsRWaneIkfmO7CWx/RYALawlG3oDlMAcnAi1K6ctwS+anx+1nW7esvndNvJGkCVDCHatTqccVUIkZhcaXdP+897+u0e3G6rldOrTZ9HtboROnfcWfK8Xq8TosWI7w/3flZ2mfgB41890Bw9S5NIj4QQgYEYNgRlsBttL2tErWg/AEtTKG6ICZ+S+4Ro5ycMciLBN2PPMNv4VMMF3ZvmL24/UieiDZeB+4BWFDdeV362W/ufSY7/9f3cucHJeqjzNEPNZdTmNN2Y+qb3yVGW6prSBI/d1PoDue/8djr6lpEV2cfT1+d3gW607FXZhE9q+a/JqchkALAMEuNAKJTF17RIbSVfcz0GXqPNIT7G9+jO4O7b/UPrGgA5QKFsjRtDFpDNRG8GFbLdjb50Oy3bddzzEIt/tHH8DBrMsrzF7juf/IG7/Hli07hYfw+6zs5WvbRbuRTMPXqPDkHfLtVW9Hq0D7j5SrKD6FZTySwEoAyiIaMg5MLew5X7FQuOaCBfFPfBZcJ+2ZPaIwveEIgu2Id8kHGGfsESHTnyf/9dvR5ebi2URalv46nr5m7H+oVSfRffn3hP/2bpPMfk4257ln8n0cRx6Y4hAnCY0dHfLePSRzikelMteEOt3BsAcyEAK7p6AlNIMiEoA6DpDbcrnxggFoZUvKn4L27zAGYCmGfkX8GVLa3VntQx1B4df33ZOEuey/T/m8Iru/RFk3/TTw2zL/wd2/kxpJ07odpXhslShmcEaZA66esvNIGH1Ndj9IoA+lNMyQwBdgIWh1NxQjgLo2ghWSi7l9GWp47rH9b6I48aJ+9wxNuTg3kI9eiidfhDhHnXVVLsaf077n8z3robdjz68/4bO0k8vNJuda7+Yu9C4siWih7qENyG60xGmOgKs82bN/GoAOkAfnguEG2EAGPwl7Am4fFDsp3WUhVX0gVXHo453ygSeD3o9cEjiNrqHkL0NK28u+p9ajMOUlTb7bDt+KoWzWPia6iPf2myt1vXYKxMClt7HlBsyVONolaFBsZUvGE4CfImmvAMAnrac1oFyWgAlpVvWTogMyGnsa2HFuidUdPM18h3UOw33I5xSiEL4Qmd+Jps9VUymWP5xxV10J4Vvh9kfgfudyU/joSyeWF9IxNyxrzyuRAR3Y53QzBQcPo5d4NjtJMMrjayqPRk6kX16H3DQblneupPoeMyOUn0MDluNNlQMQ7xOetD9QlCOYmezPpov7G2V3XuseUu19O1YXOu0/5j1Cez826zJsn6Zdp6coUZGrKnt2vI4H1oXLFAPlTe0sXJHP7xK4roFwP42FnoFUtMHlwDBZy+fEYCcVsC7tN5h5H2hr85xpj0tKkvB+2+yvT8c5I6FSdp7Zo4HQn0m1nns5LazOdqZgZ/B2OtJ1HNFeOiqn3i7Ro8LjrgwswmRiwWQoZ3IJkzQGlhA9e0NvnnYxDGqYtVhkjKxK+7rZT6UiLstLArWsRyuQhbRPZL1T6G4z1ffmy++Se/8we6Qo2rsxSjXn6OdIZ3rTjd8vo27v/u4J0SqyXg02cw9PJ73nXtA24/oyzKwgAJbweYZgI7r1J3y14sMjGj7DtTskJXBxev9kRcgiq9zZUTnkFnOIpeXyg63xTKWVlJvqeth1ubbegC9py0q5mPFpxFPJj3S47ged6/BuKDwrOLF1Hwp0ah1ak9YYD8uVpK3tWhKwtDbRPalB2TnXqnacxMDI/lUvgNdBIaORjcvM2gdUULIZFDeQadvi2tFt4tAhPVG96Dq6lt/+Y07StFBy3yK9koIT7Tj5x27i2HQqaUF4LghIyrTmEi0T9zfdorEAsUSfMDeKKOZpEgch2CviQWrgBk4BabERMrBzstnHCEWFHJAkmdkbZ2e572mczoL92A+bptNHPuy/k89cjkaK6BRApCm7fTZlM0YL9X5N1c9LBcKAxVzoAQyWSt9vhvNABNyZAPCGOSJCEIGZbcUIG4PYExEKNAad0C33noWYQY3QMNU1LCZlgahs4ImxpzWXeP6/Yd8Wt5938hM0byV3KyVN1MVz+TjvyjHn8Le6u6+p4uMBs6ERorQghcq2BtkIYeZ+7EUwTnMwA7QfW2x1IYMc6DwdATMjPGWJDCH1xSA3pzaFIa6tJ/O/pwdWcPSYt9GvdF4NDgdHv6UmljAkQh3z8CcbKL7tukZrO/0+4AJSaqEitbApk1lQUK0jlmH6Yi30hAOoAg4Ya8pH2dPKN0L9GiZVDl3zOqX6283S44Ij/TDl7POiKfe3pbvYXU+r0NafrRtlsC5uRUdNOL1ZqbgFBSjgAjusTwuW6E2nAkyphnDhQCWJ3uLVz3S8ug9M6Anrwwx4XXws7y52nkbBnaBCyFcEi4An6ef83TTinFShAzoihGJkzbEUAzvWX5//+lfKZViVmUxNZ2cwkGBFQQSIMJJP5DrUEsGqKHVxAxZK00T9z65CaasJ+tCA08HXQWXiAwsbldHuRnA1OwS/Bn444HniAo3E1eSCVqUHUiMnS7a/5PGn3i+e2b0jf/gaN1Za7qxGPQ3t9oaxcGSCNTA7Hr9HQ3Uylp5AkZIDXiesQJH1Nf9ncVuyDJm+SqQY+2DvQC0QAREv8vvrIgMyWPSeWIrOcVbk7iJtUIDr6BaKmqON7M3Ld5bH8KFw7tleS1jJzbzeserkCqgASMYhQhUYAVLDC0VAfPgQhQvydzSNsIpxDbkCk0vywfHw96DZU4bIExEF14HFK/awcCLDQ5BuH3hxZmeN8xoc8BpraQUn9p+/uzPAttW7GSgp4invwFlSnOgOSiBkInBZRFeyRvJoRr6UqCEhvAWYaI05KTa6OV96y8E9lJaRAOfwRuTk0tmJ+ivrkzvnCwIPkM/Dx+LV6EZeRkA/Gi1NN1vYx3r/baa2582aIHSNANGxFyK0JxpDjamBEQgUma2geyCsO+pp6xU/taH9upKLdrBSnfxzWGpWGkpOhagAfDuW1z6iwYgKSXsgVvQSNqV9g27i2Z/EpsNxH2m3ZA1djeQgCRLAIQaqHm9mzIWxkSEtVLr7pnGhm2zGeM1tDXQzZUF5ahWTij7pT0ABJbEgWsgP/Kqgeg1MECWsCBLRAsjSZpPfDvMt57CuNLogrPjX/n7KK2oTYmYBU6ImbwiSqiFJaCCUkAHdix5Eq4JUWhJJ4+ETsg4GGRrxxYfeOi0MvSBQGVQAZ3GX86FAMDSPjUkts2H0DR5nmWx7oSnAs7kjWPzsZy5Ddm4QgVWYoS1sgIWbzwjQvsJ7/jn/pDwvcJD8jGvM/yd91QkW8R5NWp1/kO/WwyAPpCBII/eekH5ogBIMsdVYgfYh20DW2Y7MW2yqZzpN7x+snf2w+nK73mDvFGKYJIE3wu+l6/1A6bySijR+97/PJi1eOi0PWD2CLLBja39zdHJinmh0O3a4nJ+9a6mW2AAMyiDFl+5rog7vq1yY/Nwwb178EYBuER9ISBLPIII5Nc+OkfC1X+T7bq6UtPR9/zsoey0V/Oy3NgBQ4q+erJYKcvZAJ0sm5ggN+LL8/Dx5iBrb+Ttg7PR0/kCkbt28cj7u2c+WjfQDKwo4hVT8msCAKC1bKRk4phKiQlqA/cTdtgOEUZmn1D7NuBrT+NNP4vdzdH8MeaNsoE1e57n3L5an626q28Lvjlvy+umnG6i3XiB4MGVpz/3LN1CHnx9nbvfl+y/qesY4UDfv+IxeK0AXHEHuGY45liCHJlS3Cuw2thmShnSCPmP3t22P4T0ZDZ8LmZVKLPInN7Cc2t3U/+uqj4HXc05cyBRWkS4y/ySoIMuRlcrc1bJLeMb7tXjzpxqhYLomKpXuCW/1qsbwQZWCZPcLodwyewKgJbjTp4BMxTflvHT27b5HwZ7H8+nnyuqzTC7nHWG/eXUG+wt51U2exbt1ay+krWTBAWCkem0SwgAGuCd99h0mJRhPsZomlagVkhELpx45Tv+a757AhHKSLntpTT0doDONjVLPmb2afivOT7SZqMvtDt/WKYtTPdidSHufGG2c7mcTGazJ/2Qd8uzYfZcVYI1vYQDWCdxo3fUbd33snbPDh3OlhezB38Ki7BEiDhif2kXkkSylU0UM2AcNGLzugiEEFE/Cn7asWHY9NRtbeVC226xuxbml1LasenVarDUP//xeWyZORLVAC2RhEgtOcYAAEdebTQxGIsIYP0dR/8E241QAKvg66lnb3eMvwgGJEmpBSuwcc6TLmUYtVUnw9UeVkPshPgJ44WEOYDKm0tJNL+CRrY3qusYkmMGVUIFNEACTDx14/kN2nf/7cP1zsLootqozWvbLdY6skPCMekeAsxvteqLAPACDD+Q6cxK4WIGb3m15Lp1ozK5X6U/TV1jqJlVwj69Blv5NKWGaoE5rRQr8ADDAOEFubOKXnyMsUQIdlTHVrDbB48AS7LDbj21t9rzRQMAAN1gQ3BZBb8r6KlYdYUWZpbNmD+b2RnGffD5wAl8DNXGCTkiZ/KamJAt0JiVSOmGGbu8+lU/We5vsLrW+5Mfv7aJsGl5BuRIgTh+O1te6xm47XBxAlXGveQdoAzxObFBovFi9DcBlwKX3QbwYKxchdmcXogNue1astB4qoAMfEFxdNE2f3qMJyf/5TOeLfp62dZCLjsiW0N4DjeT8P/FAIDrR0Jw7Xg2M0sGtRhRO8muiIeIvxDPJTxB/AGzK44x8En3pHwWUAELxB5QEgwhvxGIJFXAd/3C1uh/z9+Y9R+f4Bcje7IcGCFevcV6/CUZOBgjRrlaEoCSNpGyhH3QgDFiEbGYpQHzWvGYdI3JkkhNhDqkFNmShYUyvRjqK+Cv//wyfPo3rfhPVFRy9zFglsHjTf/9/wFTIpd+AivWxwAAAABJRU5ErkJggg==\n"
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = image.img_to_array(img)\n",
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7ALTeLEZi6e3",
        "outputId": "1875b7a6-e035-4433-e519-80608266b7cc"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[[ 42.,  59.,  14.],\n",
              "        [ 46.,  61.,  18.],\n",
              "        [ 52.,  67.,  24.],\n",
              "        ...,\n",
              "        [ 39.,  60.,  21.],\n",
              "        [ 35.,  56.,  15.],\n",
              "        [ 37.,  54.,  22.]],\n",
              "\n",
              "       [[ 36.,  55.,   9.],\n",
              "        [ 44.,  61.,  17.],\n",
              "        [ 54.,  71.,  27.],\n",
              "        ...,\n",
              "        [ 42.,  60.,  20.],\n",
              "        [ 40.,  57.,  15.],\n",
              "        [ 40.,  56.,  20.]],\n",
              "\n",
              "       [[ 39.,  59.,   6.],\n",
              "        [ 37.,  58.,  15.],\n",
              "        [ 43.,  60.,  18.],\n",
              "        ...,\n",
              "        [ 45.,  64.,  19.],\n",
              "        [ 47.,  66.,  21.],\n",
              "        [ 44.,  60.,  23.]],\n",
              "\n",
              "       ...,\n",
              "\n",
              "       [[ 79., 101.,   0.],\n",
              "        [ 96., 121.,   4.],\n",
              "        [100., 121.,  26.],\n",
              "        ...,\n",
              "        [125., 109.,  58.],\n",
              "        [ 88., 111.,  29.],\n",
              "        [ 94., 115.,  10.]],\n",
              "\n",
              "       [[ 69.,  84.,   0.],\n",
              "        [ 77.,  97.,   0.],\n",
              "        [110., 135.,   8.],\n",
              "        ...,\n",
              "        [ 98.,  53.,  11.],\n",
              "        [130., 106.,  60.],\n",
              "        [123., 147.,  35.]],\n",
              "\n",
              "       [[ 47.,  68.,   0.],\n",
              "        [ 38.,  59.,   0.],\n",
              "        [ 89., 111.,   2.],\n",
              "        ...,\n",
              "        [ 71.,  61.,   2.],\n",
              "        [ 93.,  62.,  15.],\n",
              "        [116., 100.,  38.]]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x = np.expand_dims(x,axis=0)\n",
        "x"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f4UgXmORjayI",
        "outputId": "2f45a200-58e6-42a9-9369-61730cb6e95f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[[[ 42.,  59.,  14.],\n",
              "         [ 46.,  61.,  18.],\n",
              "         [ 52.,  67.,  24.],\n",
              "         ...,\n",
              "         [ 39.,  60.,  21.],\n",
              "         [ 35.,  56.,  15.],\n",
              "         [ 37.,  54.,  22.]],\n",
              "\n",
              "        [[ 36.,  55.,   9.],\n",
              "         [ 44.,  61.,  17.],\n",
              "         [ 54.,  71.,  27.],\n",
              "         ...,\n",
              "         [ 42.,  60.,  20.],\n",
              "         [ 40.,  57.,  15.],\n",
              "         [ 40.,  56.,  20.]],\n",
              "\n",
              "        [[ 39.,  59.,   6.],\n",
              "         [ 37.,  58.,  15.],\n",
              "         [ 43.,  60.,  18.],\n",
              "         ...,\n",
              "         [ 45.,  64.,  19.],\n",
              "         [ 47.,  66.,  21.],\n",
              "         [ 44.,  60.,  23.]],\n",
              "\n",
              "        ...,\n",
              "\n",
              "        [[ 79., 101.,   0.],\n",
              "         [ 96., 121.,   4.],\n",
              "         [100., 121.,  26.],\n",
              "         ...,\n",
              "         [125., 109.,  58.],\n",
              "         [ 88., 111.,  29.],\n",
              "         [ 94., 115.,  10.]],\n",
              "\n",
              "        [[ 69.,  84.,   0.],\n",
              "         [ 77.,  97.,   0.],\n",
              "         [110., 135.,   8.],\n",
              "         ...,\n",
              "         [ 98.,  53.,  11.],\n",
              "         [130., 106.,  60.],\n",
              "         [123., 147.,  35.]],\n",
              "\n",
              "        [[ 47.,  68.,   0.],\n",
              "         [ 38.,  59.,   0.],\n",
              "         [ 89., 111.,   2.],\n",
              "         ...,\n",
              "         [ 71.,  61.,   2.],\n",
              "         [ 93.,  62.,  15.],\n",
              "         [116., 100.,  38.]]]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 34
        }
      ]
    }
  ]
}
