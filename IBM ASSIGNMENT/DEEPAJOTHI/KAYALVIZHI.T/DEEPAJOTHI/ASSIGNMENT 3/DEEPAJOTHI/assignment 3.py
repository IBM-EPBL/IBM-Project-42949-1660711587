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
        "LOAD THE DATA"
      ],
      "metadata": {
        "id": "BKLaU35W7Nod"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import sklearn\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "df =pd.read_csv(r\"/abalone.csv\")\n",
        "df.head()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "WpK8MZQicU23",
        "outputId": "479c957b-15df-4ff3-d92a-a08e7d1e878c"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  Sex  Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  \\\n",
              "0   M   0.455     0.365   0.095        0.5140          0.2245          0.1010   \n",
              "1   M   0.350     0.265   0.090        0.2255          0.0995          0.0485   \n",
              "2   F   0.530     0.420   0.135        0.6770          0.2565          0.1415   \n",
              "3   M   0.440     0.365   0.125        0.5160          0.2155          0.1140   \n",
              "4   I   0.330     0.255   0.080        0.2050          0.0895          0.0395   \n",
              "\n",
              "   Shell weight  Rings  \n",
              "0         0.150     15  \n",
              "1         0.070      7  \n",
              "2         0.210      9  \n",
              "3         0.155     10  \n",
              "4         0.055      7  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e0f586fa-2f51-4001-a40c-eb2341fdd93f\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>M</td>\n",
              "      <td>0.455</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.095</td>\n",
              "      <td>0.5140</td>\n",
              "      <td>0.2245</td>\n",
              "      <td>0.1010</td>\n",
              "      <td>0.150</td>\n",
              "      <td>15</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>M</td>\n",
              "      <td>0.350</td>\n",
              "      <td>0.265</td>\n",
              "      <td>0.090</td>\n",
              "      <td>0.2255</td>\n",
              "      <td>0.0995</td>\n",
              "      <td>0.0485</td>\n",
              "      <td>0.070</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>F</td>\n",
              "      <td>0.530</td>\n",
              "      <td>0.420</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.6770</td>\n",
              "      <td>0.2565</td>\n",
              "      <td>0.1415</td>\n",
              "      <td>0.210</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>M</td>\n",
              "      <td>0.440</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.125</td>\n",
              "      <td>0.5160</td>\n",
              "      <td>0.2155</td>\n",
              "      <td>0.1140</td>\n",
              "      <td>0.155</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>I</td>\n",
              "      <td>0.330</td>\n",
              "      <td>0.255</td>\n",
              "      <td>0.080</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.0895</td>\n",
              "      <td>0.0395</td>\n",
              "      <td>0.055</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e0f586fa-2f51-4001-a40c-eb2341fdd93f')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e0f586fa-2f51-4001-a40c-eb2341fdd93f button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e0f586fa-2f51-4001-a40c-eb2341fdd93f');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "UNIVARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "oNagCnv57w3K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.hist(df['Length'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 352
        },
        "id": "oEkkM1Tt819W",
        "outputId": "75492939-7c5d-4765-a0ee-29d87083aa48"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(array([   7.,   60.,  147.,  304.,  460.,  778., 1051., 1017.,  324.,\n",
              "          29.]),\n",
              " array([0.075, 0.149, 0.223, 0.297, 0.371, 0.445, 0.519, 0.593, 0.667,\n",
              "        0.741, 0.815]),\n",
              " <a list of 10 Patch objects>)"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "BIVARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "gVX2LoNU74e2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(df.Length, df.Height)\n",
        "plt.xlabel('Length')\n",
        "plt.ylabel('Height')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "x41QPzxw86AO",
        "outputId": "7320829f-6ec7-4459-f3d4-f3e685799a34"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Height')"
            ]
          },
          "metadata": {},
          "execution_count": 14
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.stripplot(x='Diameter',y='Height',data=df,palette='rainbow')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "OzzSG4SN_ARA",
        "outputId": "48cf6890-1258-4053-d416-dc32fc247dc9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f0df2ea5fd0>"
            ]
          },
          "metadata": {},
          "execution_count": 18
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYsAAAEGCAYAAACUzrmNAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOydeZhcVZn/P++tql6zbwRIQhIIS0RACKusioqogIgoKIoy4ooyM24z+nNmdNydUVRQcRd3dMQoKAgiiGwJOySEhOx7Z+29q+re9/fHObfvrUpVV3eSSjrk/TxPP33rveeec+72fs9+RVUxDMMwjIEI9nYGDMMwjOGPiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmmT3dgaGyoQJE3T69Ol7OxuGYRj7FI888sgmVZ24s8fvc2Ixffp05s+fv7ezYRiGsU8hIit25XhrhjIMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoaxkxQIWc5Wusjv7awYRt3Z54bOGsZwYBXb+AmP0E2BDMLrOZqXcPDezpZh1A2rWRjGTnA7i+imAECIcivPUiTay7kyjPphYmEYO8F2ekt+91AgT3Ev5cYw6o+JhWHsBC/mwJLfhzGeFhr2Um4Mo/5Yn4Vh7ATnMosWcixmEwcyirOYubezZBh1xcTCMHaCAOF0ZnA6M/Z2Vgxjj2DNUIZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyZ1EwsR+YGIbBSRp6vsFxH5uogsEZEnReT4euXFMAzD2DXqWbP4EXDeAPtfDczyf1cD36pjXgzDMIxdoG5ioar3AlsGCHIh8BN1PAiMEZED65UfwzAMY+fZm30WBwOrUr9Xe9sOiMjVIjJfROa3tbXtkcwZhmEYCftEB7eq3qiqc1R1zsSJE/d2dgzDMPY79qZYrAGmpn5P8TbDMAxjmLE3xWIu8DY/KuoUYLuqrtuL+TEMwzCqkK1XxCLyC+BsYIKIrAb+A8gBqOq3gduA84ElQDfwjnrlxTAMw9g16iYWqnpZjf0KvL9e6RuGYRi7j32ig9swDMPYu5hYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDWpq1iIyHkiskhElojIxyvsnyYid4vIYyLypIicX8/8GIZhGDtH3cRCRDLA9cCrgdnAZSIyuyzYJ4Ffq+pLgDcDN9QrP4ZhGMbOU8+axUnAElVdqqp54JfAhWVhFBjlt0cDa+uYH8MwDGMnqadYHAysSv1e7W1p/hN4q4isBm4DrqkUkYhcLSLzRWR+W1tbPfJqGIZhDMDe7uC+DPiRqk4BzgduEpEd8qSqN6rqHFWdM3HixD2eScMwjP2deorFGmBq6vcUb0tzFfBrAFV9AGgCJtQxT4ZhGMZOUE+xmAfMEpEZItKA68CeWxZmJfByABE5CicW1s5kGIYxzKibWKhqEfgAcDuwEDfq6RkR+bSIXOCD/SvwLhF5AvgFcKWqar3yZBiGYewc2XpGrqq34Tqu07ZPpbYXAC+tZx4MwzCMXWdvd3AbhmEY+wAmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqYmJhGIZh1MTEwjAMw6iJiYVhGIZRExMLwzAMoyYmFoZhGEZNTCwMwzCMmphYGIZhGDUxsTAMwzBqMiixEJG7BmMzDMMwXphkB9opIk1ACzBBRMYC4neNAg6uc94MwzCMYcKAYgG8G7gWOAh4hEQs2oFv1jFfhmEYxjBiQLFQ1euA60TkGlX9xh7Kk2EYhjHMqFWzAEBVvyEipwHT08eo6k/qlC/DMAxjGDHYDu6bgK8ApwMn+r85gzjuPBFZJCJLROTjVcJcKiILROQZEfn5EPJuGIZh7CEGVbPACcNsVdXBRiwiGeB64BXAamCeiMxV1QWpMLOAfwNeqqpbRWTS4LNuGIZh7CkGO8/iaWDyEOM+CViiqktVNQ/8EriwLMy7gOtVdSuAqm4cYhqGYRjGHqDW0Nk/AAqMBBaIyMNAX7xfVS8Y4PCDgVWp36uBk8vCHO7T+QeQAf5TVf9cIR9XA1cDTJs2baAsG4ZhGHWgVjPUV/ZA+rOAs4EpwL0i8mJV3ZYOpKo3AjcCzJkzZ9BNYYZhGMbuodbQ2Xt2Ie41wNTU7ynelmY18JCqFoBlIvIcTjzm7UK6hmEYxm5msKOhOkSkvexvlYj8TkRmVjlsHjBLRGaISAPwZmBuWZhbcLUKRGQCrllq6U6diWEYhlE3Bjsa6mu4WsDPcbO43wwcCjwK/ADv8NOoalFEPgDcjuuP+IGqPiMinwbmq+pcv++VIrIACIGPqOrmXTslwzAMY3cjgxkNKyJPqOqxZbbHVfW4SvvqyZw5c3T+/Pl7KjnDMIwXBCLyiKrWnB9XjcEOne32k+cC/3cp0Ov3WYezYRjGC5zBisVbgCuAjcAGv/1WEWkGPlCnvBmGYRjDhMGuDbUUeF2V3fftvuwYhmEYw5Fak/I+qqpfEpFvUKG5SVU/WLecGYZhGMOGWjWLhf6/9SgbhmHsx9SalPcH///HACLSoqrdeyJjhmEYxvBhsJPyTvVzIZ71v48VkRvqmjPDMIzhSOc2+P234Iefgsf/trdzs8cYyqS8V+FnYKvqEyJyZt1yZRiGMRxRha9/EFY9634//Gd4x6fhpPP2br72AIMdOouqriozhbs5L4ZhGMObNUsSoYh54I97Jy97mMHWLFb5z6qqiOSAD5F0fhuGYewftI4CCUCjxDZy7N7Lzx5ksDWL9wDvx32jYg1wnP9tGIax/zD2ADj38uT3iDFw3pV7LTt7ksFOytuEm8VtGIaxf3PxB+GU18DmtTDrBGhq2ds52iPUmpRXcTJejE3KMwxjv+SgQ93ffkStmkV6Mt5/Af9Rx7wYhmEYw5Rak/J+HG+LyLXp34ZhGMb+w6CHzmJLkRuGYey3DEUsDMMwjP2UWh3cHSQ1ihYRaY93Aaqqo+qZOcMwDGN4UKvPYuSeyohhGIYxfLFmKMMwDKMmJhaGYRhGTUwsDMMwjJqYWBiGYRg1MbEwDMMwamJiYRiGYdTExMIwDMOoSV3FQkTOE5FFIrJERD4+QLg3iIiKyJx65scwDMPYOeomFiKSAa4HXg3MBi4TkdkVwo3EfXnvoXrlxTAMw9g16lmzOAlYoqpLVTUP/BK4sEK4zwBfBHrrmBfDMAxjF6inWBwMrEr9Xu1t/YjI8cBUVb11oIhE5GoRmS8i89va2nZ/Tg3DMIwB2Wsd3CISAP8L/GutsKp6o6rOUdU5EydOrH/mDMMwjBLqKRZrgKmp31O8LWYkcDTwNxFZDpwCzLVObsMwjOFHPcViHjBLRGaISAPwZmBuvFNVt6vqBFWdrqrTgQeBC1R1fuXoDMMwjL1F3cRCVYvAB4DbgYXAr1X1GRH5tIhcUK90DcMwjN3PgN+z2FVU9TbgtjLbp6qEPbueeTEMwzB2HpvBbRiGYdTExMIwDMOoiYmFYRiGURMTC8MwDKMmJhaGYRhGTUwsDMMwjJqYWBiGYRg1MbEwDMMwamJiYRiGYdTExMIwDMOoiYmFYRiGURMTC8MwDKMmJhaGYRhGTUwsDMMwjJqYWBiGYRg1MbEwDMMwamJiYRiGYdTExMIwjN1GFOXpbX+WQu+GvZ2V/Zcli+HPt0FHx26Ntq6fVTUMY/+h0LuBTUu+TVTsBGDEpLMZfdBr9nKu9jO+/AX47/8EVRgxCqZNh6eeglNP5Sho2JWoTSwMw9gtdGy4q18oADo33sOICS8l0zBm6JEVe2DVfdC7DQ46GUZP2zFM+zpY9nfINcHMs6Fp1E7nfUhEIcy7HVYvhtmnwFEnO3vbWrj395DJwtmvhzET9kx+YrZsgS9+1gkFwJatsHmr2/773/lxNpixK9GbWBiGURFVZUvHI3T1LKOpcTITRp1CEOQA6Nj+OJ2dC2lomMCYcaeTyTSXCIWPgbDYRaZhDPltC8hvmofkRtE8+RyCxh0FRLcvJ1r1N0AINi9GOta4HUtvh1M/CuMOTwJvXwO3fwKKfe73krvg/C9DdpcKz4PjZ5+D+//gtu/8GVz+b3DUKfDvl0JXu7P/5Vfwxf9zpfvBsPAJ+O1PIAjg0nfCYUcNPV/bt0E+n/zW0t0vEVqHHmmCiYVhGBXZuO1vtG37OwDt3QvpzW9g2qRL2L71Ido2OmfZBfT0LGfKtHfRMu5E+joW9x+fzY6hc+XvIMxD58p+e37zY4w57j+QINNv064NhPO+DFEBoggphklGNITldzux2PI8PHMLbF6aCAVA50ZY+yhMO2XoJ7ptA9z9Q9i8Eg4/DU6/DFJ5K6GnEx64NfkdKfzqaxA0JEIBsHUjPHwHvOwSWPU8/Ow62LQBzjgfLngbiLjf3/0SPPsUPP8chP6c//Rb+NXf4KBpsGo5fOKD8Pg8OO5E+OzXYer0ynmbMRNOfSk88A/3WygRjMeUrhOHfnX6EVWtHWoYMWfOHJ0/f/7ezoZhvOBZtPJrFELvAFURVXJBK0GhG7RQEvaQGf9CJtPC5qU/JN+9gkCaCPKugzUThjuMpGmZ/iYaJp5EuOJPFNfeRxAWyfT6tCIlWywmgUOFoBFyI6FrM4QFCKOkuQWgGEJ2JDS0wnGXwGFnOvuyefDgT6C3AybPhg1Loacdjn4lnP42QOA774KNy5O4Tn8LnHwxtI52Drx7G4wcDz1dsGIhXPe+JGxvwTnkYghR2UmeczFc+e/w9jOgc3v/uTFpGmzbAn156Oxw5xKW+eE5p8M3fgFvuwDmP5Cynwo/mVvhbnm2bYNvXuc6uU88GX73f/DAA3Dqqcy+629PLVA9pvrBA2NiYRhGRZ5f8z168msBkCgi8L5iB+cvGWYc+nHa195K9+aHAAjCiMC7lpLwqmSLIYGCSI6g0NsffzZ2mGFENi5lR5o40kidY/Xx9G+HERRSNREEjr8MWsbDnddBVIQogkKZN59xCkw+HO78XmJLO/2gEXq6XR9FiIujGCal9TCEfCoP5Q7/6FPgZZfC5z6Q2HoKSf4LAxw7ehx881dw8TmQFs5sFp5cx4D09MDS5+HwIyCXS66KyCOqOmfgg6tjQ2cNo460080S1tFNX+3AewDViG35lWwvrK0Z9oBxLyeQ3A72MAhSrRvCuPEvI5Nppq/z+SQdkf7ttIvOpEQk3YykIv3hRFNHRGknmtoWcX9IaZgogkIRHroJ7vqqEwoodcbFEPIhLPoH3P391ImlhCIMobsTNHLOPCqvNpTFWanQ3bYW1ixLha8QR6VjwwjWb4RLzoFMWXPYcTUakv5yOxwxHU49AV40C+bPGzj8ELA+C8OoE0+ygj/zOIqSJcPFnMwMJu21/BSjPp7cfjOd4UYAxuSm8eJRFyMS0F1ooz2/ktbcAYxsmALAiOYZHDH1Wrr7VtPTs5LNW+91EYlQDLIcPOkimpunkmsYD0BD8xR6+jYBoAJKgBChQQDZ0ZDfSlDF+Yu6JvZ4u0KQUlFQTTnZlD3twBO9KnXIaZ+djj9dOSmmBWsQ+SnXgWIIq5bBb29MhUkFSolpSZyqSW0FnNiNHgNdnUmfRTnLl8E174aHH4BIoM+L8Pp18LF/hbvu3fGYnaCuNQsROU9EFonIEhH5eIX9/yIiC0TkSRG5S0QOqWd+DGNPEaH8jWdQ7wmKhNzDM3VNsz3cxMK+B1maf5KC5nfYv6HvmX6hANhWWMnm/FI29SzgqU0/YEX7nSzY/DPWdNzfH6Ynv47OnmVkc2MYNeIYICCTaeXAAy5k1OjjyDWMp697JVvX30am5SAyDW64qJClv64gQlhoh2zZYJyUkwyiqN+3p2sl1cJXdf5pwirOuSTOKs3wg2mdrxYmXUPpTk2MkyrutkrW3D6BCRNd09N/fx0++gE4YhJc9lpYudyFuebd8MB9rrmqr6wGu/T5HaLcWepWsxCRDHA98ApgNTBPROaq6oJUsMeAOaraLSLvBb4EvKleeTKMPUVIRC+lncBddWyK2hKu54Ge36PeS60sPMUYmUhWGpjRcCytmbHko56SYySKWN15P8XithL72s77yRc2ke/ZTCG/yhlVyQUjyGZayWZH0dQ0FYCejmdpW/EjFyaMyHjnK2Fy7v19FsUuFFKiQL/DlTLhqEl5LaMS1WoB1cKkkQH21QqTrqFkgqRGkT7JavmpxKlnuf8f+wA87IX84fvd71/8EeY96OOvoDqvu2Dw6dSgns1QJwFLVHUpgIj8ErgQ6BcLVb07Ff5B4K11zI9h7DFyZDiSg1nI6n7bi6kwsWyIRBqxMHyCNeFKRspojsmeQGswkhWFZ/qFgiikm6304CZkrc0vZKxMAI0QApSov8O6q7iBTKpUjypBlGdr99PkCsWkaSiKiKJ2IqBY2M7KZdeRkSzZfF9/mKCK05Yq24Mqve8t0nmrJgqDyb9Icvxgwpd3dEMyVPbRh1NpK9x/H8wYD62tsH17ZdE862WDSHRw1LMZ6mBgVer3am+rxlXAnyrtEJGrRWS+iMxva2vbjVk0jMqEGpHXcOAwRHTQ19/UVM75vISzmM0sDuRVHMsZHEVei1QagZjXAhujTcwt3sFPi7/l7+FDFLVIQQsl4RcWn+Dp4qNs1U2sjJ7nrvwf3f5UmIB0+78iUYHt4Tq2RxsoBAHjc4eRo7E/fDo3gWrFVpG0LaNKgKJaoFob0EAtK+l87hPsiqhFUXJ8kD5jrbhZkRbffHf8SYmtGLnaSbEIvV0wenTluSH/d/NOZLoyw6KDW0TeCswBzqq0X1VvBG4EN3R2D2bN2A+5W1fxJ11OnoiTdTKXyiwyZe3Nf2cpd7CYCKWBDG/jeGYwviTMerbxBCvYRhft2s1CVrCOLYygmXOiFzNWRpDVgLv0QTbqZhqgX3gWRUvYEK2hm05aaOHkzMlMDaayJFxYkkYP3SwvLqYr8uP4y4RIIi2rNYRsjpaQSQlhaT9w7ddrMEKwzxNQvS9kKKRrCkGqlpGurqRrLhmBYur39EPhvIvc9he/6ZqeHn3YD7vVJN4oD/c/Cie8uDT9SbtvQEU9xX0NMDX1e4q3lSAi5wKfAC5Q1eExvtDYb1mrnfxOn6eXkAjlAdbxV1byjLbR7SeiddDHn3mOyL+seUJ+zuP9cXTTxyLWMZf5bKMLgE1sZR1b3H7t4nbu55d6B7/S29no7ekaSoaIbjp9fN3cW7iXR/IP00uq30GVTBTxWOEetqrvuC5rty6pEURR/ws/UFO9QVktYBcov9CNWWjIwJXXJrZMKi0RmDYjmR8RRbB1s9ueNt31USzaCCedVhrvCSfDrMPh3akJg1OmwrUf3j3nQX1rFvOAWSIyAycSbwYuTwcQkZcA3wHOU9WNO0ZhGHuWVZSubxQQ8WddAgINZLhKjwXZscjZTYE8RVaxhd8wjyIh2VTRNF26z6TsIaHfpyWdv5L2MlGEEvFs9EzJCxuoDqm01z87Il17KKtJ7DNNQ/VmKB3Q5WRIOrnLaygi0NICV3zQjda66XooFmDOKfCW97sRUx+6Egp+gMDKZXD9F+HzN5Sm8T/Xw7++Hx55yAnF/1zv7F/+Klx1NaxbC6edDo2N7C7qJhaqWhSRDwC34y7fD1T1GRH5NDBfVecCXwZGADeLKxGtVNXd131vvODp0ZA7C5vZrAVOz47lsEzLLsV3GKPj2QG4mQJRvwfPE3Ibz3M1x5UdpTQScDfPsZh1FImI3X3s/CPcSwBlzT79v6VEINJh+vsgRNDUfISajr1K/0NJ/FraTGUMgYxU7pDOZNzIp2LkmoiU0mt77hvc/yuvhde/Hbo63DpQAMsWQ2/pqDVWLd8xjUNmwG9uq5yvI49yf7uZuvZZqOptwG1ltk+lts+tZ/rGCxtV5VM9S1gUdQNwS2Ejn246jGOyI3c6zvHSzDuYzZ90OT0U6ZSudIpspINf83TJMRm3OBAPsYyAZGRRSEBARI6AUbQygixtfoRSTL+jFiGKawolM5irNJyn7aqV7WnKhKBfpIIATY+GqhJ+v2Mgle3vXwiSxf/KR0wFAk0N8JW5cO3r3WxwgIYmeMM/JeFGj3V/MdMPg5mHw9LnEtvLXr1Lp7K7sFqnsc+yJOruFwpwpffbCpsGfXyb9vLdcBH/XXycP0YribyDPFYm8vHgRK6Wo0vCByhFKbKA0q/AZar0AAjO+YdEbKOD9WwlYuARVsmxFbZ1MEJQpYmpZMbwIMJUm8S2v5BeZiOTuhaZKi4zW8H+2ivhgb8kQgGQ74UH76yergh886fw2jfCMSfAhz4JV75/SFmvF8NiNJSxf1JQJURpqjazFYhU6SWiRXYcFthQoazTOAgnV9SIzqjI1/UZ1vsO4xVRJwicwDg0gJt1MWtoL/HatUtWZSORSjqsE2u17uWklhGgmprRTIWCrkjt0n+1Gc/7C9WaiaqR7l8IxHVER8BpF8ODf4LezmTeRCYHuQy0NMK2Tc7ekIMJU6CQh7Mvgte/G277+Y7pNDQNnI+DprrZ2sMMEwuj7nSEEeuKRQ5tyLE1DOmIIu7q7eSHHVsoqnJx62g+MmYiQZmjf6zQxZc617AhKnB0toX/N2IKEzPJwnaHZJo5LTOG+0M3A7mZgItyAw8VvCVcydxoNaBkJN1HEHGbLuc2WUYmUu8ToioviKBU6g/QqjWCxPGnm4+iKn0HpTWIJE6tHL4kcFC5BqLV8jYIZ/pCEJpqGp0NknWggrJFCYPAfe3u4n+B894Fv/0qrHoWVi1xndIUQHvg1W9zH106+VUw5bDS+M98Lcz9Maxd4X5PPRROP68OJ1h/TCyMunJLewefadtMjyqjM9BDhEpENvXk/bprO0c3NPGa1uSrYqEqn+tYzWZ1q4Y+Xezmf7rW8PrmcRyTbaXZ1zQ+2jSd3+Q3sCjs5M2NBzI2yDGvuI2ZQQvjg+SraV1a5L6wjbmazKhWjVtb1JX8vQeNNUtLwlRdTa5K83blzuqB6j2lo6Fqk4SPKgoTUrrCRH/48qanWmKwLzVJlYw+SilELuNWmgXIChTjOQ4CYybBwbOh0Aeb1sH61HpKF33E/W8ZCVd8Cv70AycW/UkIjBoNr35n5fyMGAXX3eKaniSAk18OjTVqFsMUEwujbnRHEZ/1QgFKl39xgx18T8R3OzcxP9/JFSPGMzPXyJao2C8UACIRj4QdPNLZwSgJeE3TWDZEeVZHXazCLZo3v6edhiAiwvUvnJIZSyBwVDCCm8NVFCikmpY15QNLt9PEdqHUZ1Yc4lp2TkNx/OUCVBMRZCjNUCKJcJQJxHCQgopNbTDwMNRKYTL+x4gDYcxB0DQOFtzhDjziVDjkRAiLcOTZ8OTt0DoeHr0VnvmbO37yoXD5p2HTKpjzWmgp+yzqIRVGGR0ye+CTa2yGs143cJh9ABMLo25sDkO64oXlqoZSsgGsjwr8qbfAn3s7uG7sFE5sbKGFgG7cTNX08V1S5Df5Nn9sql9AopIm53m6FRTmRZsRKe1zqDmkFJJVUyvmevBONj1sdiCG5LQ1PS8j1SwWZNAorBhXyQioMAlTscZRp6anatcizAaIbw4qCRMELi8RkMnCjDNg5DRoWwQr/OdDczk39yUEUDjyHDjtatevAHDyZU4gRk4sTfSkN8Cz98GyxxLb+uehYxOceTkVmX0qnH8V3OX7Il5+OczeiU+57oOYWBg7zZp8kW9v7GRTMeSScS28fFQzAM/35fnh1nZ6ImVaNsvKYrGkSSeKhCBwpflAtHSgDnB9Rxufyx3khcKRlPCj/pqJVC2NV64pVBkztMO8h9I9lcWuWq2hlojsqgsealNVcuAQU96NYlHEORoBokxAEEb9VzY9dLiYyxBMPJ5MyyGw4BdJBBmBI14Lh18IgXdZYV8iFuD6Ho56FRz/lkQkYlrGUpWubTvauivY0rzu3U4wwAnYfsL+c6bGbqUvUl6/uI01/nOWc7f1cNPM8RzX0sCbVq5nux/vnxUlF7hGmcOyOXqI2BqFZCWkt0rceY3YGKW/x6D9QlPtmzGSEohSJ5rYq41mquZ0qy0QONCxleKSHX7tvCMuFzYB36nt22HSTj4QNKo+MW+H8CUJDb1xSlN9JCVCkMlQBEYfdhVB+0ryK/7okqgQR2bS8QSjDoPFv4eCHxYd5GDKaYlQgPvWdjktY3YUiloc+VK46/vuG93gjj96ECu17kciEbP/nfF+QD5Uigot2V1rje6LlEjhnvY+PrOqgy3FiMsnNvORg0ZwT0dvv1DE/HhjB6vHNfYLhaBI4EqWAIuLeV8riMjGtQzFf48Zf0zESu3jA1tW0Jh6Ovt9V5lvi0VkMBOGqotC7fDVwgzG7ZeEkQyqxRq1ggxKhWaislFYNe+upIRJpHJm0/0XOzHPorx2kMTFDunlmicTdtb4frRkoWk0vPSTsOxO96W4Q85xfRBppp4ET/0Wuv26SQ0jYPoZg8pzCa1j4Z3XwbzfQzEPJ7wWJh4y9Hj2A2Qwq0wOJ+bMmaPz58/fY+mtW600NMH4CcOhG7A2Nzzby2ef7KWnCG+e2cB1JzWT8+02SzpDmjLClObqrnVjPmJLIeJ3m3q5bnUXBVWCnKb6ApTmwM2PSIoaEQ251MiijHumMqL967GJRP3zmSQ1GipI2fFNTPHoo8ZU/HGYgKh/vpSk7BnCJC3fDxKHiZutcqkwEPXPtcqkwgQlzVxFsil7Mjcr6m9TDwhJyrJRqj0+TC6PRkkYVXKxEKATPHsAACAASURBVGiULOVREiYiF3eQq5YsG55Rf2xUFqcm4hKk+iyyYdLRHoQh2XiQQarPIv09i2zKni0U+0VYosgtdw6IKtm4QKBKrqiAWzI7E88gVyUTJvlumnASo6e/CS100v34l9HezUgY7fANjMyUl9Mw6xIGRe92WHqvE5QZZ7rOaqMqIvKIqs7Z2eOtZlGFvl7li/+hPPWoq+W/8rXKP11TvwnvkSrzVzhHdcK0ABlEyW7e+pCeonLaQRmygbBwW8i/PZo07vxsaZ4jRwdcfXgjlz7Uye0bCgjwjumNvPfQBlb1KseMCPj00m4ObAhoyMEXlncTof3OXERTD4mSCZQ8paX5bIaSPoK4tF82GKci6ZFR6RFHmbLwleJM40ZA7Ui1fopqDLXoNJgiRPWmqtL1oCqGKRu5lDQ9CRqLSNmFqdZvEmUyFFQ5svlMVnfcR1HdsxLiOpR36H9Jpa1BgEbKmJYjaW46hBEth9Lbs4zGxgPJ5cbS076QIDuC7rZ/0NexyI2+GnEQI8aeRLZ5ErnW6eS3LSDIjaLl+H+nuPlJtHsj4bI/lqQZtAxhSe2m0TB73x9ltK9gYlGFO2+Dpx512xrB7XPh9HOUI4/e/TWMnoJy5Y/7eGqtezGPnxrwgysaaKjSjBRGyhV/7uHuVa4Z6PCxAb+/sIW/rC39jGeE8tVFvfxybR+PtLuwCnxnVQ/fXtcDKVHo35YyB16ynR6fn2ymV3NO9yuUDJFNebDBrP5cPry2UpylI5dqUy384Dq7B0e6yahaDGlnXi18VYfvlzkUcYuJiLrRYkoO8Z9xVQlopJk+7drheBXhwKZjaJbRPLv9d6iPKx6GGwZBf00jkAykhi8TCGPGnMyI5ukANDYmo4tGjDsRgJZRR5HvXoVGRRpapyMihH1baH/yc0R513HcMPFkRsy8DIB8oYtwzT2uljXhGDIHnlrxmhl7HxOLKmxYV/qiR4Hyy18pZ7cJZ54lBDtOFthpbn0q7BcKgEdXRdy+MOR1Ly69PU9uDJm7uEh7IeoXCoAF20Je9IsOglyyQqqihBnYkFfWbo36xyKGRKjfTi9/I6nPq0WaWiE1LQRU3h7UMNKhN4dXTGuopI8tAvE0vYFqKNLfh+KX3RBfc+m3J30H5aXydK0hDqPpvgYRihrQCEQSkdWAkLCsphAQajJ7PAKyPg5EmBOcwQGZA2iWFgShWzsZIaOINGJztIaxwWRyQSNd0XYWd9/DpsLy/lyNyR5EIBnGNc3klIZ/ZlthOV35VaztdN9xVhGKQcCBrSfSrE1s2HpX/7FB0Ehz40E1r3lDy9SS373r7u4XCoB820MUJ59FtuUgGmZdih5yHhoVCZrG1Yy7hI3Pwup5MHIyzDwLMg21jzF2GhML4L7blEfuVSZPg9e8RRgxWjjppcKfbvGlrawS5uCxJ+CxJ5RFi+Dd79l9YvHcxh3H87e1l7qz+1cXefMtvSgQStJfUBTn2fMRFFJjFEOhopcNs5XtQTVRGIyTT6lFtTAl4qKVw1U7dqjNR+VC1j+Sqrz0HneyUyoESRhJ5Sk9IywdJkNE6MW19NgQf5tEEA0YTytd9DBbDuEMjgaUvBS5Tx9hta4nSn2iNZIsGZp4g5xPY7YRESEf5WkIdnSII2WMy4kETAqSztnWYDSzW85lUfc9bC2uYVT2AI5sObt/fxAEjGucSVMwknWdD/cvRSKSYULLsTRnxxFGPWzrfJpcdiSTx51LpkL6tdBi5462QmKThlFDLxCsfAjuu47+6736UTjnY0POmzF49nux+MtvlJ/8T+JAnn1M+Y/vCptSn2IKy2YR3XGH8s6rlFxu1wTj7oUhN/2jyPPbdhSLkc1CPlTe98de7l8Z0SfJQE5NJ5sq1mpKCDTd1DOY2Wjp4KnwUVRaA4kpHQVTO86SY8vmfvUvr5HernJsqEPtm5D+5rPSq1yyahxFP0IrEKGBgAIhIEQqBKnjd7wUpR8gKq1lZXg9J3I4k6teo2YyvELcV89+XLyZAklTYoTSlEmWhqgkFLVoCFp48YiBl7huyU3k8HGXsL5rHgAHtp5IS24CAJPHncvkcbv2JYGGCSeR3/w48Z0JGieQHXnoLsXJ4jsoudPrHofODTDigF2L16jKfi8W//hzqWtZ8jRsWK3ce2f18mwuV1oKzheUYlH54e+VOx+MGDca3ntphmMOFyKFpgahN+/G+zfmhHxReWR5yLU/c+3BxQypZiJXa/jk7Xk+9lftbzIqpmsEu7/bpCo7fGO+Qg2iWq2h2nY1AaqOb6Kp0DSUNBllUA0H7AQvjaecgAj4YnASzWS4hzWs1k4OldH0kWc9nRzJeAoUWc42VrKZdsq/AuzmfI+ggcOYyNEczEwG32F7tBzBY5p8K+Po4IhBH7urjGmcwZjGGXWJu2HMUYw88j30bZpHkBtF0+SzkGBID8COZMq+ACfi5mMYdWO/F4vRZaPtsjno2K60pOb8ZIoQ5uj3ipdeKqxZB2NGKbfcEfHbW5U+lLx//rd1wMe+WSRogEIIUw+CFW3O8R4xA55apXRnkiJyyejBdE1hqLPIBsNQj63SITEY5x+lagHlI7T7m4DSYVLOvBj5EVFxH0FFJ+/a9psQXpQdwRbtYj0FXI1Ak87wZAgRAUF/c0vgf4VeXs6RAxklrvR+LtOqXqtTmcYCNvALHicCQqR/SKoQcD7HcCSTKx88ACdkjmFCNI6NuokDZCLTgoOHHMdwJTf6CHKjd6P4zb4ANjwNoa+JHfpyaBlin4cxJF7w8yw6NilLH4UpL4LxB+/49q9YrHzhGqVzu2veGXkAbNvitqNG+ld7nnqo8sqLhEmT4Ls/iVi5GiQDBe8k8w1Jr7CK6+MAiASidP9CLhXee8lCNtkOs9rvpPKNyXYxW3s735h0cBcyiT1MCVO+IQkTBMk8iCBQ4sJepsQe9U9WLZkTIdovEEEQkUuNqoqX8sDPd+iff+HtIko229+o5tLzeZooAe8dPZEswnENzXynawNbtcg7WybRIgELwm6UiB/0raMPJYvwkeZpzMmO5K09qTV+UEaQ4e0NUzk2GM0y7aSHkBOCsazULjbQy4tlDIryjG5jsjRzuIxmKGymi+fZzIGMQlA20M50JjCeCrOLjd1P92ZY+7jr4D7gRXs7N8Mem2cxAE/frXz3fW5ipgRw+eeU095YKhgjR7sVgzu3Qxg4oQAnFunPAqxcCutWw5//6oQCoJDW2VQtIN2nEJX3L1Si3s1K6XymGt6jKDXUtUoeokjcpDApHSVVMq5fSzt2o0i4anwrUxqyjAoCPt22iT6EKUGWk1sb2VQMeSLqpltdkT+MhPNaRnDxiNEck2su+a7F/xtVOrJmZtatP3VWbiyLo24ODZoZ45sfDpYm1mhvfz6OCEbysqwb3jmepNR5pIzmSBJhOFOGXgtwcbaWCMMUBliDyNj9tIyHw16+t3Ox3/CCFou5X3FCAc4x3vw5WLlYmXUCnPAq55Bu/5WyOf5KZrpjN7UdBkrYALfMVYqp5qhqzt8PfQcpGz0/mLGn6X6B0Hda75BAKnz/kBuqNhmlyRah6IfJBv2lfp9nf87ppiEQigXhzNEZrj64lXWFkK+sa6c3SmZ1qwrFIjRllYwIbxk7gn+fNKZ/YuGrRraysRhySC7bb7u3p5MvbGujLSxyZlMrHx0ziRFDaMceFWQ5IShdPvraxhl8vW8Zq7SXI4MR/FPDtEHHZxjGwLygxaJzS7IdZaC7D+7+qft7zXuVC64R2issMKlQ4nTDBvd7MAvLwQCjktKOfRBkYscesKOI+NpARv2XIwUkStKTENc5XiZYAQENefjMUTke6g6Zu8lP5FIhDJWDmwPG54SlfUVS07H42LRRnDHGtee/64BWOsOIly7cwJbQSUYYCV+aPI7Xjm0hW9a50BIETG8o7YA5s3kEZzS1UgRyu+njOjODVr7WfDQFjcgN8KlWwzCGzgtGLDYvh19fA8vnwZiZkBkJPR3Jfi0rtP7t53DBNTBlpt9fJd60QAhlbTGDmo2WkI2EYqgEWQgiJ2A+4n6C0DddCQQII0L401VNPNYW8r7b3eibDMJBLZAX6C7CqQcFaEZo6w2Z1xmSj9wkLlQZ3Sj0qdIpycfBRueEK6a3cnRHoV8sAA5vyvLEaWPJiPBwe57/XdVNV6hcdWBzv1CAG146Kpvh5sMm8tUN7WwqhLxxXCsXjRtaW72IUI/xKyYUhrH72ac7uLcsg9+8D1Y8BJnRThzCAEI/qq6YS3UE+9oB+P6IFsi1QF6Uvj7n99MjnuLmJkUpNCX2MND+pqH0kNd8UyqtTCIoxQqT4G7/fJaCCuf/Tx+9BddJna6BKMprjs3Q2ihcfnyWIya5nXcuK/KHJUWmjBSuOraBcc07KtXz7SHfea6PnhDecVgDx493mX18W5Ebl/WRC+C9M5s4cqTL4F825fn5ul4Oagr44LQWDmg0R2sYL0R2tYN7nxaLG18Ny/z3T2Lnnm+kv8G9mBpllBaLQmPSXFMiKP0C4ZuePIVcIhBhyrGH2SSefGNqKKxo/0zp9GgogCvODXjfBc7Q1qHMfTRk1baI3z0dEq/4fekJGf7zNbZ0gbH/oV3rIdOINNlggd3Nfj0aasVDqR9xk1B6slgRNJe078fBSvoR0gvcFb1jl1J7riBMOlBpHQ+9RXh+pT80nZamxgOpML5JueqSDJPGCQdOgEcXK7OmCC86JEl84kjhqrPcLbj67Ih/PB8xY7ww55BdnLBkGPsYWuwleuyb6NbnAEGmnk3mqMv2draMFPucWDTnD+b758LKB6BhTOoDV0VoGO9GP8XDVTMIDQ3KCW+Cu29ORVKlryFQkAJc8h6YNB2+9w3o2A4TJsGH/y1gxmHClq3Kf30+ZMlSN2ksXs4vW3BCEyqMbIEPvz3L6S9JhGFajYm8B40OeOPx1gRk7J/o6nu9UAAouupu9MCTkDG7uCyIsduoq1iIyHnAdbgW/O+p6hfK9jcCPwFOADYDb1LV5QPFeczqT7F8kdsON0HTBMj3wCEnwyU3QNAEN30IljwAE2fAW78mTD/efcTo2Qd2jC+ISofJNjXDK14vtI4STjxVadsIBxwIGf+BhXFjheu+lGXdeiVS5es/UB5/Rpk6Wbj2XQFjxwoTx0Jjwx5ck2M30RsqN63v5bnuIq+b0MiZY/fPprCChjzIetq0hxfLBGb5hfqM+qHdbRVtJhbDh7r1WYhIBngOeAWwGpgHXKaqC1Jh3gcco6rvEZE3A69X1TcNFO8nGosapDQuyMJ/7bhsP8U8ZFO+rqdD+dXn4em/QyEL7e3OrijTjoaOdhg1Dt70PmH2CUNz9IWiktvFT5gOBy55chu3bk6+ff3j2aO49ICmAY54YfKd6EmeIRl3faXM5ngZwkd5jCETbV5A9MjXEkOmkcwZn0MaRu69TL3AGM59FicBS1R1KYCI/BK4EFiQCnMh8J9++zfAN0VEdAAF29b6BOO6Tuj/Pa3Kt1KyZYXi5pHClZ9z213tyi3fhmUL4agThQuuglzjzjv7F4JQrOwNS4QC4DtrevY7sdiivSVCAfB3XWNiUWeC8bPhmKvR1fdCtplgxqtNKIYZ9RSLg4FVqd+rgZOrhVHVoohsB8YDm9KBRORq4GqA2Qe9lH85/j5WPuCE4vU3Dj1jraOEt3x06Me9kGkMpGTRboDW8m+b7gdkd/jqBTRUXafF2J0Ek+fA5J0u+Bp1Zp/o4FbVG4EbwQ2dverOvZyhFyAHNAS8++BmvrWmB4CmAD4yrWUv52rPM0oaOUMP5l7WAJAj4BViy4YYRj3FYg2QXgVuirdVCrNaRLLAaFxHt7EX+N/DR3LxpEYWd4ecO66BqU37Z4n6kmAWx+lENtLNUYxjrOxfTXGGUYl6isU8YJaIzMCJwpuBy8vCzAXeDjwAXAL8daD+CqP+nD6mgdNt8A+HyRgOwy6EYcTUTSx8H8QHgNtxQ2d/oKrPiMingfmqOhf4PnCTiCwBtuAExTAMwxhm1LXPQlVvA24rs30qtd0LvLGeeTAMwzB2HZsybBiGYdTExMIwDMOoiYmFYRiGURMTC8MwDKMm+9z3LESkDVgBTCCZ6W3btj2ct4dLPmx7/94+RFUnsrOo6j75hxt+a9u2Pey3h0s+bNu2d+XPmqEMwzCMmphYGIZhGDXZl8XiRtu27X1ke7jkw7Zte6fZ5zq4DcMwjD3PvlyzMAzDMPYQJhaGYRhGbXbHkKpd/QPOAxYBS4CPp+yfAPL+b53f/xBuWfNFwErcx93U/23wYYspe+R/KxCm7EVgqw8fpuIIgbXeno47AjqBQln4COhO2dPH5CvEH/nf+bLwUSqv5ekqsA1oS9nTcZWn2+Pzo2V/W/w1KpbZe7w9rBB/exV7RyqedNpPVohfgaX+nMvtHVXsivu2SWeZLfTnVn6dyq9lbG/34aOyeDZVSberwr1Xf49je5Ta31clfKEsfPraVTvf9f56pG1FH75QFldH6h73pq55R1n49L1PP3Pxvvi8imX2HpLntxv3rqSf6fJ7X0jt6yB5ZqIq6cbn+3xZXuN3oPwZ6qtyzXpw72v5M9pexR4C2yvEH/m4Kj27fVXsld6l+BmqZA9xn2sov//pa5c+rtp2CCzH+cOoLMwk4E1l4fP+Oj/nz6XPn6vi/OjjwNyafnoYCEXGn8hMoAF4ApgN5PxFPAu41p/864DLcA5kJvAHf8KvBK73228DbvXb84DP+O3bgS+SOKE4/GuAc1M3uSuV7sf8zdgC3OTDXOqPCb39295+lY8nfjm2e/tZqfjbgP9NxR/bu1P213l7HP/3fJjjgS+nHoofpuKc67c/AnzUb6/xccZidru3rwKu8Glt89ey15/3R3zYDcBXSV6i9+EcwFrgCyn7Nf6ha/f56PX7riV5Sa8H5vvwZ/mwEfBJ4ObU/Zvvr9vrUudzPO7b7Ap8NhX+p6lr8RHcJ3cVeAPum+4KfCl1j88BTvTb9/nzi+/Nh3x+/jkV/1nACf4c/jl1D84FZvnwm8vu8Zv99uXA51L22X77hyn7y3w8YVm6M4Ef++0PANf57c+krsNC3Dui/v9TJM9QV+p8DyFxFCHumTkL937Ez0TsMM7CfYQsLIvnXKCVHQUy/ex2kRQczgLGkjjYWDRe5uMJy9KdCdyQijtOdzXwdOp6HuO3fwf83G9/AvcJ5gLumb7Zx7ENVxB4Hjd593s+b524b+bE78a/+2M7gZNICiCfJRHouIDTDnycxPmm/dPnSYTsk8AYb38lzjlHOJ+11Z//bJxIKvBqYBTJM32D3/4UcBqJP7vMb38/dc7HAed7+924Z/pZH/+VPs9vBy7wYV6M86+/9tdhhM/bQfvSPIuTgCWqulRV88AvgQtxJ7xdVe/BXdT5wPtxD0YDsAz3cBZxL/bTPr6DgWm4G3gIbvai4h7WBm9vAQ7DObdRqnon7mY24JZt366q96jqF3E1mhacYgNsUdVbcTe8BXiRtz/q4+nycXQAoY8n/hDsCB8+Pq9m3A0LUvY/+PBx/C8DelT1UZKHDxJncCvO8RSBi0kcYQEnltv9MfGnpbeo6k3+Om7FiXUI5FX1yzhntM5fx/jl+ItPeyNwpLcB/BbnNOPajQKqql/z+0PcQz3Jb7f5fHUDp/v0VVXv8Pdvgb/HU3E7HgUW+3gP98fhz/Wo1DnP8WFOB47w9lf6vAL0quo83P2e4a+7+ntzHc7BXAqMS93LR/y1u9SfA8DfVXWxP5es/4+/l6t9Hv4J96159fZL/PU/MmX/q48njv9sb1+Ku+eKex5O9Ok2+vOKCyLxp/vuInHm8Wzd0J/vCm/v9fYef74/SYWPn6VHVXWVT3cTSfP031W1y2/HtaTQn1cc7wYfvsfbryEpdQdAtz/frlT8OW9fGl9D3PsSp/tz3DMQpa8nMBKY7Ldfi3smtvi/7bhnayvu/rbhnOd9JAL1cn/sWv/X4//eQ1KjaSOpZbX4/dtw7yP++H7/5PMtPu3TcZ9c2O6faXD+5hM+vQ2qusCfP8ApwDv9dvzMhjh/9ymf1ktwhYXYzx3u83IUrsAUP4vgChoX4gTjKZwvOt3n7QDvX+N3qJGhMgxqFpcA30v9vgL4JvAV4FlvexpXsn/Sh2/HiUDc9NOGc36xQt9EUpWNmzGe9hczdp5xLeLDPo01JFW8NpKRYs+SvHQKNHv7chJn2ucvfoaklBSXVOJ4NBV/N+4B+zyl1fki0FqWboh78Rq9vScVVwF4B3A/STU6bsZYAXzYn29cuiwAD/t4Fvpzis9tvrffS1K7iuMMcM1/6Sr9Sm+Pq9xx00dP2fnG57wduIjEScX2vL9uP8K9lD1+f0fq+UhX2yNcCezjqWsd5ysu5aavUR54s48rHc+W1DPXXRa/VAjf6+/ZAan8K05kwb2U6XPu8vbflYUvpOJPp7vGx395Wfgi7n24ldImT8U5iy2p+xRfz0tS8cTP1zaf5uXeFl/nKJVu+v72eft7/O/4msbn+59l5xvf94fL7IVU/MVUPOu9/e1l4fM4H7AudT/iknsnSe0jjns77tn8Xlk8q73tSnZsEvp1FfsjuPcpbSviCkpXVrCvA75edr9C3HOc8cel7Q/5a7S2wjX6JvAMpU3L6u9vulkuxD3jH8MJY/ocnvbx3IITi/iZiHBfI30Q937H733kz/lB4KJ9oWaxKyjuovwK+Ju3ZXGlkfiGZXy40X7/UtzNeRan8Cd4+19JblIL7oEF9yAWfdgeXLMHwD24B3shrkbyFeAXuAchfoglFc+tJG2HzbgmiZzPXwF4DOd8/16WruDE8WMi8jK/r4BrgsjiSnKP4l6+Rp/3uFR0vD+fjSSicKCP41u4BxGfh7jE8WsSIQBYoaoRToAXkLzs43EP7PMkpbGApLQ6B/gHibNoxdVWWnzYbd4uuBfx33C1ythJLfXxHO3/P5uK69Wq+gWS0l7R/1+IE8523HPxuL9GF/n9v/Ln3AuMEZEzvf23OPGP835FKvxTuFpsI665648+L0/hnFfOx/MEyUu6GmgRkYtJSn3P+euXxZWc43RX+3QP8OmehXtOvo1rT87gSqAj/DVr8jZwpewxuHuw2F+fwMdxA3AHSeGoRUTO9va45ir+2r3d2//i81/APdPvwTmf5T7+CMj6eK7FOZ6nfBxNInIJriSsJO3j8fnG6cY1oAaf7rdwz96z/rxzOLG7jeT5jPk9zjmvxonGQ7jaxgRc8+hHSWrfB+FaD/6AK6XHIhbh+khj+10kTv04XNPYAlx/wFp/Paf58F/GPdPzvH2yz0uvv2YbfT5H4Z7pt+N8Qlx7in3NrT78j3HPahbXTLSM5PlXH3altwW45/szuGfh1T5c6PO0FVcbPdbHtwpXgInjugfXnHWoz/dMf8ydOCH/mogcygAMB7FYg2928EzxtkW4hyAOMwt389bjXpjNuIfsQL//fh92k6r24Zqt1uJKrOL3bcE5uSLJQxs3PUyktFPuJG8/BHfxV+JucGw/wMezyoc/H9dscBDuRRiHu75f8uGzPvwyn8arcDe/6OPd5P8floo/wJUitvp0/8WfNyQd0rP8+a/x1+N1Pj/NOKcRX6Pf+rRi0TzGn3Ncqp3u7Sfhqrpx6S++Psf5vMVt0TncA3sU7uFr9vltEZEv4NquT8A5gOW4ezCKxOmPwgljBjheVdfhmmNG+WsRi9q5/v8s4L/9ucXPxZM4h3iN/x3iXooJuHv/LZ/uFL9/Ik7sNnt7fC8PwzmEX5bZJ5L004DrEzkc16R0NO6lBfiGqnb4a7AW90KDc3pLfZrLcH0Q4Jx/HP8onPPJ+HTPxTmflbg+J3Ai+yIff+xMwDXBic//0SS129N9vK/E3YcAd78+lLJP93HkcKXp2H40STPJB31803HOLPDp3eTDn+HDx4L4cb8tuPsVN3W8IhV/fC/G+nSb/d8RuPcG4Eycs55H6f3txd1zfNwf92mNVtUlPu24UCTATFXdjBOUE1LHjvD2uIbwV2+PhXYWzqke6ONpxBUSt+IKCLeQFCwyuGe6E1c4Cf0xx6vqAzjx7MCV7GORPwr3DizC1X7Ave+r/XnfgRMA/Dlt9eltJ+l7nE5S8Nnm4yuQCNhfcO/Xkzgf9VJVXebzOktVl+PE9uW+OfBvPq9VGQ5iMQ+YJSIzRKQB11E4F/gJMFpEzgD+jCup3oB7qfO4i3Uv7oV7BHipj+9pEZmOczzP4kpe4C70s7iHtR33EueA50RkFO4hiUeVtAK9IjLR2/twzQkjgW0icrqPvx13IwLcDTrWh11P0ofyUxF5cSp8XKJbjuu0zOJKSc/gXpYtItLqwwe4WsMRPp2luIdBcFXgjD9+De5l+FecGGRxL+AvcY5mHa6EOBrAl3jfTlJrmQC0isgbgLd6+/f9dWgSkXf48FlcaXOE3/6tP8ci8DVcyUpxDvOnuJfsv3B9FoHP50Rc7eLLPl0B+kTkg/48+3AluwYRmYEvveNKgTf4eNaKyAm4jtwbSJz7/f7+BrhZq8d4+wpf6j0rjtvbVUTeiXPev8M13YG79x/39+AGnPMF58Cv8nm8yP8G+ImIXENSoj/H25fjXtYWnIONS25rRORCH/8InIMAd5+24O7TIyS1qoW451twI12We/uncM+H4oSpw9v/inM+fbiSY+zY7sM5fcHdu7iDfh6usJP359Xpw/8Dd+96cW3hcQn5az4/8XWI+zXuILnmV/j4IXkP436PLn/sHTjnV8A9G3H+A9zzfra/nnGNeoG/PlNw4nupt2dE5CLcszuORExbfW38ozjB/pG3Rz781bg+rJWpY77v434OV7PEX6N7cfd+hg8b+85O3HsyBnef4hpBn4i8Ble46MAXcPwzvRH3TDxJ8qwsxNVyZvtzftzb78XdqwxukETcbDgP946NIvFBOZyv/BNuMM/t/J/mgQAAA4NJREFUuBaMg4FHvB86CljqfeSpwBMiMgF3PxcwEHu7z8K34Z3vb87zwCe87dPAz0jawtfjXqp2XJXwOXYc7llpyKtW+J1uRywfrpbeV+m48iGbcfzV4qmWn4HCl9tDXDPAxgrhK+WzpMO5zL6lQnitkGata1fpOijuxahmrzRkNKxi7/H3uNKw3UrhB8pPpfBxv1Kl/FSyd6fiSY8M6q0Sf7V0C2XxxPbV/nzLj4mHVJaPSNqYssfDIRX3fsTvTKXzjcrSDlP2dPzdqXg6SEZgxedbPhS2KxW+L2UvpOxxG7rihPE5nECn712loeBx/JXu7xac0FTa113FXimuuD8o3boQ56evSvgnqPyeVRqWHeFK+pWe6fQ1TdvT6ab3deP8wdIKaXwCJyTl9jU4Eez196HPx/OU/7uqlp+25T4MwzCMmgyHZijDMAxjmGNiYRiGYdTExMIwDMOoiYmFYRiGURMTC8MwDKMmJhbGfoeIhCLyuIg8IyJPiMi/ikjg980Rka/XOf2LRGR2PdMwjN2NDZ019jtEpFNVR/jtSbjlKP6hqv+xh9L/EfBHVf3NEI7JqmqxdkjDqA8mFsZ+R1os/O+ZuIlME3CzsD+sqq8VkZNwS3Q04SYyvUNVF4nIlbiZy624pSG+gpsVfgVustP5qrrFr7VzPW7WejfwLtwM4z/iZjdvxy0hQnk4VX3Wi0ovbhmGf6jqv9TnihhGbbK1gxjGCxtVXSoiGZJl6GOeBc5Q1aKInItbOiF27kfjnHgTbmWBj6nqS0Tkq7hvEHwNt+TIe1R1sYicDNygqi8TkbmkahYicld5OJIlLqYAp6lqvNSGYewVTCwMozqjgR+LyCzcsgm51L67/eKBHSKyHbcEDbilE44RkRG4D9jcLBIvO7TjNwQGEe5mEwpjOGBiYez3+GaoeCn3o1K7PoMThdf7hdf+ltrXl9qOUr8j3HsV4L4hcRwDUytcVxW7YexRbDSUsV/jVxb+NvBN3bEDbzRuATZw3ycYNKraDiwTkTf6dEREjvW7O3ArGNcKZxjDBhMLY3+kOR46i1vC+w7cUurlfAn4vIg8xs7Vwt8CXCUiT+CWoL/Q238JfEREHvOd4NXCGcawwUZDGYZhGDWxmoVhGIZRExMLwzAMoyYmFoZhGEZN/n97dSAAAAAAIMjfeoQFSiJZALBkAcCSBQBLFgCsAARQDyobCqz4AAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MULTI VARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "5FWz6VzE8Htr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ax = df[[\"Whole weight\",\"Shucked weight\",\"Viscera weight\"]].plot(figsize=(20,15))\n",
        "ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 588
        },
        "id": "w4E79VZGBa-6",
        "outputId": "1190e065-0e5d-4db1-9e77-5b6811adb753"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1440x1080 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABPkAAANOCAYAAABjsdlCAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdebhmV13g+98+VRVIQkCBQhQCwQkCIiK52F7obvERgQvNbblyQdP6tBexnW6r7dA0LZob6CYtCjSJSGLAgARkCkk6AyTEJJWpklQqU6Uy1Tym5vnUqTO8+/5xzn7P+77nfd89reG31vp+nidPVeq8Z+/1rr32Wmv/9hqyPM8FAAAAAAAAQLgmfCcAAAAAAAAAQDsE+QAAAAAAAIDAEeQDAAAAAAAAAkeQDwAAAAAAAAgcQT4AAAAAAAAgcMt9nfiFL3xhfs455/g6PQAAAAAAQHQeeOCB/Xmer/SdDrjnLch3zjnnyJo1a3ydHgAAAAAAIDpZlm31nQb4wXRdAAAAAAAAIHAE+QAAAAAAAIDAEeQDAAAAAAAAAkeQDwAAAAAAAAgcQT4AAAAAAAAgcAT5AAAAAAAAgMAR5AMAAAAAAAACR5APAAAAAAAACBxBPgAAAAAAACBwBPkAAAAAAACAwBHkAwAAAAAAAAJHkA8AAAAAAAAIHEE+AAAAAAAAIHAE+QAAAAAAAIDAEeQDAAAAAAAAAkeQDwAAAAAAAAgcQT4AAAAAAAAgcAT5AAAAAAAAgMAR5AMAAAAAAAACR5APAAAAAAAACBxBPgAAAAAAACBwBPkAAAAAAACAwBHkAwAAAAAAAAJHkA8AAAAAAAAIHEE+AAAAAAAAIHAE+QAAAAAAAIDAEeQDAAAAAAAAAkeQDwAAAAAAAAgcQT4AAAAAAAAgcAT5AAAAAAAAgMAR5AMAAAAAAAACR5APAAAAAAAACBxBPgAAAAAAACBwBPkAAAAAAACAwBHkA6BWp5PLhr3HfScDAAAAAAD1CPIBUOuSWzfIL3zydnnymWO+kwIAAAAAgGoE+QCo9cDWQyIisuvISc8pAQAAAABAN4J8AAAAAAAAQOAI8gEAAAAAAACBI8gHQK3cdwIAAAAAAAgEQT4A6mW+EwAAAAAAgHIE+QAAAAAAAIDAEeQDAAAAAAAAAkeQD4Baec6qfAAAAAAAVEGQD4B6WcaqfAAAAAAAjEOQDwAAAAAAAAgcQT4AABIwPduRgyemfScDAAAAgCUE+QAASMAffe0h+emP3uw7GQAAAAAsIcgHAEACrn90t+8kAAAAALCIIB8AAAAAAAAQOIJ8AAAAAAAAQOAI8gEAAAAAAACBI8gHQL3MdwIAAAAAAFCOIB8AAAAAAAAQOIJ8AAAAAAAAQOAI8gEAAAAAAACBI8gHQK08950CAAAAAADCQJAPgHoZO28AAAAAADAWQT4A6jGiDwAAAACA8QjyAVCLEXwAAAAAAFRDkA+AWozgAwAAAACgGoJ8ANRjRB8AAAAAAOMR5AMAAAAAAAACR5APAAAAAAAACBxBPgBq5cKifAAAAAAAVEGQD4B6mbAoHwAAAAAA4xDkAwAAAAAAAAJHkA8AAAAAAAAIHEE+AGrlLMkHAAAAAEAlBPkAqJexJB8AAAAAAGMR5AOgHiP6AAAAAAAYjyAfALUYwQcAAAAAQDUE+QCoxQg+AAAAAACqIcgHQD1G9AEAAAAAMB5BPgAAAAAAACBwBPkAAAAAAACAwBHkA6AWa/IBAAAAAFANQT4A6rEkHwAAAAAA4xHkAwAAAAAAAAJHkA8AInLwxLSc86Hr5av3bfOdFCiVMw8eAAAAiBJBPgBq5UIwoq7tBydFRAjyAQAAAEBiCPIB0I9F+QAAAAAAGIsgHwD9GNBXGVkFAAAAAGkiyAdArYwhfI2RcwAAAACQFoJ8AAAAAAAAQOAI8gFQi4036mPnVAAAAABIE0E+APox97S+jEwDAAAAgJQQ5AOAiDCODwAAAADSRJAPAAAAAAAACBxBPgBqsbxcfUzSBQAAAIA0EeQDoF5G6Koy4qIoQ/AcAAAAiBNBPgCIEGFRAAAAAEgLQT4AAAAAAAAgcAT5AKjFrML6mIoJAAAAAGkiyAcAEcqYrwsAAAAASSHIB0At4lQAAAAAAFRDkA8AosJ8XQAAAABIEUE+AGoRrgIAAAAAoBqCfADUY305AAAAAADGI8gHABEiLgoAAAAAaSHIBwAAAAAAAASOIB8AvViUr7acPEMJiggAAAAQJ4J8ANRj6ml9GQsZAgAAAEBSCPIBAAAAAAAAgSPIBwARYSomAAAAAKSJIB8AtXLFIauT03Oyef8J38kYicm6AAAAAJAWgnwA0MDvf2WtvOWvb5OZuY7vpAAAAAAAQJAvdr/2+XvlfZfe4zsZQCOZ4vFodzy9X0RE5jp6RxsCAADAjblOLtc+vEvynL4hAH+W+04A7CoCEQAM0xt/BAAAgGOX37FJPn7jEzLX6cgvvf6lvpMDIFGM5AOgluY1+YoYn7aXtdrSAwAAkII9R0+JiMiB49OeUwIgZQT5AKiXZfqGzSlMUh/t6QMAAAAAmEWQDwBa0DzaEAAAAG7QJwSgAUE+AGig2BRE2/RYFntGGcoIAAD2aJyBAiAdBPkAqKU5FlH037QmUfPOxAB0u+ahnXLTY8/4TgYAAABqYnddAOrxQhQA3PmDf3pIRES2XPROzykBgHBofjkNIB2M5AOAiNC/BAAA8Id30wB8IsgHAA0UHTjWNwMAAAAAaECQD4B6GuNoxaLK2pIW69vjTkdbTgMAAACALqVBvizLzs6y7NYsy9ZnWfZYlmV/MOQzP5dl2ZEsyx5a+O8v7CQXAHTQGkybmYsvGLbz8En54Q/fIN9Ys913UgAAAABArSobb8yKyB/neb42y7KzROSBLMtuzvN8/cDn7sjz/F3mkwggdZo33tA2yvDfff7e+b8ozrO6Nuw9LiIi1z68S9573tmeUwMAAAAAOpWO5MvzfHee52sX/n5MRB4XkZfYThjwnXW7Ze22Q76TAQzXXZTPayoAAAAAABCRmmvyZVl2joi8XkTuHfLjn82y7OEsy27Msuw1I37/t7IsW5Nl2Zp9+/bVTizS8ttfXivv+ezdvpMBDLUY49MZ5YtoIB8AAAAAoILKQb4sy54jIt8SkT/M8/zowI/XisjL8zx/nYhcLCJXDztGnueX5Xl+Xp7n561cubJpmgEkQmf4bF534w3NiQQAADBs24FJee0F35WtB074TopKmpeZARC/SkG+LMtWyHyA78o8z68a/Hme50fzPD++8PcbRGRFlmUvNJpSAMnS2FcqOnDE+BAayiwAoI2rHtwhx6Zm5Vtrd/pOCgBgQJXddTMR+byIPJ7n+SdHfObFC5+TLMveuHDcAyYTCgAa5UqH8vEWGQAAwB2tfUIAaamyu+6bROTXROTRLMseWvi3D4vIy0RE8jz/nIj8soj8TpZlsyJyUkTen1PLAYiY9n03qIEBAIAN9DHG4z0rAJ9Kg3x5nt8pJXVVnueXiMglphIFACK634iyJh8AAEgZwSwA0KfW7roA4IPGqafqd9dVmGcAACB8Ons+/pEvADQgyAcAbdCjAwAACeKFIgDoQ5APgHoap8Syuy4AAAAKxDwBaECQDwAilNHVBAAANmh8+6oAuQJAA4J8ANTTOR2EjTcAAEC6eKE4XKaz4wogEQT5AKAFNt4AAAAAAGhAkA8AGuiuyaczxgeMRJkFAMA82lcAGhDkA6CW5r5SMVBOcxoBAADgFrMpAPhEkA9AAPT1lhZH8hHms03f1QcAIF30fABAL4J8ANACMT4AAJAiRqwBgD4E+QCgAXaUAwAAKeIF53BaN2MDkBaCfADU0tyJ1L7xRkxv15VmMQAASYuoqwEA0SDIB0A9jQGrxY03CEEBAACkjlkeADQgyAcALWgdyQcAAGADLziHI18AaECQD4B6GgNp2cLwQoVJExHeJgMAALs0zrTQgGwB4BNBPgBoIdcYgQQAAAAAJIcgHwD1NL4p7m684TcZAAAAUID3vgA0IMgHAC3QoUNoWDMIAACLNL6dBpAMgnwA0AD9NwAAkCJecAKAXgT5AKgVRh9SZyoJQgKAeXMdnXU+4ENGZ6MPtQMADQjyAVBPYxey2L2Wt9n2abz+ANJz7cO75Ec+fINs2X/Cd1IAr+j6jEe/BYBPBPkAoAU6ugCQhqvW7hARkU37j3tOCQDN6BsC8IkgHwA00N1dl54cACThxKlZERE587TlnlMCAAAwHEE+AGigmIrBTqUAkIbjp+ZEROTMZxHkQ9p4wTke03UB+ESQD4BeinuRxWLTipMIADBocnp+JN8Emw0AIsImXy48sPVgd6kAAKiCV5EA1NO4e1t3JB9BPgBIwvRsR0REOlT8ABz5v/7uHhERec9Pv9RzSgCEgpF8ANTLFT9QMV3XPnIYAABop7i7CiAhBPkA6KVwBF8XG28AQFIYwQ2gCs3dVwDxI8gHQC/FT1La+28apzhDB8W3FRAERnAjddwDAKAXQT4A6mkMWLHxBgCkhXof6Jepf+XpGpUDAP8I8gFAC1rfZtPtBgCzivdNbLyB5HELjEXwE4BPBPkAoAHWZnJH88YrANJDjQTMUzjRQgWtL4ABpIEgHwC1NHeRio6t5jTGgjwGoEHGhksAAEA5gnwA1ONFcX1RvV3ngRqAKlRKSBt3wHhM1wXgE0E+AGhB61RSpclqhGkvADQoHtxjql+BNghlAYA+BPkAoIHuw57ndKSAB2oAGrBMAzBP6wtO38gWABoQ5AOglubOkva1mWKarqs1jwGkpahWOx0qJQCjxdQHAxAegnwA0AoPe7aRwwA0yLIwR3A/tuuI7D025TsZiEgWSBTr0ts3yjkful5Ozc75TgoAOEOQD4BaIfQhGWVmH9OCAGgSWpX0zs/cKT/3idt8JwMRCaVdvnTVJhEROT416+R8gWQLgMgR5AOABrSP6AggPlqZ1jwGkJaiXg1xM6DJ6eojmaZnO/L5OzfL7FzHYooQgxBexvpAtgDwiSAfALU0vxHtPuwpTmMsyGMAKixG+aJ22aqN8tHr1stX79/uOylQKpR22deIw0CyB0CkCPIBUE/jm+LFjTfoytlHHgPwr7vxRuRV0pGTMyIicnLazRRHhCsLZMyaqzUENfZXAaSHIB8AtBD5sx4AYMHiMg1x1/zFLN0JIhZALbz3BaABQT4AaGBxJJ/fdIyiNFmNaM1jAGmKvU7qLHxBgnwYJZRbwFc6uXMA+ESQD4BamkdLFFNUNKcxFuQwAA0SWZKvG+RbNkGoAuOFEgcOJJkAYARBPgDqaVzzJVP+tBfTSJOYvguA8MW+FutcpxjJ5zkhUCvyW6AxXvwC0IAgHwBANTrNADSY6K7JF7fudF2ifEAjoYxwBBAngnwAANUYMQBAg1R2Ve8sbLyxjEgFRqBoAIBeBPkAqBX5cxQqohgA0CT2tmmOjTdQIvZ7oCnyBYAGBPkAoAE6cu7EPmrGNbITaCf2e4jpukA7GteSBpAOgnwA1GIQAQBAm8hjfNJh4w2UCGWtXF8B+VDyB0CcCPIBAFSLfdQMgDBkxcYbkVdKCzE+WUaUDyWyQN7GBpJMADCCIB8ABG7z/hMyOT3rOxnW8EYcgAZFnKATeZVUrMkXSgAH0IbpugB8IsgHQK3IB0sY85a/vk0+cMWavn+LKeuKckB5AODTYswr7sqomK7L7rr1zXVyuWvDft/JsC6U9jj2UbcAMAxBPgDqaXzO0Da67J5NB/r+n44tAJhVtEWxV6/FxhvLeEqo7XO3b5TzL79Xbn9qn++kOKGwezaUq5F1kVcNAAJB8w0ALdChsy/2B2oAYYm9SprrzP+pYbpup5PLB7+0Ru7bfNB3UirZvP+EiIjsOTrlOSXwyv+tAyBhBPkAoAHWW3GneKBW8LwJIGFFvR/Si4cmo7q7I/kUVLr7j5+Sm9fvkd+9cq3vpCBAAd2qAGAMQT4AaoX0IAV7mHoMQIPudN2AQgdNqs/F6br+g3ynZueHFT5rOY8sAABUQYsJAA2E9JAXuiKnifUB8CnE3XU7jUbyzf+pYCCfTC/MHT6NIB8acF2E6acA0IAWEwCgm6IHTgAJy4rpuuE8yTdJabG77oSCSvfUTKAj+cIpIq0oKCJj+boMyrMFQOQCazEBAL1CethsqhiJomER+BgwChVIR+jTdUMbyec/x9wIru/h+MIEljsAIhNGiwkASFYxdUzB8yaAhBVVUEjxjSbTdec6xYsV06mpb3phTb7TloXxyBJQ0TBCQRGBYkcmZ+Tw5LTvZADJWe47AQAQIi0PeVrSYVMx8kzD1DEA6Qpx440misCghjq3G+QLZCRfl/+sg4i3qCuXf97rLrxJRES2XPROzykB0hJYiwkASA0j+QBokMpIvu7GG4bT0sT03JyIBBjkC6iMNBHa11MQrwYAZwJrMQEAvUZ1tEN6CC2TsyYfAAWKOiik3XVDbwuKkXwrApmum1orRbvcL/ZRvgDCEEaLCSBJIXSVQn+ACsHiTo+eEwIE4DO3PC2vveC7vpMRtZA2HWgykk/T98sZya2SoiIylrfddQl+AvCINfkAIGCaHsZsWZyuS6cZKPPJm5/ynYRodafrek1FPSGldRiqfQAA6mEkHwBAtU53uq7nhABIWhZglC/090Chpz9WobTHzpNJeQWgAEE+AAhYCv3J4iGP6S8AfMoWQgYhrbvVZrS3pm+ZJbfanW6hBF+9Tdf1dF4AEGG6LgDFUpiKitFue3Kv3L3xgLzgzNNExO503Zsee0Ye2XFE/uRtr7R2Dp+4lwADFqogNt5wJ/DkRy+Ud2+BJBMAjGAkHwA0oP3BKaSRJqP8+3+4Xy5btalnTT575/qtf3xALrl1g70TeKa9vAIhCel+arLxhkahBJNSEUMfAwBiRZAPAAIWyfPbWMVDKhtvNNdbTFIoM4ANi0vymbuJjp+alXs2HjB2vEFNUkoVgaq0t8qMYgeQIoJ8iaCRA8yZmevI+t1HfSdjrJhu+ZyNN1qjDdDhX3/iVvnAFff7TgYaKuogk7fTf/zqg/Irf79a9h8/Ze6gPUIfyRd48qPFdQEAvViTD4B62jqTX1+z3XcSulKYMrM4XZcoH8K29cCkbD0w6TsZaMlkrfvEwgujU7Mdg0ft0SCx2tpckfBe8qTQNotIeBcGABLASL5EaOywAaE6NbP4MJZMR96jxem6nhMSMEop0F53h1eDnariJYat6i2Wez+UfiwxL11cF5tAiimAyBHkSwSNDmCOpk58KA8+bcwtPAVn6lf/0SuFcgKEqHhRZGukcujTdUOTSnYn8jUb09RPBJAegnwAANVm5nicaIsRp+Z9/s7NctGNT/hOBhyycR91R/JZCgo0CTpprC1CC5rwUkoXjWUaAGwhyJcIFl0HzKHr7tbM3Pz0aAJVzdEEmPfR69bL527f6DsZCFxuOcjXZiSfhnqDeh9tuO6v8bwFQAOCfImgyQEQqtk5SwvSA4Bn3d3DLYUjbAcRXQltZFwqwUntV8XXVQj9fgMQNoJ8AFBTpqj3lsJL4y/es9V3EoIXczn5468/LOd86HrfyUACbNxHxSFtbSzUDfLZOTwGKOoeANZddOMTtL+AQgT5EhHzAx6ApbjnkYpvrd3hOwlAY8V0Wlsvj4oRZZpeTtVBW6YT1wUi0l2yYnqWGReAJgT5EpHKtAHERWsnMoRnJe559KI8AO3ZuItsj7RrdHyNjW8A7W6KtPeHNBblmJy+YpmIiJycmfOcEgC9CPIBQMBiDt7Ymr6Wot4HnXhLDBCe7pp8tqbrLvw5oT0aMwL1lVZcGYicftpCkG+aIB+gCUG+RPAmCzAnzEel8CwjymcMTQBgjsn7qThWaBtLACKU20GptbWM5AN0IsgHQD3No9V8B9B9n9+mUEeeaJTHXFCAgNm+NZvc+5pqC1oBtOGr/5hK8PNZK+ZDCZPTs55TAqAXQT4AaqmN8ahNWFwYyWeOpod2AIucBeADrU6pu3TivRFERJ69fH4k3xQj+QBVCPIlgsYYIdJabjU9KynNIiOWDQRTtZYHAImwUAd1bI/ka/W7eipdTe0uwuOq/5BaP2ViIZKQ2vcGtCPIF7HetyqaOmoAUMUEI/mMePKZY/KTF9zkOxkAhqB/FieCHmnatP+4iDDhA4BfBPki9tX7tvlOAhAlTZ23mNdaI8Znxr2bD/hOAoARrI/kC7yJCK2NS2UttoKm/pAG63Ye9Z0EpwK7PYFkEOSL2OzcYs1LJQykhXseBZ7BADOsjLpTuCSfxvYjCySaxMhMpERjXQGAIF/UevtDodXB31u/x3cSoIDWzrKmN/WjckhnzsGLQB6OgRQV7ZzW9g7NxF7thhLcCSWdAGASQT6o9JtfWuM7CUBwQpvWVCaUkRvabdx73HcSgCjYqGLtV9v1T6Ax4EhroBPXJW36agoAIgT5khHbwz/gk9bY0zfW7PCdBKOot8y44u4tvpMARMVk1eSqltPablUVWmsQe/OlMRAMAJhHkC8RNMWAHb7vrd4HiXs2scECyhE8BZqxESjrWL4fud3d0rSchwvag8cUf7voTwA6EeQDoJbWvoPyPi0AwIIwp+vOqxN80tj20u4CeimsMoCkEeRLhMYOGwADeu5tHoIAAL1C7/6F1n9NZRpraNcFAFJCkC8VNMYIGJ3JNA1uvEExAOBTcnWQoi+sfVoolFNUlgHANoJ8EWNnSoSOIlyud9TAVQ/u9JgS81jrBQAQktTW5MNwPIMB8IkgX8R6m5dUpg8gLlpjPEH03ZTmHRCiL969Rd7xP+/wnQygNq3taFX0X3UjqJm20OsXIFbLfScAblAJA+Zo6tTGfG/zJhxa/OW1j/lOAhSwObrYdl1epzrV2K7QGqARhWU5JgThAZ0YyQcAEaLjBQCgLYANlCoA0IsgXyJojAGDGFIAAMkKcb1Qrc3WRTc+IX95zbqRPw8wq0UkoX631oIFJ0K9P4HYEeSLWO/UjBA7pAClthx5BAAYpUn3z2W78rnbN8oX79nq8IwwIZTHCkayuhFKeQBSwZp8ANS59Ym9ctP6Pb6TMVLfpjb0bBCYmErsup1HfCcBCAJrnLqVSm6n8j3rIl8A+ESQLxExPdQhfr9xxf0iIvKjL3qO55ToR5ARKfvyakYAwZ1UaluN7QpBSkAffTUFABGm60atf7SRt2QA0eFhw43BB03qMT/Wbjskv3LZapmZ6/hOCoCa2tSbMVW5s3MdOf/y1XL/loO+kxKFqtNg8zyXT970pGzYe9xyikrSEVVpBoDxCPIBUEvjaAJtyCHY9qffeFju2XRArrhri++kAF6l0iRpepFlKs93HZ6SuzYckD/62kNmDlgikaJS6tDkjHzmnzfIr/79at9JgQX00wGdCPJFrLeTxhsswBw9jz+jxdDv8vGgSYd1tP92w+O+kwCgpib9P431YNvWoMgH282KovioCkVZmu3oK1M2UQ4A+ESQLxVpta2IDMV3tFHPYuQZUqAwFgGoFGrMwdQtXtQVE0RfjKr6Mk5j4BjtcVUBnQjyAVBL05ShXkqTBQN4DgEwis3qwdaxo6nTWra7nYWMsN18R5Pfhvjux3E9LCN/AZUI8iWCOhgh0vrmV1OQj6n4ZpGbAMoEWU8oard8KK6Zq6BT9Nkd5E0AW7T214FUEeRLBHUvAABIzdTMnO8kQIGiH+zqJV0q3e6q2ZlKfsTm8d1H5Qt3bvadDKOmZubkJy/4rty8fo/vpADWEOSLmKbRRkBMsvjf0SeLt9FAPO7esF9e9ZHvyOpNB3wnJSg+qkHbdW/uaLpuKn3vUFrKUNKp1Tv+5x1y4XXrfSfDqB2HJuXo1KxcdCObiSFeBPki1tvPYEofYIf3O8t7AuJCdgLxuGchuHff5oNmDhjwSwDtsSfbWet6um4qyM7hUnkZHGKNGHA1DlRGkC8RVGgIkdZiG0KnlhFpzaSQbSl8RwDz2tzvLuuKUacy1ZZ1p+saORqq0pzfUzNzsmaLoRcAiQq5r0nAHzEjyJeIcKtgAOOkdG8zIhmobnauI+t2HvGdDK8Cfv5MTlmwoO3IqKL9cLYmX+RlL7TgzrDk/perHpVf/tw9suPQpPsEwZuwSi7QDEE+AIgQbyibIZCIcS6/Y5PvJFT21zc9Je+6+E554pmjvpMSDWoHezojMtdUnhdBngnaRi80xgSLlyAnTpnfnCeVYqbwspZiVC9SQJAvZj0tTGhv3ABUM+rW5p5vhmzDOFfcvcV3EiorHmD3HTvlOSXwqcmLCx8vO8rO2TZo0nFcuacS5CmjOR+KEjGhOI2wR3PZBNoiyJcIHlwBcxglBwDpsdmXsv1iRnu7ZX3jjWL0jvJ8CE1Zdvp+/hh3XxWBX4pEc76vbxPM2EAKCPIB0It2uNSozgpZB5jHwyBEwnqwDSmtQxlOP6O20GUx8JtaMQuxmkllB2SkiSBfxKi6EAttU0+5t+KlrKi1NjdqoatIuH4jz0NBmD5581NGjhNykLdO2n3Ug7bP6XrUVmxtyaCqX09zNhRlwsY6jZq/N4D4EeRLROydDUQq4AcqV7i3zYptGsd31j3jOwlRCSnIE1tZ1iDE+jaUJI8sr4buucXF9u3exCHVESZUzU9fL2vHnbXTLRNoKsR2JsR6HKirNMiXZdnZWZbdmmXZ+izLHsuy7A+GfCbLsuwzWZZtyLLskSzLftpOclFHah0NwJW+e4vOAhT7va+s9Z0EeMKDDHrV6RL6KDojy6uhxBSHoW9sRtX6RdtMjF5FgIodl9PEZUfMllf4zKyI/HGe52uzLDtLRB7IsuzmPM/X93zmHSLyYwv//YyI/N3Cn1AixDctAMW2uVf+wFm+kxAkxc8jQC08wJiXWl/K5fct2/22bXEugk22b4vU2pDSjTfcJKORTmf+Txt1ZSrVb4jlPcQ0A3WVjuTL83x3nudrF/5+TEQeF5GXDHzs/xSRL+XzVovI92VZ9oPGU4vGqNCAOI26tV/2gjOcpsMJB/UYVSXGCaktDSmtsKfJSCofAYrRA/nMFOTuURxFvwmy9/NVHVEP2hVi/qb2sgZpqrUmX5Zl54jI60Xk3oEfvUREtvf8/8kPqiwAACAASURBVA5ZGgiULMt+K8uyNVmWrdm3b1+9lKI2FggH7ODeQtBq9m9t7DwI+1ytQYb4qJquu6BtNVQc39XuuiEGP+qovPGGknwYlowiAD7BlstJom+DmFUO8mVZ9hwR+ZaI/GGe50ebnCzP88vyPD8vz/PzVq5c2eQQaEhJGwvAsFGjNLR0rEOjef0g+BfiMwGjFswJsXoIJsmWE+pqum6IdYRNvuufcdfD5sYblAO9QqzHgboqBfmyLFsh8wG+K/M8v2rIR3aKyNk9///ShX+DEjy4IkRaSy2dNzc0TRkDgCbW7TwiB46fKv2c7bpH+6iVUcEgU93XxY03dOcDzBpXfop1INl4o70QHzO56ohZld11MxH5vIg8nuf5J0d87FoR+fWFXXb/hYgcyfN8t8F0ogHaLCB+IXasqor4qyFQId5vTNc1r04xeNfFd8q7Lr7TWlrKNCmzPl4Ml07XbVmOXU/XxQLFdWaRNMoEgNhU2V33TSLyayLyaJZlDy3824dF5GUiInmef05EbhCR/0NENojIpIj8hvmkoo2n9hyTH175HN/JAKJAfzBeIQZxAOi2+8iU7ySob7dG7a5rqkrudKfras+JSClsW/Pc4nzdRMoZM8UAnUqDfHme3yklNVU+f4f/nqlEwbzf/vJa2XLRO30nAwB0o7+KMUIaIe97LSwNyINmvGy8Yfv43fm6zY/xn772kDzvjBXyl//mNUbSFLKqwR3Nd6Dd+JTmb15fnufRTXWP7OsAfWrtrouwhFJ3/d1tG+XPr37UdzKgWFxdJWhGUADjMGghbWFe/zASPSpvTfVli7q9zfGuenCn/MNdW4ykJxZlgR/N90x39KjiNGoXYtZpLpOAKQT54N3/+M4T8uXV23wnA2jEd1Ao5s5KKC8q4E/M5V+z7Qcn5cZHWXo5JNpHrYzceMPcCUREfz6Eou510VhV200TBU2rbsCfS4SIEeQDgJo0dlZj5GXKGBcXYzR5KDg6NSN/9LWH5MjJGfMJ8uRdF98pv3PlWt/JcC7E6qFNnea0PizbeKPlA/niJgs82ZtUlpu+X4SOQ3tf3ai8CjkPWZ8TMauy8QYAAN656EsG3F+FA00eaL5w52b59oM75eznn2E+QWPYfPiKKWAJHWzXvd2NNxw912sObrmkOQg0arMXVBdiOeeyIwWM5AvYtx/cIW+66J+l0xleW/GyEqFj165yIXawqqIKA5qLt2ZAHY3KgYfCM3J3XUNpWdxI1XbLkkjLFVgFM6w/abOLGdszWJOs0tqH7+7BE9k1Anoxki9gf/bNR2RmLpeZTkeeNbFsyc8ZhgzYobTfAgO0dkpNijkwjEX0AMyzWT/Yr3p0l4iy729qui4P9m5pbm1SaO9doV8B6MJIvoC5eysJQCv6qGaRnYgFZdkeggPmjcpRU8GDxem6tvvMlI1hNN4zIyZCYQiN18+GE6dm5ek9x+Tk9JzvpACtEOQLGG8lETutXQreWMYrkX5sNGj/gPGa1Gl+Nj0qO6uZoXyuqoxUXsCX1cGag0P05dobd3kVX3oRGV4X3L/loLz1U6vk8WeOOk8PYBJBPgBqae8gaEAWIWXUEUA12gPi5fdyu5u9COiw8YZbmutomyP5lN9uSdMceAZMIcgXsNJKihYGsIL+Qbx4MEM0KMoQHmgLubORfGl0vuu2lSpLYd73h41DR2P0dPr6v+NbN13a33wALRDkC5jWyhMwhYBLOR7gDCM7l6AbHDaeYyBS7z720a6M2l13UbuC3A3ycUMYFfK0ZPqY7YXcBR1WcgP+OkAfdteNQMgVLBCi3luO+w9AJY4rCx5gzXrtBd+VY1OzvpNRW5tS4LIEjbo9TN02xWEmwo1JqVL1urgbQVkf03Wra7S2p9IOcpVkxXb9kB5G8gWM2bpIhdJ+ggopZY2LDmNK+VkVeYJQmawyQgzw9dI+gK10HF/L9C+OFFSeEYGpel00tiPlo0dRLsQ8dLs+J+ADQT4Aamntf2l9O9krhDRqRLbBFoqWe5+9baPvJKihvW6z3WYtTte1ehoM0DyiWPs9oUnpdSQvAVUI8kVAcwMKACGhPl2KgPGiNuWDbMQ4tuqeUMpdk4X9m5yB6bpm1C1Xvsuh69Oz9mOgcb8gEw0sRZAvYjQwQPx8d5wRHsqMW+Q3NKjTJfRRZG0vQVOsvxbyRhEhov6LW8jXd1xNwDM0QkeQD4BaIXceECbK3FJkSZhSL8uMQJ0XyuhkpuvGKYzShzLNNt4wn45xPvHdJ2TdziOln6NpQAoI8kVgVGVFPwZIAb0Vk8hNmFYEFVjk3S2yu1+dEWw+8s72KXPHi+2nUv5ieNawca1iyJcqtBTz2bmO/O2tG+VdF99Z+tnuFjxDKoNQXooAZQjyAUBNqXTeARHR04sfwvk6S8k8uoWPoOqCQLLB9uXqjuSzfA8zUrAfI2rT5TJg1uRMY6frNk0IoARBPgBq0TksRxaZRZmDLa5LVuoluZN6BgzQHnwaGRAwVCcvjt4xcrjkVQ3gcBu2M6e8ItPSZ6qTDCVJBqwiyBcxOjKAHb2dWzoLceF6Yhx21w0HI/nac/kAX7rxRss+bfFdbC+on0qxq7vGIdMgm/nMLU/7ToKIjC7XWq5qk/LFczJiRpAPgFpaOg+akUdmHTwx7TsJ6lDGgHCFcv+6m67rRjoBhPFfNJWgpy0Pbj/sOwmNubz2ps5FeUUsCPIhOf/jO0/IOR+6Xs0Qc4ym9RJpTVevENJYl4uv9KGrHnVwFqTE172YehvHSL6wjLpepq5iMdJngo03jKj79WLPj9SFdHmrtI3pBOkRK4J8EaOCGu7vbtvoOwkAlDo1O+c7CeqkHiwyxdd0tVS7AsqXslrKUnqb3L4ap1a23TCj01k4TsPOcdV6MLW+d/n31VGWXDRj63YesX8STzTWCb2aXF820kLMCPJFgOevZsi3kOi6WJrKjqa0xOCM05b5TgIi030I9jWiz/DxZuY6ho9oh82RfCHWu9ofZ0eu+WVqGt7Cn03zIcRrblPV/Egp39518Z3dv6cS7NVyfesEIcd9Usv3AdoiyIfkpNLwAqjvjBXLfSdBHfq8i9q8+Y9ld91QpsEGkkzrmozA0Zh3pjbeaHoLh1LuXYshWzSPUtMykn50EH50+rSuyZdXiPgzyg+hI8iHZOloNjGO1o6XplRpzaNQnZ7ASD5KTHPtdteNb7qulgfQYTSnzQfbu8ra1vZyLj7XN8uH4KZ/K0G2wYUm5SzsGhEYjyBfBEY9dPAWArDjT77xsO8kjPS801f4TkLQXvHCM30nAYoRNwoHQZk4GAvWFgP5GMlnSLX8SDXbAo+pLzHqMmq5vHXqCV6OIwUE+SIWWwNjGm/59eMSlSOPzDr3B8/ynQTUsHbrIafna9Kuco/6QVBmXqONNxRmXevpuo5310U/hUUKlrkMpjUayTekLqCcIhYE+ZAc+nfhoLFNm48XFRofbn3TnCeb9p/wnYTKXOeji/NpLhua0wb3ipGdzHIxo/LGG/Tk4qbk8pqu7xkog9AR5IsAHdlmyDaYYKMc3b1hv3xjzfZq54+4ILv6bqHsFoqwxXKrhhIkYbT+vDa5oCEHTaWhu+9GGMVXvdrXRUNhCpCWaqysPh32Y6dpr3MuJXkK2MQ2ggDU0tK5ce1XL79XRETee97ZjY8RY9bZKg+f/t5Tdg7s2R/+04NGjsNIjHaKoEIs9Vko5YE1+fppD26Vlau2yS+OH/oGJKGJpd6Dbk3apVBeWAFNMJIPyaLjgRiE8sDdhKtnsS0HJt2cyLGrH9rlOwnoEeO9qvkbsSbfPM0jGqukzVTyOy1H8tVNh95cN6Nquep+jHhK0JpsvOF0IF+Nk41Ns+L6EqiDIB+Sw1vckNDYpoy+lg5ch0Uh5YWtpIYy+sFmkM9GwDagomVMnUvUuu+2cDLbpTeMu8MDzwV83D0bUr2O4WrN1mXqPhJAkC8CGtqmPM/l4zc+Lut2HvGdlMpiHFURGzpe9dFpAXSKpT4Lpe2MJb9jtv2Qu1HU3QFlTUfyVSz3qRS7qt8zlPpCKy35N6o+1TLyrUk66C8jZgT5IuZyxNr0XEcuvX2TvOezdzs7Z1PU6YiJkv4VIkYZW9SkWfWWf4lfuMS/fleTbHD14P7V+8o3mDK+8YajXmA6fc36GzIgHlqmvtbbd2PMqM72SQFUIMiHZNHxCIema/X/XHG/7yQAgApaRnEMw5p8/TSOWjnjtGXOzlWU1QlH+UDp66dlRBri1KS6Hxfw11hfAnUQ5IuA5k42EJt/fmKv7yQkw1knKx/612jRZjRH1oWDIN8Cxdlw+orFIJ/ty7W48UazhqVq+lKJDXB7uaEmn0dO1639K1YQRAb6EeSLWCodDcSLh7S0cfkRgyKm4Dq4autsodyXgSTTmTrTVF3l3bMrjOQzdd9QHoCldh0+KedfvlqOTs34Tko7dXbXpTJAAgjyJYTRG/MYgh2OEEqs7/tq8PTc5oA9QbYfFtOsubrxXTdHwXIWnrHCx3TdhiP5TCYmApU33kg040x9b9v595lbnpa7NhyQ6x/ZPT4dZWsvDvl527RvOzAp53zoerl7w/7SzzY51bCqINXyivgQ5IsA9VEzVOT6dTpcpLoWRw35TYcJgx0wayOTqEXHIliyqE1WeMvFRC8fzce8JvWbq1j28888zdGZejbeCDFQHzDa17jZvL73bzkoIiLffGBHeTrqjOSr8BlXG/QAthDkg1E05jCJ2EK5wXuObkl9lDPYRPnyg+Ue+tUJbrnKuTp9xrbBueJcWtvIdTuPdIMaIaj78ofbUeQ9n71L/tVf3eo7GUOVXR8f12/Zwi45sxXe2NSpS3hxiRQs950A2MPbSoSOh7T6yDGYRpla1KZddd0kd6vPRPsCNpsPG8e2lV7NzajLtLUdyWc7MPCui+8UEZEtF73T6nlc01z+Cq6SuHbbYUdncmfs9W2ZsUWQb65KkK/J7ro8KCNijOSLGEONxxv11udY6IvPRuTE9JzvJKgXQgdaO/IQVQU5XdcizfcOL4mac5V1Vc5jbG2zhT95sHdLzV2oJiH1uJohpfG2WN4dydcp/WydXCpbXRCIAUG+CFTtAIX2VtuWccHPh7YfltdecJN8Z934BWgBrRT209RjmYHxQqrfNfvSPVt9JyEplNt5oeRDWTLbvrjujuRr+vutzh6fuuWK/NOtdLruiH8/NTs6AHfhdeubJ0h6R/KVf7bJSNtxdYHGoCdQB0E+JGtYe/DIjvmh9HdtOOA4NQA0COWBGOGI7WEhlHuEkXz9NBbDKlfI1EuY7pp8sd2QnpWv5cZ9mKpvrS3fMGOc5cuKIF+FkXzmhvIBUSDIZ9m1D++S257c6+Xc9GPqoy+C0FBk2+O+R4xcjFDVPAqW+7o5V9fVZQDI+e66FEAY4KoYld0Xw+5V2/fvRFZ9440mhn1nblvEgiCfZf/xqw/Kv/+H++2ehAqpkXHZRoAU8G//8WnfSYDoDuRg0f1bDjJqpgcj+eZpzgWXaSvujcbTdSsmNpX+Y9V2QXP5w6JR5Xtcee79HRvV7fKJ+TBFlbrc9PlTuY8RL4J8SA8Vd3DoJI4W60P9tx9sN82jjjhzcLTUvq9vLm7R76zbLe/93D3ylfu2WT9XKEFfgnz9mkxT1XCtT05XWJCrgqI4TFh+eqfY9SM/2tGSfT7SUazJNztXIchXI4Ua6jXANoJ8CbFZpdluxB/ZcVj++OsPS8fgkO1YgyNwS2spCr0T88DWQ76TgAVUlfptOzgpIiKb953wnBI9dhw66TsJKmju61RJ2qe+95SItB9ZU3QfGx+nbjZGPhSo9sYbisuhDbF83XGl2OZXnJqZk7Xb5vuBcxWe/erkd9tNeIAQLPedALQ36mE+psrrN/7hfjlwYlo+9I5XycqznmXtPKl1QhC+WEtsMU2jl+n782PXrZdlE5nc8vgeo8cFfOmNK7hozjQ3mTsPE+TrpbNPuFiAbPe/uhtvWD0LllJcSSwIue9/xV2b5e0/8YPy4uc9u9Hva41Ff/jbj8pVa3eKSLU1+Xo/ked5pZHLwz4TbkkA+hHkQ2tTM3PdbdJttZPFFu2nLTc3+JSKHDGKZefA5RP2v8fld262fo4YUFfqN6ztDfi51QiTI/8H2TiyrdHXTY7qquy4LKPdc7lqIyO/Aat+vcizwasdhyblgv+1Xr65dodc9//+y0bHaHMdbQZHn9h9rPv3aiP5Fj8zM5fLactH3+dVkp3xOgCBY7ouWvvy6q3yVcvrAE3NzImImb4Z1TZiEmsHepmDIB/CFsroCxcxhcX4xdKTmX5YCSTbg0knqmlbirv3SOPfr1agInnPVqpuUJrbsaExGVcEv46enHWUmH6urmndkXwzc9XW8UzkVkWiCPJFwHdHtsoblraKCt7kdx13LCp+wC9fQb7Q1zK0wXcbM8pHrlnn5DznX77ayXlsSbVMO+iaoCWnl2ihIrO98Qb6abkNtaRDm6q3g892pMqo7N5+SlmQb9zRtPZ3gLoI8iUklFEPPpFDCE+cpdbFdF2E7cur7e8kKyJy14YDrX6fptcPdted1yQbXOWcy0vUduMNilM/8kMPmwG4cUvAuCoDc5VOtPiZ6aoj+cbUBbwLQOgI8kXMVfsbbEVIBwVQa4IgH1DZsN0CF6cnGp6ua/Ro9oSSTmcUVqkuRwex8YYdZVcwhWDg03uOyV+OGFk+M9eR//zNR2RXw42Axt0jTteNG7Ymn6P7t8q37B/Jl0ChA0qw8QbCYqDeDjYoCSSERY81ocNcUP/AOuS2sfkgpjk/Qpi9sP3gpIOz1M+HmDfe4AWSWyHch239xhX3y45Dw4N4dzy9T762ZrvsOTblOFVmVL1bBq+yyete5bmt92yzZdN1EyiTACP5IkBV1cywB5+7NuwXkXh2KEX8Yu2rMNUOqC7VdffGCaEO+Zd/dav1c3z6e0+LiP4RbGVXq223rO0ajfpLk1t18yOA29Gq3uJrc+dv05quX/e9x/caT8s4zdbkW1qp0JYiFgT5EBTbla/rRgnAcCE8oKeCS7FI+/ufYSNgze+uG0aBCCSZ1j3xzLHGv2s7D11eIh7e/Qgh113XFdXWmNOlbor3Hz/V6nx129re+3t6tv1O2MqbeqAUQb6IDbYh4TUpdgXYxiZL87XynbZRa4/4Tldbs0PedAf+ldSoWzbI93Z8BQZDCcbZEtBgGSc0zlCoU0Zbp7/tSL7E76clKmZHqtk27mvXeYlZ5aM289hmrbHtwKQ82eIlRK/ePCjbeCPVMom0EOSLgO+Oh8u1s0x81cH05nkuF9/ydOu3ToAPhyan+/5f32NcMyFNZwF889UN0Dw6ynffqK7Akhsc19nL5YSI5QCcy303hm280fK7/atP3Cpv+/Sq0s9Vec7sTUvVuj+W/jIwDEE+JKtoAtZsPSR/c/NT8mfffMRreoAmDpyYLv9QgOY8BfliftB+33ln1/r8kckZ2XZgMrhgiU1NssJl9ikcrOWNzWyP/5aI7wuarMfcbJiiW9UAf5MXATc+ulvW7zpa+/di5PtODKVNqVPObn9q9NJM8dftSAW76yIoJurewQarWKB1cnrWwNEBt46enPGdBCtCXLNGuzOfVa/J/8VP3y57jp6S7z9jhaUUwTQXz2Oh3JmMBtavTjXftmy3bVJ6f33V0/vk/J95+djPBxIbUefpPcfkd65cKyIiWy56p7Hj+upShBIkKzMsiOZqJHel3XVrJOWr920vPW4s1w3pYiRfBEbVa5qn0TT17kvuNHaswbe6xBQQolhHWUX6tYKy5+j8EgZcCoyi+T4lxqdfbz+1rCy1vZyUB7Mq3/s18/2tnyqfvhkSF3WkiXM0eWbUXP8DqSPIl5AYKuMdh04aP6bLNQUB02J9cBl8i/ojK8/0kxAgALEG+9uI8UVnG3V6Oq6Kk8ti27Y89KaVfmP12B13YTuh1O2+02nq9GHkNlCOIB9aC3VIc979c2mVHup3QnoC6f/VxkOUeU3rtcOTcU4Jb0Jr21DUA73pM1E3HD81K2/46M1y3SO75LUXfFce2XE4mDonlHSiGt/TdXtVmj5o7nSqcZ/FoazPNXTjDUtpaaJvVHDF3xn/nZU29kBFBPkioKmBVZSUkai2EZNYR6toDaYgbU3aW5dl2XRw/IndR+XAiWn5/a88KMemZuXSVZuMHt+mjqbOkQIa61SnI/kcnYwXVP24DZfSmCej+pLjyrPv0Xu9FCUFUIEgX8RiqfD2Hp2yctzuyAc6ZAhYLPc5EKuQ79GypGv+aiHnuynFxmJ1bNp3vLtru+08rHX4ll21YjOnpoGJWF+o2ZZ6vmkMrverlkDXV7E337IKmdgkffqvDdAcQT6od/fGA76TAIzhex2S4ecPvVtN38s88jRew+734uE61QcZRvKJnJyZq/071zy0y0JKhnM5EqhBvLORVIJamkZxaWZyV+dBZur25gm0WQLq5luT8jgs/yjXiAVBvoSE2vGw9YAyLj8Y3YdQhHlXN+Rilzr7p/CmqEtDbQtQbvhDi+GTBFJ8NCSzzgOjjfSenF4M8lXt15yadRQNE7fXqNN2l6q+jTfC8t3HnpF1O48YPWbljTc03IgeuAgW+Z7urunamk5Kqi/HEA+CfBEY9cCmqO5tpcowbZ/HA3zS1Mmyift20XfW7Zb/dv362r9HHsbLVT0wWII0j3qwmbZQAuVNputOOwzy9Rufp21fvs4ZLA+jqtK9R6fky6u3GTuPKf/hHx+Qd118p+9kQKWS+8pTt6Fud6XRernBheuB6gjyQT1rVXAYfXSI7gdJn+7bfFBuXr/HdzLg2G9/ea38/R2bfScDinSn5g75WZvYbshVb8dwrGr5RCYfePMrzB7UsibX79Rs/Sm+jTksX3NtR/JV8NnbNlo/hxZVy1bAVYhRTV+yjcvn4mft+sgtftfTy6Vee49OyfaDk32JCbndAkxZ7jsBQBlXg09oExCa//vSe3wnwRruR6AdWw86oYxiM53OXMKbptmEy5F8da5R275g2yBf72+HNAKo9TTllrS8pA2l3tLK52Ucd++/8b/fIiIi3/qdn3WUGiAMjOSLQeTtlq3OVJFtzGADkAKqOrQ1OBpFc/fDdGwjz/Og+wtV0z7taocKqRc4aBtkcDGST6Pj07NWjptmbro3Lp/NBC7HVwzjftp7ft/loVH9MObLBVzVAyJCkC9qg2/PlLxMq23CcE27dE0hs8cHAMClnLdWS9jYXbfulDvf/Ysm5/e3Jp9drUfy1d7ts9XpjJk8ZWn6dcUvqCQbouS7jPk+fy9FSQFUIMiH1mwv5m5td11aBEAt7k+M42sK2stfcIaX81ZlurnUMtWuEdMj+cwezokmI32abNbRVJ3UtZ6uu1CWjRTpgGLpNoLdvUrLWAA3TqjVXKDJtqLZxhtmjgNoRJAvAprqIzsPBHaH8g3rODIYAkhUzD28iOo1X1fpja94vqczjzcsP5rmUVk7HsotYjq4kefh3UJNsqB3xJvtS+2yLLl+MUA/ch5r4dnnIoeH3auurmyVe6nq82fv58YNUrE9gAWwjSAf1LM2ko+OB2p48pljTkc4tBbKkzjQQNAjzCxy/Vyi+TJYiekYzN9/XL3V3MFGaJIFcw6vad+aXpbPO9fyBL1p5fG/Os11RBDGZKCLdnBcmzLu/G3vkbrrsedj/q9XoktzIkEE+dBabzVs482Hs84UFb+IiDy0/bD8033bfCdDle0HJ+Vtn14lH7tu/ZKfaezAxvACclgQnsA8Cr5Kgsb7vYzJNIdUtZi8VMXDrMmNwD5y9Tpjxxql6qiVXrZHvH3l3m3y4LZDIlKvbLbN+VQ33rClam6GWGeGQnPWuk5b1XLWWw+E1J4BdS33nQC0F3sDamvIdOz51tS//du7RETk/W98meeU6HHk5IyIiNy/5ZDnlKSLzlh7tnYqhwIOG7RQmk47G2/M/1n10HVSYGNUTpMjznbsjlj/8LcfFRGRLRe90+p5BqW68YatZNi4B2IydldcQ5nisowNf/GqR9WXwGXtAi+TEQtG8kE904+lI4+XVfgMkrRsYYtnRgIgChEUYy0P0Fp0N9ftab2aBo16f630CIqvg8mgWXGo0PoGfdeyYn5YjvH1cVl8Zg2235rW6zrnQ9fL+Zev9p2MkVhaoZ0quecri12dt9ILygYj+cafEwgbQT4YZaMxt7cmX9k/APO6QT46q86Q1eYpei5tjbftw/Ve49RzyGQd0g2iBncP1V/zzvZIvj6OKvrdR07KfZsPGjuetmJw14YDzs9ZtQ5OvR6yy+EI7rJTWUxK3Xr3+keekXs2Dr8nekfyhVefA9UR5ItYLA/Jtith6niUYSSfG7935Vr5pc/e5TsZ0YqproulfXMh1YCoyem6Ntbkc6F/VGa1/HC78Yadzw569yXt2xVX2RLbyLc6X2dqZk5FOjSdd9zvu/hO4+q8cXWK65qyNyVfuGuz/MrfDx/dWhqrjOv2Q8JYky8Cvjvw1oNw1tbkm8836nP9fF+j5QtBPqcjHBJ0/aO7fScBCJLTtZkCeQoy+U6m6Ug+33nVe/aqSbG98cYoNs+679gpi0cfrmnXNc/DGGFUeU2+GvfAY7uONkxNO76fo5oyu7mQ2XNpzdHe7zkx5kYL4R4ExmEkX0IC6ZcvYXxNvgo1N5U7ehUdge0HT3pOCWwLtJqEI9rLR1/T1TCx4wJDg22j5ofjPBf5oec92+gxQ+sa1FpfcYHLEes++qUmTlmlj9j0u+m9o/rZ2HiDvnczNuvhsddEUWGtfL/1fI7yhpgR5IN6tnfXpY5HGToCiEFM5TjUl1a2jH3ISzSv8jyX009bZuhYRg7jXG+5qDqiym2QL5yMdZVW07tCN0n33966Qf7qO09UPH67n6M533k77vQmuxu2ui6hLb8A1MF03QiMquQ1v2GvTRlz5wAAIABJREFUgyoYvvnuSDURYJK71m475DsJUM5V+xZKEKL70spBgxlCjmzcd1xueWKvseN1y1tgkfJGI/ny+oHBpkIoS8PYLAYaqpxPfPdJERH5s7e/qvWx6tTVprP15vV7DB+xvmFlpU6ejPusiXawalmuWxe0TVntpREqbwQzfuMNDfcfYAIj+WCUjRfArvrUsQRF4ZbGUhPWY+hS7/ns3TI9x/qH8C+0vXZsjXwfResDkcmdVEXCHfnfd30qXis2mBrOVVmPrS9aJ99M118XXveY0eM1YbPcmDh2m2M4+24Gy0X/mnyjP8coP4SOIJ9F63Ye8Z0EJ2xXg6Yr2ipthesHJeg1NTNHwMmDYYu/aw0oaHH1gztl1+H5dSOH5V9MnVbKQnW1F0zvHckVaMDB2vSu2qNLanzWQlb3Tdet+Duxr8nnStNRkKHkSYMl0ErF00K54XTDpaH/pmh33QZrRMbUJwIGMV3Xoq/et83JeUbVa4MVXqid9XFvWtpYsqA4lT2GeNVHvuM7CZDgZsk5Nz3bkT/82kNy9vNPlzv+7Odlzda4pzy7as2Cma5b8d/aCuU2NF1fFMVg3G6MGvVN161Ylk2vCTdOGHfXUjb7i4FUOZWFUoe24XvaZ5tzlVVpvmo8F1XtBEOdEDGKN/Rjui6QnASeC4wqHsz3HD3lOSVuuHpwXHIWxwXz8js2yepNB0o/F8oDpSumgzBF/yCsEF+/qgP03Jal3vX/3J3XBS0zQmzlq406WEmWWdH0q43LZhPPLZVHwA35XP+anxZ3+K3wmcojS/vnATc+DqAdQT6LYm6sXLL1xrTbaQ/4Qj1zZEqufXiX72QAxhF0r2dxpFH/n72Kqi6GnI3hO1Txsesfl/dftrry512PKEnlOrjc2MQkVw/hVbUJDGkKAoZWDnwqXkBVWirHYH//4zc+LtsPnqz0WRdlq+kpxgb5Ghz0lsf3yKZ9xyt/ftwz0rjTK7pd+/Sma+yafNzjCBzTdSMQ+1B40xVtTPX2r16+WjbtOyFvPfcH5PTTlvlODmCMr2ot1Oq02BGzeEiaGNJ7januu/rBnU7OE2p5aCMf+T8BvRQzPV23OGwo339B35p8lUfs2Cv0Gu6npmlg441mfF3zS2/f1J8Oi+fSUK6r+sAX14iIyJaL3iki5c9YGp4xq1S7VdPZ+7HAqnOgFkbyWeR7jTdX9fJgp9d0g2ArFxW0W63tPjwlIvF1CgE2eKynWCx/IhO5f8tB+eLdW/wmyLA/v/pR+cq9i+vc/sU1/ndN1GRYG2CiLf7Tbz5S6VzamO435DVGI/X/nuGEtKAhLYNJ0JAmW2wER373ygfkxkd3Gz9uE22meY4Sc9Cl96uZLhptDlc9LfXamBAu5bA1VjUENQETCPIp0unk8jc3PSkHjttZU8nZW0jD5zH95rzK8UJonETcPGzleS7Ts+wuC7foaNWzGITI5L2fu0eueWjINP6An6C+vHqbfPjbjzo/bwgBrV69LxdNpHzn4fHT3bTep6b7DcVLB98vb+tqsvGGzSs6uKlHaPeXbzc8+oz8zpVrnZ3vD/7pQfmvLetdrrA9LqrfUEYvV9/tefGTYXwzoBmCfIrcsWG/XPzPG+S/fnud0eP+u3/xMqPHK2N6ZzZb7Qsdj2r+5qan5Mf//EY5OT3nLQ1KnyNVCz3PAk++c8VIvlQ7rbZ2YQ+G5cXZfTg6NSN/9s2H5fip2dq/a7w4hLom34i/j9PbhzNdcpYE+QIqmq7uI015cs1Du+TKnhHUw5Qlt86LgNDuLxfGr3vXvrBUzfOyjTe8qx7l6wolgAk0QZDPorp1x1xnfrTUqdn5YMr+46dkf4VRfaoqWTHfKXT18KYtH6twMargn+6f7+A1edCCHzF0XEy/LChz2a+9wen5TOuuyRf+pVdlsBhqbSYW14wzcCwlX/Ky2zfJ19fskCvu2uw7KcHurttk91qrG7WM3URg+A+ff+ZpllLTXAxtbFtVA0xKqhPvbOSDyXt11KE0lHSTaaA8IhUE+RQ772Pfk/M+9r3Gv++rInP9cN6U1mlGdbgcoRHqaBDbIihGKg1bk89mXp/7g8+1d3AHFnf/HN0d1tBZhzs27pdMpK9zYXUx+4WjN/kepmMwi7tXZwv/H0bF3z+Sz39QpknQfMUyHTVX34L9FT5PIHBBd6p7udCmw2sxeF99/IbHZfWmA2bPYfRo5uvQyvVbz8eGrslnKkGAZwT5LNLSVBWNpq2Ka7CO1N73HXVd+r6HlotXER0jxGawA0gZH687XZdswoLJ6fnR19rb5FHa3PPGg3wNj+v75VjvtX/2imX+ErKgyXTdUMtv0yBGKF+38sjQGt8o5vbLxlcblbOXrtok779sda1jNUlf/5qf9X63bHO1vkcySwVj3GFjLotIA0G+BLiuqLR3yEYlT3u6x/H9IAGY5rpEh96hW9xdN/AvooyPduGh7Yflynu31vqd7sYrPf+2//i0wVT50/YSfN8ZK+SHV57Z+PdPTs/Jx65fLyIhjs5azL1zX1xttLLNMt8b5PvbWzfQd0lAd9Oa4O6dtsyU7XHBYpOj4UYdaWZhKan/9fDSzbza3L9laa975CZB5+TX8kXUCPLBONOdNlsdTpNrGMWNDIJ7vR1AF/doaNPwBnWGBHkGUdeF4d/+7V2NN+AysiZfSRsewh3SOwrwdS/9Pjnr2SsaH+sLd22Wq9buXDhuWPpG2jRZmd6w3tE7n/juk5V+J+/+6XlUZM/fQ6pLbfehTZ7fV766Llmmzuci3VMz80G+i/95g9Hj+rqb+6bdh3QjAzUR5LPId+Xh62G1bAi2bzFW6UxlRGwWXh6LiJvRaaH39RZHS/hNR2wGAwtaY8BjNzTwdF6fTN4HM3OLlVFo91eTy2O1Dze4Jp+H6bq+g4VlQnnRVDWdtabrWuzL+srXtmedG3NDmvxKbafrmv7duisoVR/J13PcYQcO4/YDShHki0CditIF84up2jEumaEFzbR3WoG6esv0MhdBvsDu+UGLa/KN23ijGK3oJElO+Wgn2h3XzoFDL8e9NAbUFCZprCbFzGYwJJSN2YbpG13uoSSEEvwbpHUk34lTs+5O1kLvS4bR2peNJkdoc1YNdcG4+9j3QB2gLYJ8CbH3UNHPxFvg//1HXtD+ICOEtlEI9LnkV1/vOwnR661HJhy0VKH351xM133P61/S7gAIn4f7pNnuunYSOlFzESff/Ysm/T6bSV668cbi/48Zr2QtPSHxXZYG+Z/83c4vfmqVk/O0rYlm5sbloNbcda/69HG3S8EAvhDks0hL5eH8bUQwbc58QkN9O9orptEbIVixTH/VGfrozt770sl0XetnsKsb5LP4RVJ8s+3zLjo8WX3jjLGPgTXbuLKPN1vnLQ6h3QFtr475qbL1jx9BF62WUV832Gyo8AJq4KNO7Dx80t3JGrp5/Z6x6fQ/XbfNxhuNf7WV3vMOe2eTWpuGeOl/Uk1QSAuyisiSp0rTQ7BdNQRU6yXIoC7tD3ra01dF733vYrpu6Jk2bnfdF5x5moi0/4p0fs2p0q79p68/XPt4CcZhh7KWDYFlcLPpuubTUdAwRa+pYWt55Xlu5EXxgeOnys+vNO9M7pJKG9Pv2iE72vYymVuuc75OXVCl2m1yf4ydrlv7aIAuBPkiULdRPDw5bbazMDj9wtyRFw5vp+kpDjvs6IH146OnreOX4ogm13o7gG5ifGFf02KjkmHfguLanM8H6/0VHvxj1qbYLinzhq7jRDe4Y+Rwfawcs0HbabPMDx66ypl0tf79fvlz98gr/ssNrY/zho99r/QzTfPBWv+p6kYHNU6vNI5Zalwb6+IreRsV1yINdT5uq39Wc/UFICgE+SxqWinZrHN2HJqUn7rwZrls1SZr5zA+ks/o0UQGczjUTgX86S1B+gKQvlNgRm+uLlvoidnM69Dzba47Xbf/i/zkS59HHTdg+8FJ+cwtTzdcr8xdZpoqksZfvDmu85qcr7f/ZfLeDu5lQJORfOZT0bV0Tb7y39Eygq03GUUpeGDrodGfN3z+UEdBhppum0yVad9Z2253XT/Pin2nDb3jB4xBkE8hY9XewPSdXER2HJpf2+GWJ/aaOsuSStJEvd235o/fl5CquWjgaQOX0pwnvjt9ptz25L7u35dNZNbzvJjmGmr2jVqT7/ff8qMeUqPbB7+0Rj5581Oy9cBk6WdtlYdQA29LglxKbxhb9UVoIz+aXB6bbcjSkXxKC5BHo/JfW9teN6hSbcpl4+S0YjOQbLPKMJFuX1War+LcW+cMXZNP2X0GNEWQzyJXgYCyCsn1m2ctb12ropOJujQH+WLkYnp0iJf0179wX/fvnc7wIF+W2Q+QhubkzJyIVJwq6LN5qHHhFh+kudg2hZa9Ll6Y1tFkVFebZL/906vktRd8t/94CvIhJXWy++qHdlpLh2vjlgQydg6Lx7adgrxjMBlS/b7uH5E7Zk2+wOp6YBBBPoWc1CsGW4YlS9+YO/TC8eyuyTfs8KHV7QQq3cokk6//h5/1nYxkuNh4I8QO3aqnFkc7jtp4I8CvZV1R94c2KquKqL5Si5vSVj6ENl1X/Zp8FYKQ4/pqZZ545pgcm5qt/4vDU9L9m4/2Qltwsmo5qVOebC4lFDNvo+LaTNctS3XvTcbIbKA2gnwK2aysi/rMZFBo8Egm1t/oS5/hDAnxYR7KZCLff8YK36lIxjJ6YqU6xUiugX+Prb57ePvhof9ep9npTm2u8uRga7kIl0/sJl/qZQOBGXOHHqlJVlkr94HdT4121zWfjK4la/JZPJdd5QWhcVEZkSlN+9ZagoO+A+Ra8sGkkL+T+bRXDDqbPi2g1HLfCYiZq+ZsVIXla3SX6Yrb3ppIudXjI16BPecFz/QD+4PbDsnk9JzZg3o2u7C97pKRfJEV1mKqbRt11ogyoWkbUyd5t/eM6mwr5AfHRXYubt2j+s5L7WvyVTmZliVgbO4e6uN4bTXa6CBhNtob37N42pzd14YsZfUJxRWxIMhnkZaHq+7GG5ZqrsGvOazizvNc7t18UH7mFc+vvWaQ7XYghg6Ii+8QQTYZE8K6VzGU64LpkXy/9Nm7jR5Pg1Mz80G+05b3D9AfNnrC94NBSAbzyuV9VbWaufWJvbJ5/4lavxM7extvhJXB7XeQNlvgXa/JFxMtwc5C5TXQuIL2FOv+NSgbWw+ckJe/4MzKnx9W9bWbrmtW9fJYje+Rp0BbTNdVxFaF0ntUF8GJYRXttQ/vkvdftlq+sWaH9fNXtXRtGDoiqEZ70x/Yc2ip0B6sfZiemw/yrVg20Kz3LmvTNhuVVpFNv5er3R6HnabuYQ+emB75s73Hpsb+rumH7L4QkMUyofGuT6EqsnlNO4P9Lnun6vMvf+yF8pLvO93R2eoZ7HuOul+VVr+l6FovpWGU57/+xG21Pm+6H1anXCRQ7QLGEeRTpGlH3HdwarDeH5ac7QcnRURk68ETlY7Zv+aP2e+3dKMQeiCoJ4UHPU1cLskX6gPJ9GwR5It74w0T38d3m9nEx294fOTPel8Q2r7ePspTk6tla0Ow4kHXRgmy0RdpNF3XeCpGH73SrWgoQT/w3Ge1+v3eZJjsA1QegWR4N1JXBgO7qbLR7BSH7DjI5OEvq/Kev9dTp76r9EKu6nlLPhhg9wAYiiBfxHxVVMZHDVhb+Hz08QniLCIrlup9qKZDYM6Pveg5Q//9j9764yJie9RQ2CW9CPItHxjJZ2r09mDwMGRFMaqSNz5v797UjX2GM3xpYnvxZTJ7gusbKIvyLZlB0RcksDeCrZPntevCo1Mz8vN/fZus23lkyc9MFoO5qrvUNh0I0Oi3zOmmW/G94zuPmhr3HGOa6brPX9+52omDq+uBAQT5LNKybpfpdRTKjHsY0bbTWyxs5NHOwyfl4zc+zlvYIbKMDoANzzt96Y7FWy56p7zrJ3/IQ2rCcmphuu5pg9N1DdHSnpmQK3jmrNsWtsl+45thBfBmw+TUsjYvAHwHTJuc32aaB4+89cCktXP1nTevPyJ8zZaDsmn/Cfnrm57sHsOGJTsOjziPttuuajnRlm5fbG68UTVQ3MawPkCrNflKftfWJkeUR6SCjTcsUvdIlLsJTAx7AGjzgGjrgYLddcf73SvXysPbD3f/32vDqOwiqbu3I6HsMgfF9nTdZYqDfJnUKztF3V9tTT4dpXJcUl2vu9uraUDo0R1HZLbTkde/7PtHfkZLkev9jqGtD6qk+I503SO7h/77x294XLYfmpTPnv8GI/dgJ8+7wdqqRzt9xfwj0skhO7GbvM86Fafh9qb7yOSMPO+MpS/FXKr9osJOMtRycesV18DFTrXDguTjTlt2i5S1Hba+kfIqETCGIJ8iTd8Wl1VYrjv9JkZ+2ayEl64hSJU/zNSQji0WpNZbtWzv0Sl543+/xXcygnZqdv5+Hdx4w1T173JdxHFMtGeLI/nqH8vl7Meqa4D15kkoMah/c8mdIjI/UreUoh1ZQ8nfQv/6xv5VvZSXrtq09HdbnLfT4CX36actExGRqRm7faGqAZrevurrLryp2r2jQJFuDeUvVlUDxW3UbS9Nrn1X5dzNRi2bOQ6gEdN1FTFZsczMdbq7LQ4/lzlLK1/Da/IZPVrPcce9gXIUxZme7XRH4LRhI1D55J5jxo8ZmguufWzov4e+fps2a7cdLv8QxpqZna8DlgT5DJVVlyOYNuw95uTli8nFvG3ozYJx17Hsa9TNyhjee9kqP6HV/Y2C0jbX5DP0IF7XfZsP1v6dZ6+Yr0snF1542nr4H5xqOeos2pZOqTs9Mqw7R4fSun3hTxfTdZcNedPX5p4oC27bKi9Vs4ryitAR5LPJYw3xi59aJX9xzWNekmG8I2L4eIOddJ/9ptf9fzfJT3/0Zo8pwDhX3L1l6L+HNppDO9+jxP7u/J/2mwAD5kY8SRkbyefoIt21Yb/8widXydfXbK/8O3UDkLWm9hpoIMp2JRx57t5jjB3JN/5cpvWNDlMWeCjYepEZWt3fJNjZyXP5hXNfZCE1zfgqY8V1PzlkJF+VYlCk++k9x+TgienRn6s8XVfpzVaiSHVo945NlQOkpceZ/4SR6boNgm421+TrO3elpTUqnjfQ+wioqzTIl2XZF7Is25tl2boRP/+5LMuOZFn20MJ/f2E+mWloPF13SH21ef+JlqlpzkiHrG+KieUK2WN9f3JmTo6fmm38+zRVftBXNcv3pg5RPHx0t9nr/2dja/JNZE7qm437jouIyKNDdrQcpe71C3FkiZYy6jIdrTa8sFRYbS0Gb0vz6eV2LrTP/KhdTyzkxNTMfBSuadrf+qlV8tZP3j7y55VHYTU8v+8ladosj4Dxiivr5BJHcvlMTiMGNKsyku8KEXl7yWfuyPP8pxb+u7B9suJQe/0Cy49QueRO6mgXC8DaEGaq4YPvoFQVlOc6Fq9nqPlWpHtJ/Wtod3VX03XLzmImGdWjfHre+o+Zrqu/OmrFyHvDNiNOelJgI6+H7SpuSpPv3bsTrfmdmcf9cNQ/+74Hl56/bjk4MGYk39LddYd/X9+5MKhq8LC4fr5H7IfIaZaVFOqyPsBgeSi7R0w/Kxq/PyivCFzpxht5nq/Ksuwc+0mBNY4rqnH1tqv1YT5y9Tp57unL5U/f9qrS4w7rQIb20OSi8+e/o62H5vKRZeG9Mfed2ixTkIiWuvXZkhhf71TD5l/S9QOa1XXBFIwsqfL9eh+axk7X7f0eQz5YNyu11PTt6tnevDN/nU2Wzzf/2Avl+kd2WyrzzQ6qqY3z9d7Y1Xk7Fde40fYCvWpqiq+n+eWosqztKk2WyXSXZELd3XWNbrxhsOhovdYuPfDAAy9avnz55SLyE8LSbaHqiMi62dnZ33zDG96wd9gHTO2u+7NZlj0sIrtE5E/yPB+6Wn2WZb8lIr8lIvKyl73M0Knj0fyBo2wdhSGdfoO13GDla/ztTIPD/ePqrSIiQ4N8S3fXbZIqpExvVzXM8qy47x+Mot61FYwftui2FQ0Kw3wbV/1711ojauCwde6vW5/YK2/60Rc231235xer5kqbq5Tnudzy+F75317x/Oq/0+J8NhV59yMrz2x9rN4+lI36dZnFCrAb0K54iqJvaCsAHtLLwsVrnQ38fzXD8vxj162Xy+/c3LczbtV1rLW17ZXTU7RN2r6AI22+d9nzlIn7qWrdcNazzY44Nl0a6o4sTdny5csvf/GLX3zuypUrD01MTJAhAep0Otm+ffte/cwzz1wuIu8e9hkT0du1IvLyPM9fJyIXi8jVoz6Y5/lleZ6fl+f5eStXrjRwat00PrhqTFOht4LurYRt1T5U9Giq9z6iFLVX+QHU1vktHdelxem6do7vcnddkfDvq3s3HZDfuOJ++Zubnqz8O6s3HZCPXrd+6M+qbrzRxvWP7pbf/NIa+cKdm80c0JBGU04X/vzMr7zeaFps7GJpM37eDWhX/XzNoGBdba6l10QMaLoRwOVD7q2qu+tqqxOLPnTpiC0HaYlV2ShPB/ttyDtf+4MiIvJLr3/J0t8dc3XL7pE6wc8qLx3+/OqhWwcMOW/JzysdJXg/sXLlyqME+MI1MTGRr1y58ojMj8Yc/pm2J8nz/Gie58cX/n6DiKzIsuyFbY+LcJkfyWe3Dorh5WIM3yEsuud3vkq2ykRnxncyKqvSgbM51UfzNKJBxSjlXo/sOLw4ks9WkM/RhI7SNfkMnKNOm9I0O4vdNLcemKz8O++/bLV8vicI0LcWXKtNKKp9i71HT4mIyK7DJ0d+xuUU5+JMn7t9o8zOVdyCdEHxlZcbLrh1g+hVPm4zgL4YtKt2jlqjXF0xWKcN+173bjogq57aN+S0g8E3M/fjoKrTdZv2ha29KA90BKJrbfoXc7be2tVw2vL5OtR0PWX6qx2bqr+J4biyGdqyNzVNEOAL38I1HNnJad37ybLsxdlCDZZl2RsXjnmg7XFj4Kp6KGtAi3rZVUMbSoM+mM4mU6NS4uq67j9+ys2JWlD1ADTgh/I9cu2K/yw/v/XTvpNSncP8HPagpPhy9tl7bEo+MvC2+o6n98m7L7lLvnj3FhGx91LEdYe31teovWumpXSMOIeJnKta55iom8q+cm+euJiCN9vJ5Vtrd9T6nSIgYyY/emYW2BjJZ3EoX++mB3XWgdT4gGurrL3vstXy61+4b+TPbbf3VV+Ma+tbV55mHOnYqOOnZuVVH7lRbnty6HJYPevkDvn+FbOkdLqugaxtU77bnT/OcgFoURrky7LsqyJyj4i8MsuyHVmWfSDLst/Osuy3Fz7yyyKybmFNvs+IyPvzVBdeUGpY/W3zAjUdyTfq10yndTA/KKy6/MNduqaKDaPv8WfR98kxERH5oeNDl0ZVyWV+Bt06DUn79oPzo66mZjqjPlLZsakZOTrldwSoiwC6yzLg4gEqpJGoTZ2cnmv0e6ZzxsbmB1an6xYj+SrmxMAydMY12ZjNV5DI2cYbY144V/l3f8IMTpry9J5jMjXTkU/d/JS1c5SN5DORte12Hm9+3KoDVEyLtTyG5AMf+MDZF1544YuK/3/zm9/8Y+973/teXvz/Bz/4wZdecMEFP3Dddded9Za3vOVH6xz7jW984ytXrVp1hsn0DnPllVc+78Mf/vCLx31mXPovvPDCFx07dszqHJkqu+v+SsnPLxGRS4ylCCIS1gK1derhJsP7bWVFcdiQ8tonV7k0bHFfbVcohYdql1zm57CH9FAu57ARP0s3Elr6/apWca+94CYRkb5F4f2pPnoq6/lclbJUb7ru6Cl7bVRJQu/zXfWNN9rvrjv2+J7ulbp1hMlmvTdPa84arsTqdN3iL5U33qj18dqa3D++u2iL9Yud41edkqltRFz1kXxxKuokm99vrjRQ5i53TZe/OtN1je6uW/Y9fFc4CXjzm998/Bvf+Mb3i8jeubk5OXTo0PLjx48vK35+//33P+f973//9pMnT6rd+ff8888/IiJHmv7+pZde+gMf/OAHD5511lkWehXz1GZeDPwNgdYnlIfoYEVUXp5reAcvGzQX54ksvMLgdCTfsPP3bqSiuPIdtgvn4L/YXMLHRd4M3w2+5+dDCkvdIIneK9yvN7/HBbl6f+JkJKSjHGzVhxo4hqk01x3JV+Weec0PPbdpciqfv/LGG91pzppbOQVMBh0Cna5bNd3Feppln666NmFTpvOveOdW/bj1C82zl49/TDfxlcpu9XE/HlcGSjfeUN4SUwXa85a3vOX42rVrnyMi8sADD5z+yle+8uSZZ545t2/fvmUnT57MNm7c+Ow3velNkyIiJ06cWPb2t7/9h1/xile85t3vfvcrOp35+uSaa64569xzz331j//4j7/6ve997zknT55ccsWuuuqq5/7UT/3Uq1796lef+453vOOHjxw50ndD7dy5c/lrXvOac0VE7rnnntOzLHvD008/fZqIyNlnn/0Tx44dm9i1a9fyt73tbT/yE/8/e18eIEVx9v30zO6yLLvcyCmuCsuyi6DhUBFQeFUw8TYmnkgSE/PmVTxJTAw5MInEIzFGP48cGlSMB8Z4oiAIcihy38h9LOxysye7OzP9/THTM31UVVdVV3XX7PbvD5jtruPpOp6qeuo5Bg0aOGjQoIGffPJJOwCAp556qsvEiRP7AgBs2LChzZAhQ0pLSkrKJk+e3KugoCAd8QtF/+9+97tTDh48mHvhhReWnHvuuSVSGhkoNPlC8MOTo2yBaa2MKnXzJPMgKDyfHGKNxQlZesjdHfBL+FGYrz5bUnl4pP3AKC2KtMLP9kQNYw00pfvUgJuAkjYND4JsH+s3+eB0zpyUk+1ZtNH5irDAj+i6Bli+WTXBg4HMeiXWu5zIdVDTAO4a28+XNY86gnnq89ICDMF0BBldl3UkkJ3yi4O9Gpzwg7YdmuMJqDkZg87t8pJ8QQz+AAAgAElEQVT5JM1RWplcQzOdqT1tOlVgjCd6wT97R/Tq2JY5j5/QMb9pkJCmv0SGqmtWUJjy1ppTv66sEWreWtKjqP6xbw/Zi3tfXFzcHI1G9a1bt+YtWLCg3XnnnVdXUVGRO2/evMJOnTrFSkpKGvLz83UAgE2bNrVdvXr1juLi4uahQ4eWzpkzp3D06NF1d9xxx+mffPLJlsGDBzdec801xY899li3X/3qV2kHmQcOHMj5wx/+0HPhwoVft2/fPvHQQw/1ePjhh7s//vjjB4w0vXv3jjU2NkaOHj0amT9/fmF5eXn93LlzC3Vdr+3SpUusqKgocdNNN5123333VY0fP75269ateePHj++/Y8cOiz+kO++889Sf/OQnB++4446jjz76aDfzOxT9v/zlLw8+++yz3RcsWPB1z5492SPGUEL903QrhQyBSpB8jfVzLAuHYMLtN9RZzfCzmXYM8qLqS1tUdEqezfCzPVW/PSYBHTTEzs+y9/vMkBncIeOqgT4tL3DCFdZx6C26LnfWrIdoAagh2BDBR/zoF2affGkhnxyezPPJQfE0kcFbRIBWmPSzWWvh7ZUVsO33l0FOVJ7BFm2/NDTRSXMaYwFJfQCAZ2T6EdTQj8AbbpBVRZMM3wcUcPueVrxc+oqhQ4fWzp8/v93SpUsLp0yZUrVnz568xYsXt+vQoUP83HPPrTXSnXXWWXVnnnlmMwBAeXl5/fbt2/Pat28f79OnT+PgwYMbAQAmTZp05JlnnjkFANJCvs8++6zd9u3b80eMGFEKANDc3KwNHTq01kYGDBs2rHbu3LmFixYtKvrpT396YPbs2R10XYfzzjuvFgBg8eLF7bdu3ZqWttfW1kbtGoGrVq0q/OSTT7YBANx+++1HfvOb3/Qh0S+kASkQCvkUhK4L9qGTJQKJ5IZBrj8hP8p1w/zNB2Hqf9fDp/dfGBAFfPBrny3TP1FLweTXVmEjumVj8wWtyZclLBKjdWxL44PgSiZQY0G8NpHKLZAB7aHevMbLHsrB+eRjSy/Lt5zwoWP6MBmjklVQlU4vgRYA9rm3Yvcx4S4IxGjWimshHl/VJLy/JqmoEtd1qYc8WnpOUmroyQhq4wdkUu3eJP61GTJIsIfqGwPS3KTlQVmyLfQMksadTIwcObJ2yZIlhZs3b247fPjwhjPOOKPpySef7F5YWBifNGnSYSNdmzZt0h0WjUYhFotRdY2u6zBq1Kjq9957bycp3ejRo2sWLlxYtG/fvrybb775+BNPPNEDAPTLL7/8hFHOypUrNxUUFHCNdl76RSD0yacoZJnLyNigOx2/i61X/rrv78bi1+9ugH3HGqDyxElhZWazdlIIdry7Zj9Un0RreGfbPrmhKQ43//1LqrSyvi1bNnNoU2NbGkmBIvwGLvgSai1hFgAx0eG9/XjHV8Llu1HvRKzxbmPGLx4jQpAi2rdcXMLHy7yINUziUP48UTBZOUvB7A2VTOmve3aJ0Pq9BIqTFwTOXhE+pUqgpYbWDDfbhHzGhTTPGkG7LqvAi8lzhp8AFk0+WT5Cs3V/1BIwZsyY2rlz53bs2LFjPCcnB7p37x6vrq6Orlq1qnDcuHF1pLxDhgw5WVFRkbd+/fo2AAAzZszoMnr06Bpzmosuuqhu+fLlhUaa6urqyNq1a9vYy7r44otrZ82a1fn0009vjEaj0LFjx9j8+fM7XHLJJbUAAKNGjap+5JFH0pGAlyxZ4rChP/vss2tfeumlTgAA//znPzvTfH+7du3ido1A0QiFfBLBy5M0TSzjyUbNHj+Q9l1GcXAO4R+ybclVTStIy7IWXLX3GFU6UXMSHV03O2Y8al2wa74qNhyZgeoJ0RtxT22UXjdcDl8CDmfmOhCBldMwv4oh1Z6yfFBwQNbhjZXfu6WWzXmMyK1RSjcYGRmfHMqeX7ADX7diw9RrW9B+jmhNPt70rMj4tSZXVN9E6XJKsf53A/u2gX0cqTYnRKKxWU1z3RD+YMSIEQ3Hjx/PGTZsWNqEtrS0tKGwsDDu5qeuoKBAf+6553Zdf/31Z5aUlJRFIhF44IEHDpnT9OrVK/b888/vuuGGG84oKSkpGzZsWOm6devy7WUNGDCgSdd1zRASnn/++bVFRUXxbt26xQEAXnjhhb0rV65sV1JSUnbmmWeWP/30093sZfz1r3/d+9e//rV7SUlJ2bZt2/ILCwtdbzZuu+22wxMmTAgDb2QrvGyRWipjJ20GzG/MG2m/blr8avLw5oiMbBn76suFlCcwEGTL+EKCQks627QhcDDzSdGfxMKDvdYtKgAXMbqu6VUNRsM328HairLMdeOCbEf9uhwyhL45JCmxCelovOHykQYqQnMQzUM99BzESRJ4U6ZrovS1Jzm4rjTQT2UejT/RJfIDeX1EIMCtXQLzyedCVwvZQimPnJwcqK2tXWV+NmvWrF3mvy+//PKayy+/PK2hN2PGjD3G76uuuqrmqquu2mgvd9myZVuM31deeWXNlVdeucmNlsrKyrXG7+nTp1dOnz49rXLes2fP2AcffOC4nZo8efIRADgCkAwksnr16s2RSAReeOGFTlu3bm3jRv9DDz108KGHHkL7XRKEUMjXAkCt9m0RnNEhntChvikGRfm5HJRloKbTcN30b3bDTwfe8utpCT3iD/p0agsjTu8M7689QL2Rbu1Aja5sOc8iaefwyacyaIQLdh7BKwDyC7zV0dPpn0++15bthdeWZVz4qDre0kI+SYE3vGJ/yl2HbGHa3mP1AAAQjWhAMxIlW+t6gu/zllChyH5zmAXj0im2W6W9UDJSuQpXFPs+N6CEv2b4EZxH5JyQ2fpndG3neNYYSyo7+X+hQPel2WLhESJ4LF68uODuu+/uq+s6tG/fPv7SSy/tCpomgFDIJxceGIRIxq1hftPg4fc3wktLdsHmhydAfm6UMpfaC3XQh+JsCYQSFNQePWpB1QO2ykBGqM2SKYl2LeBurntmt0I4XHsU8iRGWhQNnE8+XdcdGgBa2jcSW196OYR5nXs02c2H6CwZospAltmpqEuoMY/OBwD5+4FnP9sOAAA5Ebq5L0s4mo3w2tOkJtR1PcO3EDXpug7PLtgOVwzuBad2Lkg980iQYIg2M842Tb6Iad2hA8+kYncLISsavch3ABkNzzY57rxJJDtSbR6FyH5MmDChdsuWLQ6twqARCvkUhdAbLQ+7tXdWVwBA0jk+TsgnapNqOcgB+rdI1DUGFNkpS8VYvplNZ2fzBAYNNKUko68t2wMjz+wCp3Vx3tyi4LfQG63Jlx2id9QcdFxaINK8MHEYrNl7nCnYD+qg4McwQ/WE+Zuue3ap432bnAjUNtLXwfIdQfIjS90eAm8wf4NC/IQHLyzcDn/4cDMAmLRtBH1Tr44On9tcEGX2S4scSp98Rt9nB0f0B0ZLMPvEo02HSHjgxEl4dPYWeHtlBcy970Ku+mVDND0JyXNC9B7WGBcyXWT42eeuMx61J8BcxtFgc2XSijGX4vIxvHQIEYId2XOt38ogK7ouayYVNhWib6WaU1ogP3l1ZbL8jCd1ofW4IdxEo5GtQtCg4Nj8pP/2vx3jCR1+/vY6uPb/iY2IKBK64lbNJ5vjaR5lB+oMZDcpQbGxDm1zYUyJw1cwEfZy/OZWlosewlDOy4nAjSP68hfOCVdfSULqyBRiD7Bihui+UZIDM5zyDAEfR1ZXTCjvwZTebRz4dXiN0vrkS/W+ofinwh4wKAT57YbgqKEp7nimCmj2ajO/3AOr9x73gRpv8NK0Mnslm8113Wh7cfEuAMDzJlms0X3tVmuehQjBi1DIJxG8DErX1dtkM5lAcRLvl3DH2DTVNiYdlKfp9fmqSOT3+tFyfq17KEGGamuuKreKyM2IAm11rL4paBKwoNGGCxKlU2fDhCcXIt8hTY1tf4s6CCowjNIg0XLvxSVUmgDW8ui/zp6WuV00b3sBUzG+gaWu1nYpI1oz0C/QB94wfvnPFNUbS0l6UL65ZF7S2t0TqAoaxbtf/GcdQ3n+fOtv3t0AxQ9+4Lkc3fRj79EG93TmZ7Smzq7musGCND4twbMIZfh+gUjZaAptC0OE4EIo5JMILwdHkQt72tRAWImYCiRB9LqPi+hk8V0YcvfAoPKmVjXogJ9+epZsUfyea2i/dmph+6E65HOaqSFq9qg0D4X7GFLg02ic+lMfRkyTCG3unJ0QwRtE8xfRztjNpcmcc1Fan3yp/1vyHoj/IjoDmYE3kuUj5jGncELWsKIZr34oCLDipSW7hJRj0LvjsHW97lbURkj5dDT4x92RwkoB5fod4MIShDJbF8cQISgQCvkkgpd5aIJdbFn99bAxU+ELCO3tlSmdaN819iikQfH4bDPXDddCNWGf0sbfWthjSCBbJbumogWugYQEHGh9BcqnHCE5j/YPSw7cEsiyNvK2pbkO2qW7JQtneCDiAPnnuV8LoAQPP/qMUpEvPeZo02cTNI1tPJCmuOfAO2ZtPUoOoZqmI83W3OxmwI1+1cyR3YCzCph5+7nWZ17q4GgSe57DtY1wor4Zm97tLMJ7DhSpMecnO8quUZi9+NnPftajX79+5SUlJWWlpaVl8+bNawcA0Lt377MOHDjgOWbEddddV/ziiy924sl733339frVr37V3SsNBhYuXFgwadKkU0lptmzZkte/f/9y1Lunnnqqy65du3JZ6w0DbygIXWdj7KqtiyhyaPZVuO+4/801nuixw74x8VtjRUZ1KmndeEUL+hTpQFrrBth+8seh9/Kzea6gSLf7aqP5vmxoA1qH3p4+RaJmpIgWNq9VpINYkD75/BpKvN+ouqxKNaGsMeay7RKSBl4DZ/jFN1UT5nkBm+m/OPjRV34MBxFVDPvdXNA0gJ2PfAtTB38tWbCVcICWZNV4c0vC3Llz23388ccd161bt7Ft27b6gQMHchobG1tsi48ZM6Z+zJgx9bz5X3nlla5nn312Q3FxMV5aj0CoyacqJDNOEcVXn2ymXkizcSEIYYVfG9xsuc1VYTVqjiewh7GsMdelTSfoc5CXEFnSVkjNAUcaAhQ0nbIDSaJohXIBH0d9UODIj8pDGv9uc4NZuJElPJgGgR/UXJrSb1M1Nxg8RjGypEGFsa4ACZ5Aop8UMMhZjriGoNEyLO5S4LEO9/XYK0QF3hASCAp1qUxgcPTCNPdWE8kns32+tQRUVFTkdu7cOda2bVsdAKBnz54xswDr0UcfPaWsrGxgSUlJ2apVq/IBnNp1/fv3L9+yZUseAMDTTz/dpaSkpGzAgAFlV1999en2+u6+++5e1113XXEsFoOpU6d2HzRo0MCSkpKye++9t5eR5mc/+1mP4uLiQUOHDh2wdetWh819LBaD3r17n5VIJODw4cPRaDQ69KOPPioEABg2bNiAdevWtamuro5cf/31xWedddbAgQMHlr3yyisdAQDef//9orFjx/YDANi/f3/OyJEj+/fr16/8u9/97mm9evVKay7G43G44YYbTuvXr1/5BRdc0L+2tlZ78cUXO61fv75g4sSJZ5SWlpbV1tZST4ZQk09BJM11xXEh8wGW/zbcmnPX4Tq46PHP4OGryqFNbtQDdWpC5qFfqC+XFnTjayBcgOlxpK7JwS9CM10ykBtz05xUefyhfTdZ/xYXeCPYhrA67SYcJDi+l01TLbh2sM5r/yDyQNUcT0BOROMqM1uE7y0KaU0+9SBjJso0yaWmgfIdu7CehxqxCMonH8062K2oDew6wq1cwxA8gx/ugTfoS1+x+ygMPa2z47k0PktooIRJCuu7X+Zwj2zFO/93Khzc6E3ibccpZfVw9TN7ca+vvvrq6kceeaRXcXHxoFGjRlXfeOONR7/1rW/VGu+7du0a27hx46bp06d3mz59evfXX399N66s5cuX5z/++OM9ly5durlnz56xqqoqi1Dijjvu6FNTUxN58803d73zzjvtt23blr927dpNuq7DxRdf3O+jjz4qLCwsTPznP//pvG7duo3Nzc1w9tlnl51zzjkW5pCTkwNnnHHGyZUrV+Zv3bq1zcCBA+s/++yzwosuuqjuwIEDeWeddVbjnXfe2Xvs2LHVb7755q7Dhw9Hhw0bNvDKK6+sNpfz4IMP9rrwwgtrHnnkkcq33nqr/RtvvNHVeLdnz578V155ZcfIkSN3f/Ob3zxjxowZnX7yk58cffbZZ095/PHH97JqA4aafAqC2VyX1p8HpfmTtWw0dh5JOpqds+kglZNfVibuBwvOS0VkNOj1a6FRYePFA7/IztLmCQw4n3xBqGIYfZfQAf7fZ9uE+9P0giO1jVD84Afw6aaDjncaqKdRgwK6Ne3muoLqCsgUnNXhvNW3FR28fIcfQj+jDSzThzA+VR269U0x6P/QR/DnOd592vF+o3GAzdZ1VyRYAvdkAz8UAaKATbdqNQo1J0XUk/nbW9kOP63eivMEFk0+kduFoPYe5nkjhOdwaPLhsuAiAHsSenFmjZmFfJg0QfGgcK2Qjw4dOiTWr1+/8emnn97drVu32G233XbmU0891cV4f9NNNx0DABgxYkT93r17iZFsPv744/ZXXHHFsZ49e8YAALp37x433k2fPr1ndXV1dObMmXsikQjMnj27/cKFC9uXlZWVlZeXl23fvj1/8+bN+fPnzy/85je/ebyoqCjRuXPnxKWXXnocVdfIkSNrPv3006IFCxYUTZky5cDSpUuLFi5c2G7IkCF1AACfffZZ+z//+c89S0tLy0aNGjWgsbFR27ZtW565jGXLlhXedtttRwEAvv3tb1e3b98+TW/v3r0bR44c2QAAcM4559Tv2rXLUxSfUJNPUYjkMUL4pF2QIKDIoHD12b3gndX74fujHBq9vkKsRl/LgQrmMwDWm8ZsQrr5Am7HR2dvgR7t8+Hab/QhpvNrI7fxQPIy7Z+LdvpSnwyg5obdQb6sbvd9w81wESWTNBwZ7mZU5EMMKb9xQD1U00gsI/POpK2PEpIyrhCigopUN8QAAODfX+2F+y4dwESDKCSbQ60dy4mGjFsdv9Y72jkSrE9XgIamOPxwxnIf63T/4FYi75QClgAuIjWsaDT5eC6IcPnNcFy8ImqhvpBipEkm3PtHJ/xlRZxRk0/oFKRs1FajSU7QuJOJnJwcuPzyy2suv/zymsGDBze8/PLLXSZPnnwEACA/P19PpdFjsZhm/E4kMoEzaXz4nX322XVr164tqKqqinbv3j2u6zrcc889B6ZMmXLYnG7atGmn0NA8duzY2meeeaZbVVVV3p/+9KeKP//5zz0+/fTTogsuuKAWILmevPXWW9uGDBnSaM63f/9+qqAZeXl56dEZjUb1hoYGT8p4oSafolBF0JEGhhxWOsm3pkxFcaNNTlKTNy8npcmn1DKqLlQwXfETjbYozOpCnY2IfYyo2Iao8ZUtWis0tAsz1w1oIiIFYr5TIQ6ixpafQ1RUe8vQgGKmIcC6cag8cTL9O64YiwzaJ98XO47Aom2H3RMKAq2pLIBdKEQjRMKnmbVyH5EGEXww6L3t3z/fAXVNcfeEKSQEzgXUHa29O7y2Do2PXHM9n252WhG41uGyEItYp12j63K+I6HZ1Nl+C9O+3HnU1/pCOLFmzZo269atS2uprVq1qm2fPn2aSHmKi4sbV69e3Q4AYNGiRQUVFRVtAADGjx9f/d5773WqrKyMAgCYzXUnTJhQff/991eOHz++/7FjxyKXXXZZ9csvv9z1xIkTEQCAnTt35lZUVOSMGzeu9sMPP+xYW1urHTt2LDJnzpyOKBouvPDCupUrVxZGIhG9oKBALy8vr58xY0a3cePG1QAAjB07tvqJJ57obggjFy9e3NZexvDhw2tffvnlzgAAb7/9dvvq6mpXn2eFhYXxEydOMPtGCzX5VIKJz7EoEbkxeTP7FLVxIx1cUIsSc7U+njDT5rq+1RiCBFUE3I0x+s1pkMCa64YjGgkkf8qSpvJzagR9QCSZs1nS6fz9R/OFuKpZ2kdESxKj6yo6fr2SJcMKQRToNXGcKc3PZAea0jSAO8f2g/lb6AQMBjmGmaXffEBmfRmhs0mjitIVgGj89K218J1hpyLr0aFlrN5/mbvV8rer9rPAvqcx1/W61/TDRyKb7hwZMtYJ3vkTjzNq8gmk/S+fbiW+V+ME0rJRXV0dnTx5ct/q6upoNBrVi4uLG//1r39h/e4BAEycOPHYq6++2qVfv37l55xzTt1pp512EgBg2LBhJ++///4Do0ePLo1EIvqgQYPqZ82atcvI9/3vf/9YdXV1ZMKECf0+/fTTrRs2bDg6fPjwUgCAgoKCxKuvvrpz1KhR9ddcc83RQYMGlXfp0qV58ODBdSga2rZtq/fo0aNp2LBhdQAAo0ePrn333Xc7jxgxogEAYPr06ft/9KMf9S0tLS1LJBLaqaee2jh//vxt5jKmT5++/9vf/vYZ/fv37zJ06NDarl27Nnfs2DFeXV2NVbybOHHi4bvuuuu0KVOmJJYvX76psLCQapiGQj6VwHhL6Cs8MljFviYJRYRJ2QN/2kuVbmlSUAuNBoGaW3GMEd+tQBEkZsuBCnUoEe3PSXQ5rECNB3XEjf4UO/LMLrBk+xHLM3J0XbEjmKU04kHZEKr4oG0iOp8BGZdOZo0lizBbcD26roOuM0Y4Tf0fFE8Uqc2FAs93idI0qjnZ7J5IILLN3FBo4A0f3K2gNesF18HgFsJrWaLzm2mz0xlj7h85Y5lIRXZNn6zC6NGj61etWrUZ9a6iomKd8XvMmDH1y5Yt2wIAUFhYqC9evBgpob3rrruO3HXXXZZNk1nQd8899xy55557jgAATJ069eDUqVMdt15//OMfK//4xz9WutG+YsWKLcbvH//4x0d//OMfp1VDCwsL9ZkzZzqElYZZMgBA586d4wsXLvw6NzcX5s6d22716tXt2rZtqw8YMKBp69atG4w806ZNqzJ+T5o06fikSZOQfgJJCIV8qkLC+kQbrZCGDoP3JbUo6DkhKWVQgs35Ww4FUq9IqCIYEwHUpwQxNlQ0NQVwbpbscyqMrouGceCRrTkjE+i5YUVtY8wPUqTDekAgpAM9kMOs38PIyxcyH8481GVGpl/4SmypZyyLJp9EYYShzRRlcIxmDzbhN/zmzsTI3U6DXaay7Xtjs49NnnLZ3ePoqXxM2QKD39F1eafeit1H4bpnl8KfvzuErwAG8JDopzUM7948ThF4I0SIloht27blfec73zkzkUhAbm6u/vzzz++SVVco5JMIL2xWJIsWsVnDmwSyQYW9Rnrjk/r7vTX7AQAgJ+qPi8ps2XDZ4ZtPPkXaR1Vz3a92HbP8jZubuqo2fB4gYmygioiYDsHKaVGbQBPxtuK4NYIe79cE1QpoTT48NQmdzbk7KxymdIwNI4o0oiafoDpkIUie7pUNyqDdXKbMS4e4zi7kMy63gots6bN5MKm61DtRTYHrByeP0dPtz9Mc2abBZ0DkXIhTlHW0jugCDItZKysAAGDR1iOOd6Lb3tUnH0NZbuOYy5yYeAGHR8zsk49igrXA7WyIVoqzzjqrcdOmTRv9qCsMvKEoRO5zUAx0fUU1NFN4fHb3B4EwH0PR4FoTW70icfXZvQAAoFeHfFP96h70g4LMFtlcWQ3Ldh6F7Ydq4R+LdkisiR60mnx+H2BRJj9IE9QATtZ8BxLKdII2eSga2+fnZMWxCM2X5PSzKr4xAYD4iWd2a+cfHQLBusb4KXRh6XpS0pYQeMMr7ea2/J8nPoNZK/ZZ3su0KjTO0SzmulPeXAMAcjSCaXiKQlzHAUvgDQpC7d+L64cnPvna8aw1CjVE9j2N2feeo/VcZRvdysrDeS6jXGvwccKIrCoWZ1z/BNbtBqX2PiFCeEAo5FMUsoRMZt61ruIEdzkybgr95qtGfUaUXb9okeL8VumtsTsmPPk5fOf5pXDds0tg+yGkv1Nfse1gDXzrqUXENEFpOjQ0WzUM7XMx3J+QgdIWaJ9PFd0+cNBo8vGW40jDXqxQ6JjfZowv7w5XDunlGx28kM0pXDU0JNePg8jAG7zs1iufFnng236oDu5/c41NWOSHJh99npV7km5/TjaL12R3D7xAbg/ZWpUoekQigpHyLN3h1AgTgcCtLxinnkhNPrluOXTLfxYQvtlsLUQ9713njLjvxLFK3jli5LuwpJvjnVnTMkLBn4LYbrdGQXuIloVQyKcosuWg7sURa2tFtjaBH3Qfr/fXMTUOczbSRSL0GzsO1abNyw1gN2bhDsWCtFYRSpOvbRYL+RjyswVU8FYXL1AXSDjeM6hXB4sQR8b6wlukRZAjgA6yua7cNuBFxuQwOJq8eQWUA7+i6xp+rzKRchkgmKwP1x2AM37xoXu1fl/2knzyGea6LOURPsAu4yt+8ANivY7nDHSoCFfLIIEfSBNdV1bdOORxuATiEeKZc7AEkOO7MNRNv9F0IF1wmNLSKI1kqwl6iBBBIvTJ5xPMPjao0jOVbf37GWu0ZkuhvOuYw7m/hn+HNB2k+HSL5obEBRenao80hEs7oVZzgVHoPBfCB4x7YoHjGWluBg2FSCHOlTY5EahvUtMHoxmoDf+by/cS8/Tu2Ja3Mgv87kurkAzdeelDhIA6ZOXl0o5AraFKzSY2BGquq3njiTJoN8sf5JrrGpp8GvP4EU3Wf1ZVBFKva32CTNNpwGI23Roh8jLgWD2fvz0akMg097A9GU/3u0fXJb8fMPUjW3r/uTHqs1W6jAoRoqUi1OSTCQ8eur0wwMc+3mL527ywTH1nvakO7iqYy1CZn5NoK/nlR3DZXz4XVldzPOFwjJ8tyHaTYBbgNmPnn9EFAAC6FOb5SA0jAuwm3E2uSrDz1l9dXqasEN8OFK8iRQe/Y8wZUNyVz2ddUPPd7daf5rlIsFwGyQRRk8/VoTqjVgtLWhrTbwHNE9QMlWMi6pMmH38MtYIAACAASURBVEfgDQOi6aKlwH9NPtI7dmJIawmPkM8LH47Fdag52Zw1ezeRVO6Q6PbFfYyiXajIcK0hUjuSJzAH2VyXoCVLR1IaWbJFC0GJc889t2TWrFntzc+mTZt2ys0339z31Vdf7fCLX/yiR1C0ecU555xT6pamd+/eZx04cMChaPf+++8XzZkzR5ij6VCTTyWYNRckrcmsfvhwTLol8FtnRDPT79T/zXEdNlfWCKtz/mZJZqDZsYfLejx4WSl0bpcHPTu0hZ2Hg/cdCJA8VITd7w5jkyhTc0YmjtU1MZuzdytqw11f0BczOua3GXL9LqkFP9Zc0QcpY//AKwQVob3otQysgITym5BCYfZiuJA2N/XQsaLooyXBjzlN6xPR3n5eTfD9ivRs1POTmStg8bYjMPue0d4q9gkJgYtzEyKwoKjSSUJTUh/z8EE3Aa3I6SJr6qGE2zR1mXNJE/IRCGkJ51xVcf311x997bXXOl933XXVxrNZs2Z1nj59+r7LLrusFgD4gwYQ0NzcDLm5ct3zrFq1ajNv3nnz5hUVFhbGL7nkEiEHzFCTTyZ8unpQ7ZYORQ+VuW4rOrBlI8LuSWpEnNq5IGgyLDhwwqoZGuSNJ5/mgwRCCKDRzFIR5zw8B275x5dMeWgP9/GEDodrG3nI8gW4tcGPvsP7x1Jz5PhhgkxdntjiuKBJ2OWKFNaYhVri25+/QFG07DtWD5UnTlILW2UK+Tq3y3P0HU1topYov5fmxdvkBPSQBRX4BQtY6eWRYeq6i+CQkQrh04tQnjGXURfRrHRv2F8N0z/a7NgLhOfG7MStt956bN68eR1OnjypAQBs2bIl7+DBg7njx4+vfeqpp7pMnDixLwDAP//5z079+/cvHzBgQNmwYcMGAADEYjH40Y9+1Kd///7lJSUlZb///e9PAQD4/PPPC4YPHz6gvLx84KhRo/rv3r07FwBgxIgRA77//e+fOmjQoIG/+93vus+cObPD4MGDSwcOHFg2cuTIkr179zoU3i666KJ+X375ZVsAgIEDB5Y98MADPQEA7rnnnl5PPPFEVwCAqVOndh80aNDAkpKSsnvvvTcdAa6goOAcAIB4PA633HJL39NPP7185MiR/S+88MJ+L774Yicj3aOPPnpKWVnZwJKSkrJVq1blb9myJW/GjBndnnvuue6lpaVls2fPLvTazqEmn0qwmOuKLJbD6XLAUPUA5RXZYhbY2pFNvZRjc+ac8TkZPFRsR7eNdkvaM9Ja6f3+g03wz8U7Lc+CboZ4IgF1jTFo1yYH3ycKd5arBgZjeer683L/kiB7icfZvRmrUtFmxSLTIvEEm79mnmp4ShfVZ6P+OB8AAC4b5G59JfvAPvS0TvDxhkpbnQR6BNfPUl42++DkhVABrw8asiig+y2l0cwZRGNA9yJoaI7D7iP1nukjAes/Nv2/s2CawDWofQgrjbuP1MNzC7bDA5eWQE40U2BMgmmGwtsKKZi6eOqp245tE6q90K9Tv/qHL3gY6zC6e/fu8SFDhtS99dZbHW655Zbj//rXvzpfccUVxyK2UMvTp0/v+cknn3x9+umnNx8+fDgKAPDEE09027NnT97GjRs35ObmQlVVVbSxsVGbPHly3w8++GBbr169Yn/72986PfDAA73ffPPNXQAATU1N2vr16zcBABw6dCh6ww03bI5EIvCnP/2p67Rp03r87W9/22eud+TIkbXz5s0r7NevX1M0GtW/+OKLQgCApUuXFn7ve9/b/fbbb7fftm1b/tq1azfpug4XX3xxv48++qgwpYUIAAAzZszotHfv3rxt27ZtqKioyBk0aNCgSZMmpW9eunbtGtu4ceOm6dOnd5s+fXr3119/fffEiRMPFRYWxqdNm1blvRdCTT65YL2htZgCiOMy+P2j9zp00Jlutf1gnluryOa1uu1/+3OZkKZxLqlcSx2tZOGra4zBIx+hta1VPGfncPhbao1IR9nEuSCoOwglGjmIRbaBdmR8srHS8SyoG3JD4PHhukoo//XHxLSBavJRutfQQAzf1zSAlxbvhG0Hax3vaIKUBAEdt9hygJf3ehXy3fDCF57yo2DV5BNevAN8Dv/FEkYrpJbJdiaNLGZKH5TWEL4auvpV2RE46HAjX6iMT15f8ZbMY+6d1OTT4PLBPZF7PZXNdQ3exuJn1w32CxGW6MEGhvTpABcN6ObKF0OlDLn4zne+c/T111/vBADw9ttvd7711luP2tMMGzas9uabby5+4oknusZiMQAAmDdvXvs77rjjsGF227179/jatWvbbN26te24ceNKSktLyx577LGe+/fvT9vl3njjjemyd+7cmTd69Oj+JSUlZU899VSPzZs3O6LTXXTRRTWLFi0qmjt3buGll156or6+PlpTUxPZt29fmyFDhjTOnj27/cKFC9uXlZWVlZeXl23fvj1/8+bN+eYyPv/888Jrr732WDQahb59+8bOO+88i3DipptuOgYAMGLEiPq9e/fy+9YhINTkUxRMEb9k3ljhXpB4HyITza3k4m1H4BundYRTivJd05IwQUSgDEmNGq4Z6uNtyiiABlQ6VAMAaFpw9PAF3pATdRxbhp3G1IO2zw2HT9rUwvNwiYBa1EDEdijoeHwD3JPzFgCMtTxHbsIl0sUKWYE3aOauiPktqi1/895GaJcXhQ3TJggqEQ3Rwgze0kSsl/Y5oALMgj2ZgiMvJQunirIbZK6nyEO7YHtdFcwHHRfYwZNEhWzxl5sJpOEkmCTQ4usHnfW45VKaWJA1YZMvUQJ+UVqbjRxCvoSusma8/yBp3MnETTfddPyhhx46ddGiRQUnT56MjB492qGqOnPmzD3z5s1r9+6773YYOnRo2YoVKzaiytJ1XevXr1/D6tWrkRoaRUVF6YFy55139r377rsrb7755hPvv/9+0bRp03rZ048ZM6b+Bz/4QcHChQsbx48fX3348OGcJ598suugQYPqU/XBPffcc2DKlCmHeb8/Pz9fBwDIycnRY7GYlAEZavKpBLO5LmPW4/VN8MiHmyCGcDYrE9TRdSm+6MevrIBrnlnikaKk+QsNgjBrzOaFRdbmez9FtGFVNqlmYbX6Pak+hSwQZb6EG0pak1NDKtthb7FLFn0X7sl5mypvUHMObeyEJiYIAbuodYMceRD/rq4p7lIu3TMUjOVJVKsa36GE4CMgGlD1+hVdNx04god3BtRlCcIWVsZ8J5obEv5WYUy3NIjsX1z3iPQ9i6qCLJDjMNfVXS47/ByHjGtLJnCNW7H83+CmyVfyy4/gRIM1YFlC16ldmYSQhw4dOiTOP//8mttvv734mmuucWjxAQBs2LChzbhx4+qefPLJ/Z06dYrt2LEj73/+53+qn3/++a7Nzcl+raqqig4ePPjk0aNHc+bOndsOAKCxsVFbvnw5UmOopqYm2rdv32YAgJdeeqkLKk1+fr7es2fP5vfee6/TuHHjakePHl3zzDPP9Bg1alQNAMBll11W/fLLL3c9ceJEBABg586duRUVFRbFuVGjRtW+8847neLxOOzduzfnyy+/LHJrk6KionhNTU3ULR0tQiGfomDdQDz8/iZ4fuEOmL3BaXaF42Ve1gbeA7cbs69ICXyC2D+Z21xa9abvz2J5n1DcOXNl0CRQIyv6THf88L9qhdGqDmeUAxbFz5H+dwJou9eW7aHW5JNBHe8n+3v2ErN28dBMzBPwVMtN+W5SjW1/sjHjbkem9pIxLmjYwMGak/DsZ9sdeVlxoqEZPlp3wPGc2lyXq1Z+0IxflvEjwsTPMp85GsTv8S6qz8xzoWeHfG/R4THPcZpfLOOdOdiF8T8PfwX5+05R5Ts1SA3+I87M2L4HcRPyNcUSsPlAteVZQifPU9Wsc1oybrjhhqNbtmxpO3HiRKSQ79577+1TUlJS1r9///Lhw4fXnnfeeQ333nvvoT59+jSVlpaWDxgwoOwf//hH5/z8fP3f//739gcffLDPgAEDysrLy8sWLFiADFzx0EMP7b/xxhvPLC8vH9ilS5cYjrbzzz+/pkuXLrHCwkL9kksuqa2qqsodO3ZsLQDAtddeW3399dcfHT58eGlJSUnZNddcc+bx48ctwrnbbrvtWM+ePZv69etX/t3vfvf08vLy+o4dOxJvaa+77rrjH3zwQccw8EYWwO4QnwWsLMYIF0/SYvMUdY/xuQwaRCGjieE/MeZlRWVfGn7WwaNuLxOqHQhpoKrgiqYt/RacHqtvdk/UQuDptto0pKZeXgavfrnbMz00sI+Hn7+9DuY/cBEyrUFi0MJ30uwT5dfHT/9AomsSwZ14LhZzImreY/9j0c70b5mafAbc/JECANzz79WwZHsmGitJo46Eya+tggVfH4LPfzrWEometvf8XsvEmy+K2YOL9GMWFFh5ltMHIn/d+IshAY1IKIL0zTxzXdd1Iu9DlYg722ia5qsrhrRPPmQ+MXQ0xcma7QAA+blWxSidUpMvG88C2YZbb731+K233rrC/Gzy5MlHAOAIAMAnn3yy3Z4nEonA3//+930AYA+W0bB8+fIt9vTLli2zPLvllluO33LLLa4Rtf7yl7/sB4D9AADFxcXNuq5b6Jw6derBqVOnHrTnq6+vXwUAEI1G4dlnn93XoUOHRGVlZXT48OEDhw4dWg8AUFFRsc5IP2bMmHqDxsGDBzd+/fXXSJNkHqi5A2oh6NQujzsvk08+l/cyzgbpMhGq5GgV9pBdAoSOXFFQrUnIB3bfyOCH5vjhG0RtIGf+8Fwh5bR20GrPkFxV/f6aQfCDUaeLI4oDuHHlh4AEB69jnTU3rcWWCoKAxlgcKo43eNJgEQFzFEZVIdWnsqlsN1ZQ22hVaOCdW/uOJd0qNcash2/atdOPoUJrdmsXRLA2yaEacaah2QhWHilqLszZWAWPfLhJTGEI0JJp/34erV03TT5/Lvj1NC24d+h8yf9R+xBRdNNE122TaxV1JM11kzQpsFyGaMG45JJL+peWlpZdcMEFpVOmTDnQt29frOagDISafD6B4/5GAhXswDFi3q3z3xfthN6d2sL3LiAfGn05FPhg6mWH+fYoKwRGrRCNzfibwaCE1U/P2wqPf/I1XWI1WAcA8JMy8syuQulorRAS1dXnMY+qDytwyGINZlZ4MsPlNC/jTXvf62vgg3UHYNHPxnLVLwq5HiPr+gGpPvlS/3NF17X9z5rPDhoSdJCvyWfnL0RrXUMTiaEBzWmfX7gDfv7NgSzkWeqlS+tMbKdXRpPK6CdRfOKHM5bj65BsQaMBQvHBo0Yi+YJHkQUKAYO3oaYPDd+jmXY0n5+fY9XkMwJvhEewELJh1yL0G+rvgLIZHpivSL6NY2UiqtCBrEqOwmMfBzrmqSCi/Q9Wn4RvP7vE4uhX1rIS+pAQB5qbQb9BLeADAKWkfCpBIU0SWvxryS5YsRvpqoQKtGdTpDmNQg3RHEcT45VEGS4sRJTtBV7WArOGvhfM3ZT0ORdL9ZuQtuBYOqMevasXP/gB9h21No/Le+tyI9iUzjhkczSe9z6z1kkrKPN73rDUV9+UUcIIYnrzapDJAJ9/OXIms4l4zw5In/meIKpdRJRDH7QQ0oxZzJlNLIhC8tT/Gji/l4YOmjaiSWPXJEzoeuDakQogkUgkQjlnliPVh1jnGqGQT0HowHiTLvUmGO/bAZcGebtlYiVBmliZ4TAlEkzWS0t2wfLdx+Dfy/akn2VzRCdZ3abafRrJr2U2aF8GSaO95dxIOVLbCFsqa2SRQwVF2BESv353A1z37FLu/F7cA7A47RcJVH3NmKjxvhxm7esb40LBE/iK9bOs5odseb3U61oeVZ0KT0AMRE6JhK7LXwF5NPkE9wu1Tz6htdLU525umPytww/+hdcQM6cTCVUvE+g0sRh98qX+/8sNZ0Npz/YcVPkD4pcjXV946yQN2JQ1ZI0JdvPrpDBNaOANx9/uBaHOqDSuTLJhv+8B6w8dOtQhFPRlLxKJhHbo0KEOALAelyY015UIXj678OtDoHOo+LvBy0JDG92QJn/QilI07dA2z3sE64ypjNlG13OxFqiodcYK1RbSBEnI5yMdvPDhyCgM459cCIdrmwKloSVrwdL75MNvwlUYTVhNPkW6Lnn4kdtSnmKosLaTx09xHsQIaXX8GtAafNhKvaT1ULTwy1jKrvT9EpiiOlGjkHat8doCfswbGb1k9P2Z3Qrhy538Guwk2IeXpnFcqvg4RN34Q9BroNtlFX4PolvSoSDKXNeOBGXgjZaMWCx2e2Vl5d8rKysHQajwla1IAMD6WCx2Oy5BKORTFCIPnjhGScMccbdHVrkVI7ekUcFmK1E42gkQ8qH8UaimtcYCWcIQ1VokHvSuySO0AGcPa9OxCvi8HIizee7xwpNgyCjDb00+xDOsJp9cUpJ1eLzgykowfBtVOwTUVtkw43mj2Lphz5F6+GRjJQDwtYPo+0PaCweVZHw8pHiNKEtOi3jGVJs4yBDGkgI1yIIG7G1IMv6RQTnRtFRCfWzAU0DSUg5y/TQH3kAh+DaVj6FDhx4EgCuDpiOEXIRCPp/AetMvMrquF7gJdpCq4oinSprr2ugw0y2EQsSGxSLwE7iRUaRJWwSImnwoUwxF214FjT6RTeN1utj5Uv9TCmHrwVpvhSoObz75UpcUprdBDfUbXvgC+fyMbu18pYPl8sfVzE5ga4o3EZRb3so9x5jL4Jn+xvhXWSFQ1qXSpU8ugJPNSQkizV7DTgb3Pg2nleMtOzMSCR2enr/NUxk64qKWNg99evRzo0qeCL12cmVc0MoM5pFpbwl028rUUqp8XD4GKTOpFHiDz5ciXV6U3z2cME1Uz3Jp8iXoeGJrvBgO0bIQqmhKhEq+cdB1eNGKMf0Wr8jXIpDW5DM9My942eaLSBq5ip3AaA9dKpFtpjhQsrJoSBtzMcumIRO8aESk20Wz/CcdtCT3O6UQbhrR1/JMysHT9NvSnrR1+TwhUesKbbsI42n2w57t75W7j+GSCoUKhzT3g7HYFnhm/jZYt+9EWsDHiqI2ybv/Mf27eaLDPpaoxpYOxAZjaap5mw/Cn+a4B6wSzTP+tXQ3U/psXX6oLIEYp59xv+pn5FOeeow5+3WV05+waFNpHfTA3BY8t2A7rNl7nJiGNA4SBAkl6TKdBTwCbD001w3RShAK+RSFX36ixv95Ibz8BX5TQrNBZd0k8d4QX1LWHfJyBAzZVPWyD/dpn1YW7T25dWYjcE2y9feXwes/Os9XWgDczJTU70CVhFYqt1ZrmIvU32hLZ96Aq9pMw07rlD788B6CqCxNzRMK5/qCq3Y20H6i35eL5MAF6He0/aXq2BMJkWaxuq7DYx9vgSueXmR5ztKOg3p3AACAsaWniCMM6AWuotojRrCDtgTUII1fTB7HSw9A7Ydx8wYtwHcnRM7lh8nyBUsvGx1GmTIFMCiffMxlpP7/uorOEsBL87tp8snE9I82w1XPLCamIX6bju9L3jZhHVOoNAm9dVzyhggRCvkUBZO5rktaDcPMdADYUlUDU9/BBmYhlMlPDx1TdiaStdAZGwvLxk8A4zeKsJjrmt5nm1NxvxX5cqMRyIlqUutGgXTDmA23f+GeBQ0VtHr8Bo7HuPG33763wbUMWfBd64wRwg4FhHJYq9Axv4OGbvvfgGpaoUFCpPuSjHmu9bnxN6kms9AmP1f80YBeSC2mPbB8y/bYLXBAMguNuTMf3dkqZJAR880oU+a8dfIiD1I+BKjHOW0QFt3becsMjaFeZF2M6ZM++fj2ILTgKSah6xCJ8O+PQoTIFoRCPonwslk5Vi8u4qSQ9RLzKTr4t5EWXY9sPp4JvIH2yScSLXdN8v+UFifsXqMKSvnG2TQugiSRZQNZ1xiTSIk70je5tuctaYNHOxbsyWZ8sTuwdqCv1p+BjjOFd/VXq9BA8psS+8UZqS1I78z8NtsuxWghUlhS35TkqW1zrb4jNY3FJx7+YO4FtP0nqjlE+gCUZcYOALC5slp0kb6Ahr8xu/JB7JlFw0G34mzFbT76ZfWFrd9FSI4N/EhB94ETJzmpIiOhU/rkU3xshAjhhlDI5xNY2fBv39sohQ5x8J/7aaAJ3dH8Y9FO4nuvh7TMLXAG2XxQkXVoVa1FSD75/Iz6RoNHrj0Lrjq7t+VZhny1aDWjvikG5b/+2Lf6km4FrP0aaQWrn5fDurEJV3UUKShvR8IXDVLRWujCLAl0078ZUAeE8dh0Mlueto38FAbXN8UBAKCAIUCMHfaDObNjf8xz2r4UJfSkj+ZL1m+kBW83Tv3vBscz58WTmL2oSMjQ5EPtmYWWL6wcfEkaIAKfeOgAXUcUKAlYOola54QLHDBfpiJMydL1ovPvO9aArzidl71tQ598IVoLwui6iqIxFmdKz+c8liINRxl+m3WKgv1b3lldIaRci08+ISUmoZKmiBegBJ8/uejMAChJgmiuq9jOoDDlJF2VoUBLx4mGZrmE2PCjl1fAnI1Vlmct0SdLj/b5UFmduf3G8kzbR9vnoK6bDly+B46gS+cXXdhzj2DBmhtYNLG46xCtLa9b/0/Xw5hfNYi8rDOb63r93obm5L4x367Jx7DzWF9xIpknIAGrqH0N7SUOrrqGpjh8YlsziOVQp7SCxmdZQgeIstxx202SZezKpRSZLFTmZarTXFdtmGV8tEGVaEzQzZDFj5PmuujyxQlc2ZHQdeIYC1o7MkQIUWgFugzBgYZN/Oq/66H4wQ8czwk+g5E1keqSsV7S+HiRARHMl6YEHQDW7jvhqR5j827xySf0cJD57YfAT5rwFvUswJ0XSZMvqpgmH4qciKb+BqUpxhf5kRd2AR+A+pt7Hvzh2kGWv73IpI1RpNiQT8M/rVo588ko9XBtI7yzSsyFkleIXkbwWl3sfcfT29mgOZ9IiJtjDSlNPpS5Li3qmuJpjUCR8Ho/1j6fTSfBa9//+t318PZK/LwUJQig4WO0QUS8pGGFSF+S6TJTn5k+X/hxecLjks/PLVZKsxZv9iqkCv68nOa6vOPHPu94lFXMgTdChGjJCDX5AsaMpbuRz2UsoCJLDII9SuPJhraBrYW8RvLN+OTLPAs1+egQZJCEOEH+pJgin3LBJGhH5H1vrJFKBxUU3+T9l0OTuGNBXvr3sNM6waj+3ajy2VtC00z+kXweY7SH56B7TxT3/eGM5bBqz3EYeWYXqvSoYWuNdIl4H9BaIcJaQBRUXS5F7vWMC6ogfcfixtorX+xxzwu6sH6iN9dFP999pN6azrUcPrtKpJaTrTKSn2Aa0NBeVd0IPTrkCysTAGG2SllmRNPk+a92aBWza/T7yUuSPvnwZxFV+RpAclzJD27BXlBC11Xf/oUIIQShJp9EeGFiIjd+IngZ9uCFofNkcxyuf24JbNjvTRvODA00Xxe0vKi36YH2yeepSAtk+EQhQVbbq7bYZpO5LvrArxaNKKzYfYwrX7ZfVLDg7n+v9pT/rf8dmTbnZoWuYzT5FDpQiNDSohF+8fI92mxVKefizRbVbP7yREDUgRe3b/A90qrCs13kuqrygZ8Wova+tEs1zaWCKI1QVE3LKdbCWIo3oN3jeG+vGUt3w3mPfAqbDtAHAZGiySehzF9dXmb5285TWF0jAbC3uZfPcouuywLKgNN4WhDfrRPe60Bnju4FPG2rh5p8IVoJQiGfT2BlRCRtIq9lp/N58t1D8Geg67ByzzH4atcxmJYKIMK6SUJ+k888OderkC/1v/nbhW7oVTpxewDqAKaquS5qYxDkwQq1gdJayLiwQ/SQMNqupcwjAP42QgqL7RoPPk1KLz75ZPclViuBo15WvkEf6IEforoY54svXY+YarIDLh0iQ7Bhh59my0Zd335uKYx7/DNsulOK2iCfi2oNWtcaoppfdC+a6YrHPWryuXzkku2HAQBg1+E6hjIp0lCXZs0g8zIVZbrJXAYhD3V5DPycNJT9cmGEzUtojARRk0/MjOEpJWEKvIEUXLacLWGIVo5QyKcoWDd+pCXR0wbPhQwdU35aBd5IxxyhzZle1LKPo8XqSF33LuRDmOuKhN+O36Xpkih24iNp8pkPDipohxCF7T7Ska4zi3ZHhklbFpHsChmH+WzwayYTuOEheqy7lUd6a8mKcs7OSosg7uHVX5j5Fc8w9LT18YkxtPUQCdcJjOakwBpcKUi129G6JthBEBpdUtYd+Ry392XmQ5IFI3JhpS9G2JOIELZlgizRt7GM+ZF2cSO8ZGcdXkAqgbR/5KpL14l7TRbtTr8VQpIac7gyOWnRyX/TwAi84TbOWvnWJ0QLQCjkkwgvrN6P210vcFvg08xR8Gf42Sq5UW8c3mgis/aXSPoVHyL0QHxHkGsrSZNPU4xjItupxQwMucBt8VQ8AKKCM6HAcESz5ZMZ/44NtLV6dafgF9wOCsbB2m3a0h6qvWllcOQh1Ig31/WHw3sT8omjg4QnvjNEWFnuY0hYVeqDUosUK8Qn5AEQI2ygheGTj9dfqRtthlyKZb7omN9mHK9vpi/QVI5IU0r7Pt7eFjcMP5W5TFJ7ovaPXvYUbpp8zOX5uN9Ojiu59rp0LjesaRJ6eHkZonUgO3bJrRAsjnZdLvFN7+y7EpqyXbQLkLdIppDvHm6A7PCbKXfDmJPQAuWTz/reowmGz4dwWZtY1YQqpLmnXnRdtehRqyeTwM4ztZpOObjxr6DRxmNgJFrgho/rWHfVgldxtiThVUs5o8GPKZ/aJx97noPVJzN5PHyHX71zShF9sANeaBr4ppZC226yfWLRbqGphecBSkhJ0XXpwCccJEGmTz6RQ+OKIb0se3k73cVd24mrDLwHSbFDZDN7jdCLPO8RMr+2bA8crm1EvhM1fnhK0U3muiFCtGSEQj6FYOY5vCrfSDNXCczMjTq7dgKzTz5UmSBno6Xb/jfQJidpRlPE67weB6q6RQAAIABJREFUnBsWkfS3lFt51HdMGNTTf0JSILVrkFELUUBSo5jgL2i8tmwv8nlLbCVNA+jdsa1rOvs6gfZvh38nC4mEDsfrm6jStskVaeaIh51ns/JwcW4m6N4h0zGuFX4KIEXXtZEheAAJbodQFYW0OIpohJ1+7ydQS6lIGmiFCNhUEuYMtV9N3Vqe5+i6lDcSbOa6bu/Zac6cF8xUscP8GbnRCPzyWwMddRjgu7jFU2Yed26+SUmoa4zB9kO1WJdImTrU40MoyNJ85TPXDQNvhGgd4JNehKACux+6DEReBolgZXhtBpw5joBKfSiTDG+qLBkTCDmE+23SLas21HeU9WpvrdvHbyVtqFXbF0QQ1zRagJs+Z7AGdLpeHfJh/4mT6JeC8dWuo8jnLXGTp4EG7901CvYfb/Bc1qV/Xpgu0y/8ac7X8PT8bVRpRWjysU4VrCaEhylHG0AkfWnEUQe1mZ9xuJZ8AKMdUzxkmOe1LHNdlmJZvkEW59Y0gEsaZsPRyJkAMIIujyRaDEwY1BNeW7YXmliizDGAWshHkYxmHMlcdtM++STVkeDY6rq1L6pfXf2Opn3yaZ7WHQ3Ql1S6Tr9HIYForpvAB5tgwfde+gqW7TwKQ/p0ILYE67ij9XdO7x6CZgKh8gUHc+ANElTwux0ihBeEmnw+gfXWl8lc1+0WXwIsZjSOl2ZzXd7y6RYirrIZ09ecjHHVg3IiLLJ7suP+zh2kwHFByGFIPvmUM9clOWRWYIOCa0qc03U/gRKQZjs0DaBzuzwY1LuD0DL9wscbKqnT+qbJh3vukQE7tFZcNWMoy0VQLNiCzFof0oQr+XDWyn3IPDLHFE54z7oH8/MSTdRhkkTynbVPwa+q7hZSjwj06pgPG6aNl1Y+ffehE5rHS9DLPuk8YN3/476FDrTfue1gDfzm3Q3ENI0xduGtQadXgwmLoB80i9DNqcXOXhmpPUXx2mU7j6brImk2Ip9lweFAXHRdRmUaXQddV8/VTYgQMtACjznZieIHP4A9R+rTf5MEDSSwZBPBYt187vBulLEHK67SKOskqJN/seMIR4HJ//Ah5NmLFJlflfpUMzcgmcorp/2lGDkkNMbisO1gLQCoIaBu1be0DHPOz1bKYQim4ZdPvvqm5CXPzB+ey5TPX5NX02+k0I2yHMEkv7RkF/K55eKLUCfP2mAWEBg/edj2jX/7wvFs8YPj2AvyEbj2UpHTedPVcodITT6R+ajKFlyXm/uftAYdZYfc/q/lMHfTQWKaxmZ2IZ9Bp1cBDCm7vSkiaQGauA5tt/VdGB9Zlv4b57ucBvFE8roW79ublTo60CqP8CqZCCOboiDU+kjay6t2JgkRghehkE8hrN13Iv2b1wcHKlckQhe9D1ke5pb+L59udclJrnOd6Vtp4efNy9aqWnjRdECp5DAtRN1Kilw7zAuRH2uSrIVPtUjSJAF7JBt88gUI0kb5wVnr4OI/LYATDc38wn+BQyWjQaXW+AsCJN5qMXeSLLjKY4ho7peQ7/svLUc+ZzWBdYNrkCu6YrjKtsPrUitDc5BaJGQW8nn4kFV7jjue0fi7VBGq3U0BJGli6Z8cnc2igjrwBlOp/sGL5i7ruM+Y64obKI2xOHMeUZp85u/Qks68IQeS48e+3vNc3LrtGU75+MfwfN6TzOWisGF/NVV/8gQzIqajS+ZBSC5Kk48Nxt4zLdxVzJIoRAiRCIV8CkGEsMPOOKMRDXIZtCNosGrvcVj49aFUfQgaQHd1nHv7jK/IlSAyapj6zGAVxuEWmteX702ry/NCRqQwM1qKbEK172AVsAdJPmrzp1hzAgDA3xbugP+sqgCApGYUV58LnkiyNGyDhBxeo5n+lQs2TT6nua7MvhOtd2SQSlsuaX9gufBB5mUgDLy14/oK9wu8IMx1WwNUYF0sY4e2py6MrIEVcCP0rN1IXTZxP02hfWR1SaO5ti3Nd/Nekhj5eGlwzWf8oOwQmjnGY67LI2xcs/e4Q1PRTl7pxr/AtvyJoMUasJp8MuF1XhJ98jGU7tauVo03Z7myAgd6KZc1q+ErMhLRQiFeiBaPUMgnEazMh1t7j7DBb5sbFaLmbU4aIzlRS8FeJysvRdZAUcgby9GRNC1l6/a/jQeEQxTHMq0jNyziFsmWEnjDa+Q40aBtVxU2CIopFjoGidFGv/9wk+U5T5fn6w1CpTg0N7nZhmw3Qc5hGND5uZntix9zMSelZZgeLoLHDa1PPi7tNEae5uXTVu455l4P5zh9/OMtcMVfF7mUjf4dFNTgLyq0hBO0Q/miyGoAAOhbt466bJyJqr1OJaMkO/aoYsvDgXaU0PQbq7nukdrGNO/QKE+mX+06Clc9sxieW7gdS5+mAZy2643k7+Z6wG5SGOD3iCGR6MUUmARqTT5J5VdV0ylssM7fHYfqAADgtC4FTPlChMhGhEI+n0CzyPL64bNWZP2zbR7eObmozQ1xjeQ2y0OYIFBsQZiFiRJX67SmhixNPstv9TaqtFDjEJTEG8v3wuJtHP4XAwJqTqh5nLODsdOP7oTXD18HY2vfF0ZBS9T44f0kUjY/m4lF6xylyScT+Tn4CzMUqA/WiEJp/NTtOFSLPAih8vp5jyJyuNi/5en522Cdi6agai4V/ISbj2SVYJjr/nTCAHjs24MBQOxegNpcl0L7jSq6bkB7MENokaQBDVdXAGmrE7qBQrN2NqOi6xLSX/XMYvhg7YEkHTa6cDAsdzbsr3ahL1OOCE0+1nHqXQOO4D/ONYWIWvDg/Ta3bA+8uYYqH2v1WyprAACgtEcRNo1KZ5IQIbwgFPIpBDfnuDSwL+ZtCREIRfhSQBVhjlwkklfK3qiKZuwJxMZJrE8+cWUFWZ9KPvmeX7DdPZFCIM8J/0929p7EBQFIsFryHE76AP1Gw1IuulBQ8NzrI1hMfPxDDotPvlx/ty+89eEup+yHI1YuOO6JBXDuHz515EUd6GkPYkJYMc1CzTGoaNd/y6G9FUzyhqY4zN1YBQB4YY7KzfCTi/pBaY/2wsuVHXhDVjnp8hBlo+q43EWz1VEY6nXa6oQODiEa8mKBrUH2HWuwlE8z3w3N75hNoGhlAda/7GRx+eRjzuENdEJmOpDSocYcTx10dJBLa6I096ahyfwtu48mA1wWd2nnmk9lvhkiBA1CIZ9EsN7sxQQE27Az5gKCJh/NIszL1L2awvHWy7peNyFuG4Uh9RGyFAtaSsCAlvEVwUAFDY0fvPQVlP1qNnV6HTi0HvTkPE0IXLJUaDvRkPFNfgY8YtPk81nIZ6vP7UBEC1Troi/PKLVtfNbkcx6a3fNYHFiQtBY5Vge/xqsqy++v310Pt89YTuULkQeyvhPXT4/O3iKkfK+Xh277K5ndbx/3XgPn0OamnTo06by0Py0d0RSzsbt8MY8te1l2uvzUXqOFXeEjymqvy4CK4w3Ic5B5zFG6t2SCOOE6W0HxRAJyIhqT/98QIbIV4Sj3CTQLngjfZPYS8nIieAfznmvDM1jj9kxkFE1aFX4W/GzWWmFl2ZHW5DNtI8TehPkLWeYoIjRYWytUMDn9dPNBqG9CR9LDkcc8hVNCPl3g98rQNvYD5oubO8f2s7yT4ZPPzxGWy6DJl0/QUpcB2ebBbuuaFzbJPN08CS2D5UlmnkgrTGSFSsG0dh1JaqbUnIylO3pzyiTNAI/gk1dYSh11mrN06pS05roUZWoU5bFqFInAH2dvtpWPrsCddudelQSafYcXfkW7rzE0v+0KEiwyMZl7KFLAFFKfTHvfGmAmSlgXvQ6pC6bPg9eWJX2Zc/FL2nlmS2ecj77RtyNlRU7UNcbgo3WVTHkSesu84A0RAoVQyOcTaG4nxUTXtf5N3KhRVIeMsERBhwgmai+DygqIIpGZ/vmbDyKfO/JwdE3GUbq4MnH5VdEq4IEq5rrN8QScRDiLfuOO8wOghg7G0Hrn/y6Aa7/RO1BaAOjGoa7r7AcAGZp8wkryFzkRDa49J9nXxV3bwa7p30prT9HyXec6gU/r54aYRbDQs0O+REqSMK9/+TZzXRGHfgAxwmb3oB2Ughef+pq2n3mWBotxXqqeIA91VEIkD/SlLz8J9WQrr/MCr+a6fmow22EPbpc217X18bOfiXEvYrjPoDaJp1iG/QiolpMixF4XTtAPoDt4YTrYEAO5pLRehFYGZizdZfk7N7XA49yfOJ9hlC808lrg1gS8W3WS8NjLPHvw7XXwOkWwRTMSuu5apxonkhAhvCMU8kmEmSHOSflMIYE/uq7pN9gXO0I+pltRvhtUfnNd3XHDZiwUMn2fiC03WbK5D8Q6lvZ5KZJU3Z6Uj4wAqrbg6mcWQ8XxBsfzs3p38KF2PhiblbNP7Qg/v2xg8lkWbFHYzXXjqXziDl4Gf1FExkyN3GgE62xbxrFUk8S/vKIgL8fX+gxNPtY2cLvkoe0zes0k/rw8sM9lVpd8ojXEFRqivsAyPynS4OD33JYtQ/MqY2K1GBFpYXL/G+igA640YJ+Tacto8tGBTpOPvj0abJYAtNp10bRPPjwP0jQNNIIfcZGafEseHAev3H6u8wXj0LCP3ZxoxHXCBLU2W8x6WfKlEkc0xnym1DRnB1S9EQqeCRCsoD9ECBHwd5ccgggh5rp2DQ2GtLx1ENMadDDySoMRm5f+bPM3lelOOXW0hMPM9kO1xM24X0vs8fomR4S2NA0awOx7RsO2g7U+UUMPoqJuABsU2gM7M+9JJDmBSE0+Gm0E2ThwogHmbjoIt553GnWenKiWPlAa36Alr+i56SBq8kHw2lBBwdykJNcXoutCvqeeWyjte0nzEgGaFqLWOLXkocukimZ4CDJkm3ULDbzhM/PbUlVjDXAn0OqD+J7aJx+FkA/h7hpHh90nnGeffLgMmuYYFzz7ABw/7dWxLXthFMghaGuwKWvQ1ymbjbqZiNNdTLATqesZBZKgXUuECCEboZBPIljZD2/gDVKdpMiuIpi4KDMhFJIMOJM/rVbPXaIT1o2UYI0CQ1PDogkjrg6/A2/IqO3A8ZOWv5/87tlQ3NU96pVo/M8TC7DvNA2gtEd7KREAvSIbtyi6zjF2DXNdTZxfNBU2eN978SvYXFkD48u6wynt6cxPcyKRtGDc/g3UZ9FQEAIAbCa39nOWVYPBQ3umyyWvRWRayQIBP92eBi0MNn9/8DPcR+gk01OfSNB1h1YWDiiaZO3tSPXg5q5FwExRvh9TjN8yhu497ZpIk4pF2G5vy7SWvUu+3LRPPruQEGeu62xDrui6Pi+fpKjzzFsp6nR0l0Wk+tvkRODW806DY/XN+HweeBP9GMukS+jufR5uj0K0FIRCPoUgJACBY7HE81Ca2pCmP/bIX3bhIUW5pM3Eb97dAPuONUCeI3qi+2og6/aHDyltG2zgE290WM20sxP2phnUuwP0O6XQdzqO1DVh37ltfGWOJ93Ff0hPSTfHvPBiUkjOkAq8IfDYnrk0CG72HE9tfuMMYyg3mjHIdg4NvvYhjvEskJTIEtia57YsLT6j1IQLP/cyStlZFL92CE1f0JqAW11/sB/oghY4Asg/MKbdmACel/l1ofGPRTvhcC1+LcVBRj/Zt9Pv3zUqWZfdSxumf1iFP1T+aJlKNOdj00p0CrbobjOoffJRpGNZ0xyBM4Bu2TH6yOmTz1SWrSDRvs+jEc3VCsvrHiOHoG7IslawDGmSGyiauoz6opjBYjbX5QVPVyZ98tGlVWD5CBHCExQwWAphgGVRNIPkE4G0ueMWTlAy/2QduCLw+V5asgsAvDF/0eBbTJL/W/0PiYPvLvkk1OfwKeZjn2/YfwI2Ykx0zVDhoIhDb4SQ71s6XitRFSgReEOhjjXm1vF69wPyriP16YOKw2+pDJcG4otsEaA2gRNUnpEArVVk+o14T3uwFTJ+qHzy+eEmQ43o47Kh0id+sO4AdVrZZJvH/B1jzoBBGN+6uJlB0tw10Bij01r0Cs/muq7v8bwFBZp5xXK+IPnUo6HDaQVl0uSzFWZP6ZVHRH2YgERzXYH7ctGKAyTaEun1DGOui3luLpNHL0an0OQLEaKlIBTySQQr8xWiyWcDiZfJNOExvt3LDRbOFI20efByeBDdHHa/WaLRInwPBbjWXvHXRfDNpz53TRfkduBQTSO8t2Y/U54r9M9Sv9QdH7zmurom0CefAvs8M3/+cN0BOHvaHKp8RvsZ+b1+Cjm6ro8NpdiQ9ZMcy2UdomJqPTbxBgFMEDlaePYPZtpb22EO228+NQNLe+M0fFDg0eA2a1eR6sKtRZYIrZjs1zyzxFQQG31u8HN7l3EtQ9fONP0cR/jkw4FkbksDuyYdcW9s1zzmmBtmvkTK36Pys2R6JD+n7+AowVyXFdTWFpQuI1ybGkM63hqBHigarz2nNzFPQteV2PuFCOEHQiGfQuDV5DPDYTpDZGYMJjm430imr6cXMJEbFVl8WdZmCuU3S+itm/m3DztCGWaNDkGu8BrwoBVyB6nxNenFr+Cu11YFVj8r7E26eu9xZxqdYx6kA28INNcVVpJ36ACwZPth+vRpUxf58yfodjo9AB+dBmgPZzzsl9VHLr0pPIo+NnM/L6DhlzQRYXlhPuDLDK6jijw6c/lJSEPRsbgULOs+y+E5Jyr3+NHYnBEcEYV8mOdmdzG45tt4wN0SQAR02//4dFjTGaryqc0YKbQcWS6h7Zp8tDBy2TX5mmN4CaMj8IZXTT7C2GpXt9tT2QZyiea6wXIit/qxihdpwbKHuhFVf3/U6cQ8SSGfiwseZbh7iBDeEAr5FALvQmcxnyUsYG6+9LwiL+rcFKU3Dxzl2ddOmsAbzFF8zb8JBS/ceojaobSzDnTBnk0wWoAmn72/ghCoubVjkLd++080UKctbGN1sSrSfx0vXlu2F1bsPmp5poPOvomSYK6bLjrAaWT0kK7rTHRkTF1s5dFH3qCuK2hlKPMhatLI4uAIAT7XC6T2+3zrIdh+qA4ArIdPVNm8fpGSZbsksIHJd5M9L1X5dBWYBTX0GiiZ31HL/kccpPmA5KAy45MPn1ujKNe+DvLt2ehz5SK0k9x8m7H0ojliK1mTD/28jUnIh6LL/kSmYMDrXs9N4KZj1hMcaPqZxTLJrsnHCnv/2KP1mnvLGXiDvT56bWExfIIUeIN0/mMFTplDSIGOV8mXEU1DzkEq/+qICpABfUzJEjr9+hP03idECK8IhXwSwbroezG/NJiRfV3VNDyjolmD3dSzza+nX3cW8jkv7IyYyqE3Rbk8tP139X74xX/WceS01+2tZWLxRHoh9zNiIoAcYYgfmkhucGtHlXy3kdA2LwqbH54QKA2oTeaBEycdz5j39OnAG+J98gUpKtc4hRAZ8ypbeZx0kNYe2U77T9Q3w5/mfA3xhI487Jn9EV3jYorDClefsh5GB03eW/+xLJNekCYfCu+sqmBK78lcV+Bw+cunW5nzmNs9kho7QUbSpmlKUdThD/n+cDkWIR8qmMDT89n7G4dGkzYXj980syZfM4vtKQGyL2a9utmm9oVHsQyz7E+beTX5Uh9sb9eeHXCR6jXHWsezvzOXQBIS2pUdeEH0yUeolzoDZRludaHGN+4M6nUq1DXG4OuqWuZ8emiuG6IVIYyuqxDcbzHxwDFM0qaL9wBj9R2U+cu4LbVqx3kQXNr/pjBLkYltB9kWFNFkHq1rgm88PAd++a2BcPvoMyTU4D+cmnz+06Cyb0NWU5L83KgkSsSCXZMvZa4r0CcfrmWD0JBldlGY/pUSYhi++TjnD+kMSxsJlRe/eW8D/GdVBXRomwsfb6hyvKfxjSULRH9D1I2hIQsizgFk+lRpLm2AImvH4TpyJlvZXrqZpo8sGpGCBxVOk48FU95cI4gaNRDR3YVUItZBFvNolCbf+gpx5q/moBhkn2bo7zZbpjTHE1KF8G4wiqadK45gE5hsN/3tC8jPjZrei/PJxzKecGcf1zZP/W//3g5tc7HpeTSPsRVD5iIBCUELVk40Ak1x70FemKixCO+okiHe4d8aXc7im9Nc3yZOU/lEwmbhhiCRt+wQIVRDqMknEW43HHbgZHxzNlbBz9/Ga5HxqliL1gSzaKWkb9gwaelC8FkQSWveELROvPh3cHm/ruIEHKppZC9XkA78/uNJ081ZKysc5fohlpBRhwoXaiTh+nt3jvKREidUaB8W0Ozr+XzyiTfXLelRlPy/e6GwMr2ApU0yPvmS/7NqKtmrolmfZGlDGW4QcJGuzYeAIDWySBDBG13NdXXr/6h3QcBp5snmk4+6HspWpjelw+PNFfu48gUB8+UnroU0Dd92uw7XwcR/LoO6Ru9CBJb2lq0h30TQ5DOPWewe1ZTHywW8CHh27YJ5vmT7EZi3+SCHTz6xQj6UpqSX8UEUPAnQ5DODfJGQOrN47L+cCJ6rirwkscwLAauaruP3sMb48Ds4kiXwBqZq47IxW6x4QoTAIRTy+QQadonaSGgawA9nLIfXlu0h5sVFniUxKZbFwSqwNP3GpXd9T3OotP1tmCSLsZww0UKPXUfoNCJE1wvg7OOA950tBqTN6Fl9OvhIiRMtdY/BPHbT5rriGuSyQT3g43vGwOWDewkr0w9874LiNA9wmrvztQ/ZXNcf4Ma6+aKfR4tNFpjqopjIuDU2/QyzYuy2rUleDn1iAm9QpJE4qqzmutKqUQ6kriMF3vjDh5tg4deHoOK41f9rh+V/hV35NzHRoFI0Y4u5rvmiwEYido9qarPSHu1d65PJeu6auVJi6ZCe+PQ++dzTMAXe4NzM4qqwPzf7pBTB46zRdeWPeZy5bnM8kb70B6C8ZBU9Ul0qxTVPZg/DVy1vs7P45AsRItsRmusqBKSQD9g2DySnskEcgDyZ69oYsbHQeY5C7EGzzuvS4MkMyla73xGgZJgxUu/tJH6q2sLSlroZYWx0jsAbTbEEvLRkJ/EAMSClzRcUrCyOrk3O7FYIu1Lml/zm7ta6iHMg4CHIGq1VJLxozbGa97kWm0pg/9adlKa4qoJ6CaBMaB7LrKZgWQ9sG7HfjHb8YjoAAGiJJuo8spubZQvShBHy8ZRJ41NMppuH/Sm/tm41vLh4F1RVO33gutGW0eSj60CSjzgDLG4MuYMOYlqEFIJGxH7PymPc03uNyB7FVPLCwh1wtI5+frrB9aKJddtGeJdIr2fyebTlyKfrreryJ0TrRijkk4R/LNoJW6tqmPKgnI5rGtqfjxlWzTq76Qz+ANRECDNPrA/7h5kel9sdipOjfR9h+L6Q5SSeymSNYkGqPHESLnp8PrTPz02Vy00SEW4LcjbA7bbXjw2AV1McmU3v7fPVOOCihBnsmnzxVF76b/rb5zvgsY+3MFYEgUwmpg1/JKOXIEpzJsjAG25g0eQTDcd6ylE/bR6ruS7iUIjJlxwPums6GuAsAkRDZj+aaVdBs8xPH59Y4QYHDXpuAWjNdZATo4/w7tbeb/74fLj+uaXMtPDAvL8lBi7AtI2MXuNxycCCP87eDAAAndvlWZ67rbdGXbSzhcb3rwiffLxwHKUEm6GaQTLXFWV5kIsZv1j3QZyfyLOOkNLpuo7dP2TMdSkrooTbfiVprhv8uhAihB8IhXyS8PD7G5nzoG6+WFmRU5MPX8KUt9Yylm7UoSP/tjrTTv2f+ptHWGPPYyymCcKGgLkaxvQ0mgFzNlXByeYEnGxm99/HAr8DRsiojdSXfoGXBj/2CS1xK6LrOvHgO+L0zohMKU0+hsAbtY0xZtr8Bk9Qi4gGcOWQXvDZlkPQ3/An6HGgkOqWPc7dyjevA6IFjqJYqAhBDm8RTn9jnklhgr06mgMUTy/yaFC2hsNcOko40Ts++4WunlsA0FwHOfF6Zlpw6NWxLTMdAHzCEvP+iBQcgTSsirsUWKLskhD8ToYfhnCHdr60pRHyYYNp6I5xwruXRWVbu+84bD2IV7AQseU0zzXimBfEfnIwmny0pueWNNR8lC6hWzJc8xj9ENE09KUW9r4ifaqkog9Vb8tfFUKESCIU8ikE1EKXXHTpVyV7Stl7XHN9lgOr8T+OUXP45IumNfmoyXMHY1k0zUlKY9W+Y6vcHl04W7X3zFBAxucw/z6rdwdYV3EiIGqsyLYzKp1PGPy0e/X2c2HoaZ0QmYxDKn2DeNDp5c7JX6NOPZ81TYNrv9EHrj67t+Pwyh9dl6TJ5w9w9QSqySdqKCAKIj1Bm0qhibGPAb/dONjB2kei1zGLTz6LEF1sRaqsv8Yn6kA6GLOXm8gpgCgARJk0+djrkQXzJxO1rSjaTJW+pp3azq+ltwaiQUEejSYffXm8zYvKd+XTi50PEUEBHWVx0ktz8S8i8IYfwPlb59UTd9PyA8AL4GWtY6EmX4jWhFDIpxCQBy2PG2YZ5o7urFeMarzT31TKJ1+AkiG+qIBi6zba1PeNp4T6/NZGpKEhP1cdhx1Bm0rKgK7r2ANAl8I8tClQqo9EBt5QAUb/6jo9rzQ2qKjNMS+/J5rr+rQhxgfe0FzTyAL5kIJ+/sLC7dCjQ1tmdukmVMDNmZyIJlAj0XsZ8qLrUqYzJUwfwD2Mm3Z5UahrskWe9dEHpAhoBJ98WMf4uQUAABCN0WvyuR2ecW+lzGtTWx6uxVtV4PiuDkneR7sGq2CVgAOrf1A32NdoVH4W39lmwdspRW3YiKGrAPUTALxrFUc0gA8njyaOMa/IiaKpZN0fugauYiotlcfN5Q4mnVt0XXPy0h5FsLnSqp3JyzN0Pfsuz0OE4EUo5JMAFNOjWe9wgTdc6yMI1SwmtBRl0dRheW5WWzfVZjz2EgkXF3jDq2DI/i0sN/w0+zjSAuJJ6GkSCDjLkr/BlHGz5rYRzKmrgt/mvAha4hzhdRuwj1EafzN+Ids2IzRjRNf5tWp8EfIF4ZOPIS3J0Te2dQ5vBWjCB2igNuGRyGdwNFgiYyog5HVrgT98mPTTsSQKAAAgAElEQVSL9fDVgwCA/hBpXltQdWT8ZllLJJkiuqExFofGWCLtP5YHfNNFXj+aHdGL0NgY3KcjLN1xxHM5JPCSOfPLPbDg60PJP3RCX/CY60aTwpZonF54QePQ/p3/uwAZHEI0NuzPaOMTg0AQxi+5W6wZmwUL+UTyWreSeGvSIAHfjCwD0MudZbII+Uy/J11QTF0GO+9Bm4Z6wSPXDoayXrjoy6k9u8c6cJp8WAE99rxGrodHe5XmIsxO5uz1lbB81zEAwGv/iuwlq4aiVZMvaM33ECFkIhTyKQSUkI91k4ryySf8YETBE0Uszjhz3Qb7rbo5j2SpSJxDainL8bbfsggvAlsc3Nrm1MUPwqCcebCmcjFA2fXiCQCnoFEpIZ+HvKpovdl7WAf82FVBiOMnDHaV0OnNdVFrgmurPT3Mmh5zq45C2udqQF2D0uQ7o2s7X+p28ifORkA0Hs63LSuimmaZYyzF3PL3L+GrXcdg1/RvAYBZW5wfzONE8Dpm9jXsVch3x4VnwO7DSU22a7/Rm6sMmev0L/6zLlMPqSE5hHw8+HBdJfG9pgGcfWpH6XRsqayBXUcyGoiNMfyeESsXZey4Zs5AdjLg8NXmKtwxNOXpvtlIdUN0PjyS+w9Yvqc9wLlTLGmYLG5MSeXsAfT0vyL2seYvO+8MhA/hdDox35JDGQ422Y8somlSOj5lFQcQPPjHr6xI//bbdDaRyNTZunabIVoj1LFLa0Hg3dShtJq88j8Z/BMXgcnsM07EvtZOu8GYL3r8MwGlO0GnbSmuDq9t1BICb7i1p5bAb86F0RB3Cvlm/e/5MPOH50qv2w1ehNZaADeUNEOyuqGZ4/bUz2/xry6zKQttraQxIcMnn19WaDjazc81APj8p2Phv3de4AtNRC0Fbk9F7nUhrQEwtdh9QrHQ8lVKm0IkqPzW8pjrcqx3eA0bdlw8sLuwsmRBhMDqtC4F8NHdowmlobFun7sfW78uceyago02ARyt71hIC73dMzR73RxSgFoIZ0smWlvJGE/dtST/yG887EhjXze+M6wPvjwTfSy8gZ0niNfb4t2jMWnvm8x1dV2H2sYYVJ9sZq7T3VyX/YDiGnjDJT+zD1eXct3KS+h61lnIhAjBi1CTTwJ4FxGUTw8qc12SuYGPzMxiGqwb//MvqfYbHpKZGooGGYhRXAOSNrI8C3O6XJuWBa/2Bi9kaCS6CyoNfX95Pfvf1RWWv/NzIjD0NPztrAGtuR5O1w4AwGBJlLVMfPu5pXD+GV2Q71rr5ksnmdrZQLLOpD1EO/3jENL6JPRsjqPrsWvyndq5AFsGK61uqStPeDArdOnQX76z3vK3Gy+kMWemqFYCrBXSHHr9mOa/v2YQ3DC8r6cyNNCyiidh12gmIV87GNizPbB6GGtoln8hxwuzJp+9O4n7Z3AGPMOhiULI99Wuo65pggArz3CaYTonifk8U9qjKB1ZGeUTjZdn8WTzejneGIvDmr3H2TJ5rDM3ErG02TemzYGmeAJuH3W6p3LtwAbeIO4PAKIQh7Y2joHjRU02gTtWk0/SOpbQW0fU9RAhAEJNPqUQE2Kuy77hZgWN34Z0cIg0Hez12LPQtAXvrVDyt/uqQqPqjwtrv/doPdw5cxU1bY5yjfIM0wqfD3M/enmFeyJGuAoO0+/lLcqVtlv/vBw6ttj5wztgfpv7QUvwC27dQBrPk8f1I+ZVxVx38mvOMU9zIJKN7u3zkc81H+eVwZ+PNzTD2n10BwdSpEh+Z9Quu3gf8J9VFcjnVhkWzgG5HNz89y+x72j5L65Plmy3+XlzWVdxfRSNaEpFjqUZgzz7ElaShp7WiSryJQnZdBYkC6sIgTcEzR6aMei9PfkG5ojT0ZdKyRKxm1gm4C4pzLjvjTXU5Ymcg67bLO7LkdReFNGxZmGarpPHmZk+6VPOI6t8Zt42htSCzHVtgTdw+yeaT/OyVuCyPpb7PKzPvx35zj407IoOEQ2z3lm0OxEuSnjWEF2Hz7ceyiq+HiKEF4SafBLAYmrjClahla0aq3adh9UNcwBB+m0g0MMCOxM3b9h1XedXk/ewoaDR5MNh9xH6KHUoODduvqtsCIcCsh7o0i7P8jftwTB/7+cAABCRaFJs7/P83Ag0x3VY8+tLobCNeuybdkSu2I02E8S2vOCx/twtQwm+F/2bV8b33vz3Lx033Ng8JCEfJx1En3ycZYqCJfCGz5vzQzWCIiZSjF/WCycDTnYVdI+xQZamqAjhFbKEqg0wO3of/Df2L8/liwTRTQrDvsW0y/JGEANECRpfW7YHOhXkQkFeZm1cNfUS6GRb480gW8LQ0+WLua4kjbeM5Y04Osw6C25tYy6OzVyXLp3ZdYlXTb49R+n38cZFq9eZxBx4A1uhi78+m2CWBrquw7XRRY5MuPx21yB+aNUZa8y7a/ZDYywBG/ZXS68zRAgVEGryKQ4aBqhjfgMA7D/eIPxg9OaKvcjnlnqMTQNmeeNZ9Gg1rFjBQgvNBoG2ub1u2LLrKIeG334FUbBHxaPfdKR1K4XSY63BSktCBxg74BQlBXx+QJR2YkGeIsFVUp9DK+ADQJvrsvN465gl+eSTPUXdac8kyKPx2SAZ6QMx4pkXuJrrutCjCqh88vEUzPidOI161jJyUmMuLahY8Cicoe2H008s4ygxKMgfJDQ1yD7O//ztdfDjV1ZaaCEJ+AAI88qsr4bUrLX+jQu8EcxlrN2En4+34NPrqVrwOc38zM2U25yWReCboYMeXnvjUK2gix8G5GDWPXZFB/qv57l80fRMP2dscOx7WGu57IJKo1x27D8uP6p3iBAqIfgdcwuEyCWd2fzUVvnKPYy+Iyjw8YYqbH3p5y7vaWD/dvMBDx+dkw2sB7UYhUmGow5hA8LqlE9kEI+g4HawNTaRMr+P92YxbaIicROP0t5U2dTA64FG5LeRyiIL+dSeTcTxyW2uS3gXcHuYhZptcoPdssice25uMGg9F/gtU3BYD0hwq8ED87jxUl1BSuO3vil1gE0T711zq3fKT5kI6EDoe4bouunPkzGQFFy7cGtW0sSUHiq4oOAF67rN6pPPTchnsa4xFSV+BOrYPSftOrd2r3uAmQxSmnymoodwRJfGafLxgPSVrmsQJndCT9JnFvIZsPN6r4G8vLAlke0YIkQ2IBTySYDIvRG70Mr/A1nmezMLGkrbgRUOIZ9Jky+oY6cXzTOvfeNcLNUWRtCANvAGyueLKLywcIflb+p9QIommVFs7aSwHjxCoNE2QE2+V77YDZ9tOQgAfH1JMifnNXs7q08H7Lug2Yz5e9vkeO+3PR7dJtDC0myMGvko0PJ7Ef3lad1mrYtQ2aDe7aFrYZtkOmaqRJjramlekRbygXGg9dbQ635zKXx6/4WWZ177DqtlxFAwas3hyZctIH2eVeBEbggan3wsQJXGbf3hko+3XI2wPzMLcxqanMKfD9YegM2VhtmkkwCqLR8H3c5LCbb8KN/pLDiHR8gXRRPJPlddzHUxv4l5dICYIUpIxEzPrSUYf6ECTLrRggKu30j9idw7Zf8xKkQILEIhn0/gXkRpDgemwolmV3wkuNSd+Y3aEGE1/SiIsR9Y25iFfJgC6NrLvW4caC5r/dK0CvrwLQJ2V0GObwrgI6mdtWup8ShVk89p6qC0Jl+WlNwuj2TuLHfM/fKd9TDpxa+48xMV+TjHxoAeRdh3dgfqfqNbUZv07zYCXDaMeWy+5zIAxJvhWfwhseQTSoV30IxBWm3paCQCnQpyOSnx3jKaltH6bWiKZR4KKL8oP9fkF9Q7U9d1HabaIjan3xF88mEPy+n/xRzKk2X5s3gxzU0Pe1QzcH7nVNin0QrI6QU7yZQZSwuUkC9TWiPClPn/Zq6ECU9+niqPsmI7HTx5TJWV92rPnJ8pmA/BtQbLGM2J+HNUx5NEpjWREiVEUJp8YDsbcnQaqsV5eIm971TeS4cIIQKhkE8CRGrTsfIgr7dMopBh5MZmgB12Bix7oaNZfP5v5kr47XsbiGmciw/WhoaKLlyulhB4w0sgE1mI0Kvypf6V9w12ShK6fwelYGD9tt5wCGDHZ+m/hfnka6OGTz6e4EEoAYnXMUE21w0WHQsyPrVECPnMEMZDRWjPuWnbUOb1W5vfXhvVkGY5J3N6RRBhPakBQLuU/9O6tLluJPWOroIDJ4L3A0US1PGYW3PTocjSZdVYwptvGnyVZujJtqzYcaiWO68X3kKX3tmxZt762yvLiXWZn0kZfzryJxd4yDO3BU+gCVYzU9q1wpnPfKHnTIgz4Y2nRAmaWZOPog4SZJxv6Pf2IUK0DIRCvhYGkiafF/CW6sknn+1vS3RdRPqTzXH4jYsAjgTaxefFxbu46/CCjKsc4/Y0g2yV91VVJ82LurdPauuocAigl/EZ7FNi4xNuglsD5raZAjDjKuHldi4gOGNXfDKRDgnUQ4NF0SXg9jB/k0wBBCtoW0UDjWpM8RywVASrwJn4WbrOLcAWtRe6tKw7AABMKO+RfGAI+Sg75Oa/fyGEDje4tSMAQEJiiIKsGZ82HkKiW9PwPMdvo4NxTyzgzutGG6uA0umTz4m4qczbRhYT9y0WiyAOOljS05qL4sAipENdSvJwM5y5rr0wkRc8tG2r6wBxSF6aWgJvGGOEwiefm+AX9YInCnPoky9Ea0Mo5JOAIDc7IqMkzt1Y5Z4IVY9pqfHSFPbF1E3l+43lezkWfT14VRVKGJvNjCZfcLSIwoETJ6Fzuzz43wvPBACALoVtbCmMj/RvcY4yRtf16puJBKTWFiV5orTeWOC1Kezf1lZrShXMURbh+3HR6vwGTw+RFJplCMFkDW8ZvvH84Ym0mgiMpbpom2RM5FAUZXK8sXwfW8WEurjgcQjG4gnYdywzNv4/e98db0lR5X/6pckJZhgyQ5ScBEQETGACFXVds7Lqsi77c027hhV3zYtp9yfmjFlWf6CCgkiSjMAOOQ4wDDPDMMPk9NK99fvj3u5b4VTVOVXVfe979Pfzgbmvu+qc09XVFU6dENqlk1hWZRnsv3AWLD33NClgPi8Wa+pYbWGwu1VWCRv31MMWp8VjQspQeIYfkCOKfmLd0MzS1O/ezK7rjslHpQfAVN7Qnwz5FQaWuy7CM6Svy15MqkI07YdD7fPjjaaiLB0v3HXHjbJYyBmtRJQsHIS8uxo1JjJ6Y6dTwwrKhCCPhSkt+T59yX2kclga+8LazJ7uzU9Ye3bfAE199svuXYVLlKrp9AWWSEPfdN3shc1DHLaMjMPsqQNw5vP2hqXnngYzp6ix0opFZIUWPGRFSaLYTFygC7tt6wAaY1q5id8/dKR4ohP3m++8X+UyMKRbJ7HkY6CMYeaye1bByV+6Gv5MPEg6+wX7wq5zplrvVzU8pAg7YYMyfTldqlrjZvF3j33msa/ic3+8H078wtWwZvNIUCD4HKHzo7xmQZ+lS+O+D043PE523aJSlDhJ0UOiWNFNi+cdZzgs0wkIPZAovgTkQ+F8f6GWfGwIoYyzqeLDWdkhpYPcdfuzZAo9l2LU5x2UX9rv45fCW39wS3Gt464rWfLZ3OATfSc2Ms7D3VrJV+MZhlrJN0mQD13uGGflLkJk5UgKSz7DXVehT6P88FObIyQoF7FznboB4hPbNjoONyx5Ok6ISAyPNaTg4whyBWmVlnzEhUAuU8bYQHGBbzKRa1/cG+DCvy9NDjriOnUVb/mzZxxaAZfy4FTyka08tb8dg1EZs8a9KzcCAMD9T24ibV4+/LID4caPvbgEScKROg6eb/Mj3/7cH+6P5pcKQTHdlI22SuDah9YAAMDG7aN0eghC4xPL1VDWWa8um0ta3yVUXvWSu30OyjG0EP5msFsElq/887WrT4bNw2PO+wY94xdiycf4/lRLtwy/gdULaNrY9xHdh4Ms+fBKNlGCn5FR78ZH1hZVOu66piWfjpBhGXvOkDm3tuSr8UxDr65WJjRcJyCl8Wz/iwWbLtw8mUKQ1uq22EEJrNf0yVRRIiJ0MXFP/e9r4a7lG0j8SjLkS6LwVOgVbRtH8V9+fSe85fu3wIoN2xNIFQavkq8LIC8Diojw5Sn58CQLFtx7kfJnt12z0iLlRrM6Xj6EnM5j69Qy987djsk3KUDJ+m753bnWsZiXLfkAesuaL1VXzBUrmfQ3B0bYBWJ92QIJ/z7LP9xJDdeGuEq9W1WsFi+jrfkA7OObEK01J1XmKhLe2GT1vUOfZOu3MZV8BsFId13kuSgtX9QKiM9Gva6DFekD2XuFzPnU8CK+Zwh1hZbp/vHuJ42yhSUfEpNPB9XC01cqZM6jKvlmTRmAPXeYzmdQo0aPoVby9Tx4p+INhyUfd0xU3W+Z5RMseAxLvsBTGDm7nXPDmmiNVtZpdaFTyuPrRMp738pNAAAwMmamva8Kw2MNmNYlJZ+tO5EzcBXvuURLvgmmp4uPyWc9ms5LMGgFCtFLGhMErgU/Z0sqY/OI/QS+t1sjDnHP5rO841HnWPKpUvj5xLrzcRBt6OKIxUvB4bvPUf7FaLrg3YS2SZUZizU52hoXysHPRJtzMJx35cPksq636GoK/fWn7g4cct4lC5EY1/bOFRLE9h35xrmg/sdRKGqFufxC3G3V+m05GDKncjOVDTDQ+3pZpP5Pblqq1RHQFLmSD4vJh8jABGY1zyHjSryB0Zk7YxCO2Wseg0ONGr2JWslXAlKe6JHDg7X/dbmolLEmtbo66NZmjoxmh+02BzDoz+6b52wL+aEBPGgtQEv+Kk5gW7wj3QRAPRWMlTvvK4NdTEIwPNaEqYMu/nzlDhXxyRfyhU21Fh1GP59Im81QBLTx+m2j6PXUwapjEBaE21Ep8NEu/N8VVj7dtuTzttHwRjhuyVdhAPyuQt0AWe0qb2Qigu6jtCPqemlrstIscEyHP1s5/XDLh7nTh+DIPeYGH7YpVjcOd91ei3lK+0z9bdI5TOVtpinvJ1qBWMJYZCfJVdKnoBIG3zeXeo1rxuRDlCcaS6rClBXzTmOyfqttzu+Uiwlb3mgKxWiACpllUEw+m7sum5IbvuRPNowjMfly6PsWcqxG4Z4jQtYl/VrWMtt302xCtcGZa9QoCbWSrwR0c08Umx4+BXxuR1ToA7BsYYW661oG5SFJmaNXS219iCHV4s+l2wnpc+PtrH9ky7USsH2sAdOGXDH5cCVxCgwy45zYCla92TPEsyjAuuGue0871hoFX33jkXDmCYtohQXf2vRnNy9Dr3vfb48rTV1K+ZSfyYsP2gkAwhf+leHKz8CRy34Mr+m/vhJ2Qd2DUMk3b7oTK7AlmjDgKuWFEFHfgVfhUCj5+AcPt/ybO65kzBokRRd42/F7wWfPOEy5VsX8VrX1oPq92CzOZLl4329ZsPHQ288IGZPayrBQ8tkPYTlJAPN+P3f6IJx++K7Bcr3u2zf6eUW0xS9ueZxZAwm3EtDXbV5MXFp+t25JsUZ1qxUATTAt+WwuyjYLQVMWn6x85M143KIdvGV76TC4Ro1Q1Eq+SQB5YHZa8jGHRcpJuByQWImXK9R/Q6CzN2Ls6OUt12VLPtcpUqqFUNlTQ+dELI5OnqSlm5Y6o+NNRQlrom1NgKVvi4TNko980poHYK86Jh9xBd8NS5MPXHAnueyz95oHf3/yPso1a8s3W4vHyRVnMAwD/UifKIFP3veKbVyX/Pi8i+3GCAAA9JfoNl/Igsxx+u9Q+OjZsxVSaKexSqIgVTcRoCpaOAYgMS51Skw+XMvHE0gCN4B+OuRWeW5Gn371IbBg1pTSpOiFjTNHAZZlcpgUN+zuqWTRghHrQpqD/I1prYH1q5Dsur89+3kwjxFaQOfw6JqthDrhL2StxVLQzsu0hg/pD9y5N/QJfbLZ2i6PydcnK/lyl25NdL1fRMV+R+Cil9c55/SDvPwmQ9iCGjVqJV8J4J7Cp4Q+gL65/0rY+9GfB8lAGePkQR9LihWjQNInNnkhEzpRY+66ExWxyrlcIexMyFwymkK4s4WWyHsQUZbweOaWfOkasNkUMC5lz8GahhZVpPfR38fY8jWrc8XsNTc8HYN9rph86XDyAQsAAGCfBTMAbj8fLl1/Okxt+jdRNVrg9iLvprg8o+akoG3a8N8AZh/mPm5TmCqHaY3NMENsIdX3vrcIC+5Uyhg2HH1LFkldc5UwDgY+fpmHO7anZD99SdZySWilI9WiZ1jymcgPob/5lqPp9Liv2W5I6OUVAranVMmfe4jSnPoEnCfNLfmwxtUlpDah/z2ZBd7x3L2cNfI51jcOC6i9dWtMDgx0W4AaaZG7YOb4/OAPAO4EAPgFbK8owYIQwh9EnEBHH2TlvS1K3jJwy24DLuVgqoWQmdSvRfny+55KSleeLEMWLo12X+GcuKZGy1KjO9PpgEVZwt6IJWy+d//kNrjqgdXOMtSYfFVbvXGVzn1ZBk1q4yXURHczBqWOkL6PWfLF0LPhjcfuAS89ZGfYYcYQwG++AQAAc8efTkJ7MriXUh+B+k6E8huxXJB/aw1YhmsTFTrtVNZaspcAt54+hn9xySvbv/zhBFRLPuyUJXz8KDcLtvNueYwnOJyWfJDZLWgNa7by2zh03Ezurtv+1x1nT8DOs6fCKw7bhUEvQ69HQ+SWrP69iQuNBBbRnH7y6iN3hX9+8f52+lzvLM99dY7Brxt1hECVvbamomfXxa0fbV5id3/yJTBr6iA89NRmKy2yglP0/mFajRoU9M5uZxIhpfujLxi3XsYVB+Off7k4XA6Cq5ASd0H7NwRm4o2wUVdR8jkESvXebGL+4hY1RhiXnVk+Tt6xtuKku0o+n1l8cWScHLFxTkRh0ZFOAaUr+CaSJR/bUjhjLOoDYvLZ4DCEmxBwKSnJY1i73JMbt8M9K3DFR5ZlLQVf2Uj4bXO/hMqGPlLoC8/BWBfdAbsKzW3ch6YQUX3KH32hPe4HhGmwKXz7x7bAc7L72fRkUA4wWV2FvUDxF+nFjbN9bUtbN/rKVKH848RaSwHTIwaPyUcN95y3tWpVSqgX8FzK4TizbqNir5dZUwdg3wUzlWuyzLo8RGNwK751zSOw6KN/IJam8QzNE6cqGd2HXgD29TxGk/K99EJogRo1YlFb8k0yuE6alq/fziMWs1Bui7FttBFMSlfqyYM4ashnoeNT8k20TVK+IIqVO28Xl5LvwJ1nwQOrzJOxFHjvLxfDyo3D7r5RnMCmn3DtSj4qr/DNHhXoQsMIKtRFf2sJ3O7Yl2Uwa5o6BVkVHUVMvnjYLDgLGSpUmob0aszNPNSC77n/eRWrfCpX5l7c7HNBtXQIoodagKSh3WuwfW9ZBkFuUyF1ZHhj8kW469r6/eG3fAgumHINPLD9NQAwh03XBxEVwLeHO08C2JVz+UEUkU4qgQLgVVqQheNZWfncdbF5CasR2nbsg0UhnO/bhwbboyDdRIe945A4kCFKts5FvCxqyWel27lz1J5zefwd9ylKPij6rcddV0z8w+AaNQBqS75SUNZkTxmcORmtfLANg7OmdjbmAvxyLVu7zbhGsThxWfJxrO5kxacz8QaZIg8UUW95dC2dXvvf2FfdUfLZy7zrxL3jmDhw8Z0rAYBqoZleK6CzfXs7ngc52XBmj0OSCrglH+1otOrYciHuutOG+uGKDz4fZk1pjSlWCs10lny+xWCV7Rai7LIljKlRDYI2SYRKNvck7Jq+eU6dPImj6A51qaWWKR6V3PBxse9kLvhmMJx2o4E/w6wNDwIAQF87iUwIaM1TfWy7ariXAyVCodfSlnfdj/Bv2gwZkxgGQUSZJ4SiKHF+kkQLKxdGxmlrBCPxA6NXsi352qSpfWBkvKG4m/pkmzNtEAAATiO4REviRMG04pTpy3MYrlCT9xtTB/qDeMr0c/gOb2U6agKt1sUN20aL3y0ZJ9poVaOGiXq3UALwAT34rIpVOqWSz4Z50y0uXMqY2JFjy4g7aL71JF/726d8sS0QmkpMPo13Cc0VYub9hu/eTC5bxKSQJ9Qgq4KWnC7FZxXx8rpl1WNjSxanSLtXoiUfJbtuj1hacKXIs2Xvt9NMa0bHZp5VOWHiDVuGy4mCFPL3Ro+hoRes/vIkJGW68HBcrRRXQsLbrDK7biz0DRi3xZveEBD++k60D3eygBAC+abcINkO+RBjsT7m0EC4Di5S9OlGU8AoQQNiW0900zXObnHEo1NN2BPbWrna9sulcPWrhhDFHO8DNSGCIYfEnrr3iXlLKd6xi8Q5F90DL/nva4u/qc1x4M6zWrQTzuxUSj4LdFd23Y68ZsW83sLZU0jWn/myiKBLLmjn/z701GY48tN/hgtufaIo2Qvrjho1YlEr+SpCivmfshipenGuxuTDrwe7l7HddXGa37zmEUku7RRP2VD4RaKAHNMtckKOlTdvTtfiqIp5zt0XyrSSi3TXLTZ7ZbrrEq7JHWHLGvlGeoEcYLvOILOPTqORT1FtS74Ubtt+t47etuTrZuKQ1PPLkxuH4cYlaZJ5lImj9jDdilK7dauJlEzaLkVd6h4bs8Fht4tPp8aMyScgzpIP34QqArXuMXl86NQDoM829hQmJuHftkvJR3kn82fqh7f09/gPP70d/u5Ht5LL9xKssS4BnPEgXRZNPQfiwG0rpndbo82Qb6HJsKjVlS9U0I2lTQuzEKQwonBRuHXpOhat3IDBOq6g/MP7gq0mxt0wqIDcSk63pHTzP2DhLO1QC5fPtW4vDCPyYVbj+sjqVub1qx9cXZSrdXw1JgNqJV8ZIJxmBJFluvzEQh40fUFQTTk6wDalJHdd7e8+izxFeUsb3/74+uK3a44ua5GWmq5tsuQib08XGVe/XbJ6M4wniERMS7xRgruu9nfennRW4bGZYuC05PvJq6sURZOC1w7KBsBmhZtPUQkTb3gt4Xo5gBm4lZRkyXvkGX/512WwfttYt8WIRhA8anUAACAASURBVAp3WWpG+hCrnTLfdsiag1Ojaks+xToIc68tDnV4y2fK243pRqMWV2CZLqaYLOKrWRqNItMV9z/lLwQpNs5pejK1z2bA+N66OKb6+nusbkqfc3RLPuzwjfMd2pQvHIRYnnGRh/4hJxSBXEksKxnp/Hxs8vdahtWZKrPrgAnPritDkY/4/MLyG5OPis5Yp17P+3e+nRFImRo1JiJqJV9F6JE9VQnALA7U5x1ALPkozaFPpJzTKit6+D00I1ZiIf0rn8Tc7rr49WVrt8Ep/3UtfOGyB/iMNThfa745KeNcTSOZn9LSAvhCcndd7P1jojhj8q1dkkSWEHD7IN7MKpFm/qwVxuSrEtxNzcdfcRB6/TNnHAI7zBiC6YO0GDd8dK/NyuVMtGpArj25cbg0SVCpLKKWsbbohfWKviHjxEKMCTMh8xnH5mQR5lrrPExjUcIxNm6fh/bcYZq3vt21MuXBcTJS4XBMn+p13nMnP8zFul6XvkvdIq+jlHMplgV5rrUpXzgZx796xUMkTilcbulrCF6H18ct3ziWt5vxfhL0xthmstWnbnO831+AfJ1+q6Kj5GsWvOvsujUmA+rsuiUgaVwExYLOxq83oFj+SVKFus7oE5yyCMUs+Ry0lq/fBrvPm25M8IpCsss7m4YQ0EeYWHTT81Dk78U16dre3ZotrQDhtz2+Hh57eivsMH0I5kzH4w35QJtMy59wx9qWEIPktFppLfne+6vFJgfKt6MoGTuyVG1hyIXct/JfdnfddDH5fG3aq4k3fvqu4+Ck/Reg915z1O7wmqN2TyRVb6EnejHiwokllFKqyH+QvuMAuah1mbSrnAp1VvJ80HKb4irT4o6E5DUC6p6XK/m48cNcLyGBuy4WE2/6UD9c9+EXwo5bHmrLYEcPnX1UCtd74bzirsaxNP7WlXJxwtks+Yq/kYZqNAW6fnQpL+XSlKaXSX3vuscINcKH2Y3bx5yKdCdPienMKfaDOL7VslqP8ppjugIeRsJWuPWPvt4KUrJifYZPxeoWnhuQ5MbQtSVfjcmC2pKvBPTCKXgKkCZZ4R9ssyxswDTddSW+zCH+Axfc0a5nR1mvzRrzRVdoeI649PKp3HVj4owIAfDCL18Dr/z69RFyBFdNitz1GLM8xSC4JiYe/OGuJ41rmCROd13KqUBJ4FvySUo+S+yjpqbk63XFJQefuvheuGv5RnL5vefPKFEaFZ9/zWGV8UqF1H0DG7fLXPgrseDQTY1Q/p3IIIUeKXZk9DoArTEkZk6R3wNuyacJlgBJEm8gCogMAHacOcU5OOcKIcOKqISFbC9ax9AUI/6vztY/V2zYzheKCb/FFw22cko8bCHg4jtXtvg6aDU1i1pnzLSiEE1OLuS5IXTdfMSnLodf3768TYPOWcbP3vUc2HlOy6oWFYP5/KLtEl2Kuy7V4g7cc68umm+ew+4J5bpgyafSzC1GcQOS3JtGiLi4rjVq9ApqJV9FSLFcqlp5aBvjbHIUp0kgvLJSnsV22kItj/FLETvJh1BXIXRDgSCV4WEfYfNkT06h/r1snduqxQVXe3VivpQAjehYu/0HyIkNcku+8hJvoFz15pLfX7N78c34Mfk6v209QE+8UQWqGmd/dMNSVvkqMl3nmD0NN/LvxrKXwzPl+O7Tl7N4keLputH2JKokJh+rqxlmPUxmHvCtW0Qyd11cIRD2gM4ukMCSz5V4gyIz2YB9ksHurtuOyUfsSrbl29//+LYguVCZklHiQVbyXftwJ1HSAQvzgyfMYk/Qle258iWhC/zMKQNw0C6zzfLKmahAf/v58tbqOY7ea67zCQ2LTE9zcJVRnHHxkrtWkstisK0HQ86hUQtCS23KPtDmrjvenmSryZRdo0b5eIZO6+Wi6uFBnagq5i39tnnThsqkT/iyuy5G07VAoLimpmq70O0F1aIulzPakq/P3yZVbOpJiTcqkKRRuOsSeVWRXTfisau2eoux5LPR6MTky911n7mLr7RfgLsdu2lxc85pB8Fl7z8Jnn8A7ppsQ+q4nZye5irLDT7fomff1GD0Ulv3Ub7l/XeaWTqvoODqIs6ST4nJ50y8waTb/newP4Nd5kzF70YM+K7EGx3410hmjXR9y/Z43TSacX+7BOszD501m0e4IjnpYfA1X+zaVl57bx/tHLgdvtucFn2kTqPJicnXAr8f5MpBEx95+YFw6ftOUhkAsm4O6Ht0Qz7cQ8EG/RvE5mF5v9fUFKkUPlRZvnnNI7S6SnuaJfR3iu1bsP6Z950pA32WQzZMGDewJENCdHjlB2kganfdGpMDtZKvIoSOF+rw2b0Nro2z3RXVs4mUJyZbUd2ST4n55y+P3etlVye622wa6zaKu24VJutOhUKJWmudcn6Kx068EfAmNg2PwdNb/It/zLrz6S2j6gXbN8iWKg5cfrJlbmb5Pk133fIxmVyCJyJO2Hc+HLjzbDh0t9n+whKqcNdV7kfW1+E7tEkdSykWR+05NzlNAHNz1RkbaGgNmeEjRVPbRBsgtuUNS56Gh57abNS7/9Mvg+s+/MJg+WzALPmKzSxBZkSdEC3TREDHhU+7znx+2ze2eSRdPFkrPN2deiBsK9aPzNU+AXR3XQrflPM7TsvvZYTW0iqlMlww7jtuY7eaokXTTMbGl80FFzlh+cNlISvD9szH7b0DAACc+7rD2TLZoPez/N8BzZJPQG+GFqhRg4s68UYJwE2LJx5CBznRGiH9Zbz8VcS4kxS0NL5Kmvguv6XxJtNCIFJcmrtuqSIocjhRgbKxSLxBddfNXasCLDue959XweaRcVh67mnOcqNInCXDMqBES0IOOIqE//uGI5W/bWONmXhjIo6kaVDlybLBq4vH2t1incIaPZQf534ZslHmwvybDdpsUcoICE+8EdFn/Eq+pvqvBW/5/i1qtfa/WDiITvy7cMHd7rq5DI7QGM9Q0xXXdxV+jFcOqvTUybIOP0XJJ5dxtIxuZebCyo3b2zzTuetmqqDFSwwJQR0atjr/3oKVgt71d9xYFwKyZXpeXrtOVThnGcCsKQMwf+YUS7IWfqMWSj7dgERPvBE5h9So0SuoLflKQFnzsPVkpCTelJh8AqSTUC0rXix0KzLFko/JwOWuWwQ0T9R4xokw8UTLp+MzrJwiBc4cbeKtG8UZ4F9+fackRySxQOh9KFeykhJvCAGDa+4FgLCYfNTTfUzJRz1CqNxdl1HWFvNN79JlZNf1oofisbzpuD2L371xspymbS69Z1USOjJC3HXDrDrC65IIW2/n8yy7aikjAWXc3n3eNG8Zn0IxJL9RlLuu/BvlWUZr5uN8WiWfx/ehfcfSr0oYB7u+cW6MwzfhP2H3zXcbtzDRskxS9Pm+sS5OG2WcycjrbdWSD3HDsYTe6MeuI33xO395tEXGKOuG674yXyp7Fv6LoijQKfC9F32Op7hhc71tYqy684MX/Zoz8YaRXZfOC/PAEtq/HHQUj6pMRuINiN/j1KjRC/Aq+bIs+2GWZauzLLvHcj/Lsuy8LMuWZFl2V5ZlR6cXswYVlIHpnNMOKoe3Mqeag3IMPQAtwxdWnkCr29Z6LlAt+YpNJnKNg7w5G11Ynf6mnaWsJYfrzVUnW27JN0AxGb3vt53fJbbfaKMJpx68EK780PPthXpEKcURw1jMIhv5Kz54Miyc3VYS5O66PfKsVeH0w3cpibJHsVIS1xxLVm/xluG+6vTuuhgPetlOnYy065Y3Pxi9/H6vW1zJos+bPgQArbhKobS4j9sUIkohrse8QgoE03YwzX8Ekxgdd9R1yNz06Bd7YcRNFm9z03I4CRbDa5d+UiaO82Q+eOpg/SnJhdCSW1xeo6kK9FwpYr6fRlOwFVBy8bKGuZC20EPavOf5+/J4Er8iW6gCG5pN1eKsisSCOpRnk/jbXJzl63lMRxvyR0O3RkEHdO0DDd2STwtdJEQPHEjUqJEAlJXX+QDwMsf9lwPA/u3/zgKAb8WLNbGRyrQ4pF7VQzx1DawvuinPZRuIYxBqdl8FyIk38n8jn6Vj3eg4hYtjQYMrDomwLyJTYum5p8F4+7R2kGLJt7ljidRXorvs6HgThvr7YN8FUpB7433Z3l/FnZ3Dzrqh7BDZb6dZMDDQtvgrYvLFPdPBu/jjvPVqTL6uuutWCH1D1C2llm9jxuolhMFaSaDluJ9p9wVhC4mxX7Vx2CuTC5TXUljJE+Sx8kHcgl3rByHiQnsoylYbAyZe1XcDzNv6mPV+ijFnFLM20r0KMGVM+3mM9RXVhI2BUOUrZUomAQ2lk2/8TSYueQ0lRqRoMfCNkVQlk/xMpCQpDjfzZoDLY8rsuipvu/KJAjkBz/yZU+CjLz+QTUMHKSSC9722vlsznqSFHk20pMi0M658fP2vvz0CPhLRjvb2w61KW3XctAolH/AV1DVq9CK8SyEhxLUAsM5R5NUA8BPRws0AMDfLsrLMD2p0CSSlnFJeq09cZKixP3R3XTt9APeEiJ0i6XRSrWVDn52eXbdVLjq7bp75K4ZOgkajTabpJ1xd8rc/dxEAAOyzIE3WyBR4cuMwDPmsYCagdduQFpfK+u1mad11aeu23mzPlF9AFRaR96zYCD+8XlVsPLFum5qMICGSZ9dVlD3h7cWNYUS6n+D1bR4ei6SQzx9xVOxhSERQbKRYSz55XkXnxoBDnfOGvgHvvPMNwTJRMIaEdujApRTtKI/LhnWYd9T51KsOgdcevXtqSYpfhcepVkLxRiH08Sqm4dBxKMiSz5ptGVGiIEWFYCQx6xAvHSEH/WOSKRk1ZDMAoB3emVzDyK7rBleRKiCsL7iUZDZ3XRub/FDhsN3mOONft8Z+/56Ohfxb15WiQpWtKaD2160xKZAi8cZuAPCE9Pfy9rUn9YJZlp0FLWs/2HPPPfXbkwbYRJzCGoEU2y3hQoMisz27Lo/XSfvPh03bx2DJ6i2wdbTR5q+WUd11kTZ2yWmJu6e6GKcBlY7+DD4ln6k8tNOiIH92V7gR+2muaV0RCtKXUcGp2hlH7QZnHLUbsXR1KwBdIUZfTlULTh88dFfcTcMYN7L+1r+N6mLy9UZrdhtWx1QyhdO/dj0AALzzxL2Layd98WpvvYmis/Zm32U+R5kxv7Bv0xWmgcKLZMmX0+FY7lkIK5aOws5fEGWzQX523E2sDMvt+E6PxQ175/P2Rkqq6LiB6yL1xof4jhMWAfxpMBE1++yJbvwz5IaVcvfayydhiGRWmpiFHMKglXiDZ1LL/W7ZbS4ods8m5LU5FmfQyo55EGJQ9rBqKdh6Ezbl+chYa383dbC/U1Yr02gK+PFNj0u0kO82oFPf9+SmlkzaO8xJ5TH5QIRbHdeo0UuoNPGGEOK7QohjhBDHLFiwoErW1QI7/A111yWW4ygRU7s/oac78m/CyfqUgT7Ye/4M5dr+O81S/vbJ7bpdjN2OBk0VzyKUzjjTXTd2EZ4rTWMsAlPEoele4o3u8OVicMC30sM3nVnFz8dpzznT1U2b9Qn72gvBRO66FPTs0i5UsK1rk4rRq8gSb7M5/RnjPMZJ5AP+sbQp7Zp0K6OQsYybzJ0KmnW837IsLxWScTN0jXPb0nXw8q9eV/yNv5P0Y1CWQHGou+suPfc0+MCpB7T+cIicOpZcjllTTBsC21vZ5weHwJ+H/rUUOXxwunwmohMCzmjmzcLKkO371z0Kiz76BxiRLENlWVTPi0JFatBpBLnr8sB9d0LYLflcTSQr0Psisvp4E29wlZxCKPL4XnMZaxuv0Zv2UMNtJd+0of62TGbtu1dsNHgUvxG98gDxnZx/49I2TxX5nJMffHUja3GNGmUghZJvBQDsIf29e/taDQkp5v9eTBqhSJR1rnmtHOTflqI7zBiE//3Eqcq1k/afjzAmoJgY1IrDY034071PMYkxWRNlpbrr5oiNL9gJuROh5EuwUexG4o2f3vw4LFu3LRG1eBld72Cov1/52wzJZ6tb7XhRCrfE7roTDZnyO3DV+aV9+HwTLnDLPuCKgYuHzV2PKleeGXsKy7/L5Bdyn1JeVu684Ts3wdeufJhFs5g/XAo7QhkqH53u8vXb4JE1ZgIXIURwdt0ftTeBMh+EQRhxEsJpY5nYKXTzdYQ5D9NkyTftOp69aJ5xzaZ87R/dCPv3dWfb0MkujFv32C9Uj2CLMCp9APjlX5fRaedKEeS9NgU/jnZ58VfxsRyA1lZyTL4mZ+HN7DNmWEy3dM0KLPl8iT3US/IBDf7w23Ml32A/eh/jg0ZNkC5yEzvJ/RJVIPawhWSNGhykUPL9HgDe3s6yezwAbBRCGK66zyRgQ1u5A4Z/YJVBlYUrszwh8ddDpvtnlmXGYv1lh+5slDv757fD+Tc85tyY2tx1ewn0mHz5v3EPky+o3Gx9i4wElnyum8XJWtov6BO/RZOFdw1o0PQ29Jh85jfeG506pj9i2XVbN3JLvnwjWcGz9tAg0S1JkEhCwbS4hxehKDMmXw7OJjRXunhjajr4YfczQlkK5LH7lsfWwVf+/BCrPs1d1y+o8clrMTRsfE78wtXw4q/8xbges/HV49rh2XUnjruuycVsGd878llPf/2qJej1Xg5cjx4yIzrOYl4i0GQpfxLDpwxKaSCAv1e8X2Ex+fTu9oR02Mr1GA95qqDEG9K7XclJWGTE2OMqPc1rSt8FPEFETyxhpPlKxvbR1jg11aXkM/5GlIvS7ykOWlj9vMlsYYcE0Oa3GjV6Hd7VZ5ZlvwSAmwDgWVmWLc+y7F1Zlr0ny7L3tIv8EQAeBYAlAPA9ADi7NGknMDhjrjXWjIMIZ+KiDl52OQhmDfppGYFnlmXIKZt/kvzj3avgkxff56Sdz9EbttuDjaeaGKl09HI+d139rpIFMED2kDlsZLyh1E3RZu6Ns5D+30NIvAIYHnMo+TR3P6olX9VZYmO4ffy0g2D3edNgv520pCd5O+fuuj3XEapD2i5XXUNSwxDkyLvzbnOnAQDAzrOnkuql7u8+ar6xL1e69Pf1kV6eqlDCNjX4YZ4AfBOk1jXhUr6m7h0xwd5xtzuX7O6A7S7oijKn3Al30incdccaDnkcsl738NMtGQLbzJbABbOmRDk43AEO3FkN2VJGlhebjq91ja6W6ebUlNJdNyYsjoxGU7WotdWT47RmSnk/oxCFXUgXGg92WeF9Uy7FOHarKdR2KkO553sCAcI592YZwAcfeDP8buwfAKBlyTc00OdMykJ5DrmMHLeatNe00OocNsclb6pRo1fgTbwhhHiT574AgH9KJtEkQOxAO3WgvzBpDnETKgOs+ERacNuYsDa2uL2olYVjUM43UO/+8W00xhHQN1yp3XXzhU2q/uCMa6I16TGfuQLu/tRLi79dwdupoCXX7d0JN8XCasTi8gRAsQTqDc3X5/5wf3DdE/adD9d/5EXmjSImX2wmUDqqVo66kFl+l87XldWACa6SL8ebj9sTdp07FV74rJ2C6seCtYFEiirukwEWbTpSx9DDXkvx1gmPns+3MQo8bzmQrC2IdZrNcCsyva+WFa+uDIRa8uUIHV9s8xNZaXjbD6y3fvOPJ5Dl+IGWyZuM9js2Em/4o41hZJIBNSJNywLl6XtizKkZ61e6AoqC8hUrQsueTasVa41OTRyhP71XwdaOe1iem3Obj/avfhNbM/33FZ3wDwtGO7k5h8caMNWzpqW468rSxGdxFsZfEaEXa9ToGaTIrltDQ6xp/HbHhh/lx5xMU45d8pOqsRu0cuRFfadglpmL9Y77hEnQ6a7bLv70lhGaIBEIXeyRlXwFn8glX0BH2DyixkZLsQnqZbeequCy5BvUs+salny2utVuUC9aXEJMpUxNvBH7TBO5q5W9kE+NLGv11YbLwghBPrb39WXwogMXMuoldte1/O5ccz+XywUf5eexzC5ihyEW737irX+WrN4Mo+MCDt51Njp2c8ja9cDmJtptEOfmilryeeiFbtD0uHb4lFzCuJof3EUoE0Jj8uXYda5mMevc2XdgU/KhlnzYe9m00kp7JpK8A8PWkXH4zCVubw4bOpZ8SJACRj/qxZjZOWLXinL1DF3oW9x12TH5AoRjIqQlYprvpf99LTz41GYAgLZCzlHYst9xycUd68o9t+gQz2M76t/VeLMJA544tcYSV/ndHiuVPsmT0oi/mVvyyX9PsPVWjRoYKs2u+0xGioGVFN8m4QBOjrmTiCcWa8h2ssV9TkrxkEXaVQ88Bf920d0qHV3BSaRFPi3MN06JrDxdz+0/SYxgTODRS1ZVdsTLODweYcnXI+66paBPjclHfaa4TU1cu23YNgrPOudSuPnRiZvVNsXyNt/ghbs68ZDcXTeS3IhT6WLCd2BiG++FEITYVa0Cp/zXtfCK81rZY50xxAJcnlx8OZDp5pYqHFoNgceposB018W0rWWMq/E0nUplgsxf+dsj1QtEs84pA3g8LMy6Bj2o6B80r5kVnbfZHgUEF0dV6S5g8RPrnSR72eiTKts7z7/VO26pr0LTjkhoCgF9zB0mP7tsXs/jYiw1QMjBdOhhtoCsUPBRYO533M/VtIx18lg5Zxrh+9LrM56Xfd6EWIv6LPd8Vq38BC8u+dqHaSyKNWr0JmolXwno5ckeoBVkOwY2C4dMuqY3QciaO4PMasnHRVkK0neefxv84hY1G1no6/dtho2JsItKnCJUWgpLPtdRpOOkuLtIHZMvxl13EkPLrktX8vGu8wvZcccTG2BkvAnfuBoPSE/BTrOmGNeSWmALgLuXbwypSS6Zb/KrSrwRAnfMNXcdXzf5a3ueTTVOFyNhlmaNUcVr6bSVPndxgFkJ2Us3BbCVCzn0uHbudk7XgCkU1BR3XRdClAEA9syWZMvjvniHojJyoeTGPPljnPmjW9X7gYe5UTLZDvSMdTLmVNvCg6vsSqftYw1YJiXCOPOERfDao3dTyuBKJfNao8mPjVlJHLSAF5VqrPS6QmsFdpw55CxvJBlC5NwoxSGnvg6r5SZa1m1Db7rB67EXzTqUfYVcxGfN6DMeUQ/N7HLVqDHR8AzePU4MyJN6qkXEk8TsUJQJV97ApQgAqygNHabtGHnXZNTrileA7sX/iWFb3QZ+ArzACLDcdfUCE6Fzh8Jw16Whm7G0OsqY8FUiVjV14o2vXmnPppqCV67kC43Jx0V6d11p7g14hNWbeaEhvAdREc2IkXZ+IwRetsyETDLmBkz2BpT+Jof8iLDku3uFqvjG2yh9fy498UaIRSWxwe3uuhUq+SLeSWG5g425rJh85VsS21hwPC42WRKlUKEoaBzP3HIlLddd1+mFYtF+2cY998EBfnPRjtNd4rGhP/7bn7vIWT4P1cRpNsq3Yi0R0MVNqz2K4tDPWA3txO1nFnfdrHPcUifeqDEZUMfkKwFVby+5/GJPfBXeNmsHj1SoNWAGxsPoY3cn2DfvqaMmNiZM03Ma5XFi7KpUclKmMP5kzAe6EFz/OMDq+2GyK/dyuOJw2iwlOphYbTTACSKT940Gb2MSkxAmldtnzBKxigVm2TH+CiUfMyZfKDLwu61yoLrHYpsM/HcO7gGIL/yCLfuusJT3Adu4dqzxY5Qm8m/RphcOPCafnWIzQLlgp8XQtCRBOG13TL6cevpv3tbWrzlqV7j4Tnu8vQIUd10PYl5JZ1Ov0wz/fstC6Hcp1wuJV6lYdmmUW/83iTaFgH5ZIUjoe6aFlWcfwWyODKEYmsX34c+93P9EHgtLs3jn/j7zZ3gTSogAl2gKWO66SlGKBZ57FGo0BVz38Bo7D2Fek+nZaMvPpIaEkBT90Jm3PWEDa9SYEKiVfCUg9Yleh24aOpTFIAAxJh8yqH/swrth59lqEGeumXpex1hAEmIpYKCEhUr13kKpsLPrIpspGas2DsOf7l0F7zhhkZseWUITKSym0H72rRMARrcAzDsgmn7ZSOJq5fgmh4zEGxq/CWbJ96cPnEwvXKTXzhdh1G+EKVRKFKfC4STQoPWJN+iszV7AwwwUlny8Q6Wpg3iMLzvKUVbGdiGukq/pGc+VfQ5TOKoSsqrPhm6VJ1nyEWk3muGJNzD+yNU0xBMjNiYfUolUytbW5KQ5SSz5uOVNrxPMxbXXEm9YLflY/Z1euCmEMe9kuobEIkBIbEz5HZQxqgsQQa632Kele1ikBiWmq/5+BAhoNgX87g6Cct0BvYm8yZEyc0/SuafTcvfXb169BL55zSNOefRrlG4mv/e8PFYt39M4wwjVqDFBUOuqK0KoAom/ePGDm/XP4CExaYrO4qYwumkKWLFhu1KeOmnIC6Ulq7c4TmV4Mh+911xvmWRLNMYGRgbV+ogq57t/civ8x+/vVd4FF7b2zxcWKdx10bl0dEvr3wmmwAqFS1nqT7xRTWKDVNh3wUx64UBNWZTyObLP5byjLPlspx4JUba1YEhMvjc/Z084YOEsJqeyDtXcpnUcRRWpnNdqRbXeC2JCrEKyuincaO2E8jshn1ShP2DH5BPJNmho1+3ROcntodHehCeycFy3dRRWb26FfYm2CJaUfEdmYXFMyQcJDuW5/BS3P74eFi/bYFxnkk6OUCWfGmONzu8nNz2O8MIIIJZ8zQA3Slbp0HFFrZQ1x2AKjDrrhCfe4N2Tn58ybwporZ/lZv7pzY/DuZc+YK9DeBSOhxZ/j2oqjmU8+vRWRB6Er2yZx7TGdMWsLCz56qB8NSYBaiVfCShtsrcNvEx+ZEs+Qhl58osdEvWB/5bH1pmJNwJpz5wS7xJChT4RciwQ3HR52LCt5eLozKIIcRaMSSz5nG+1NzdUqeF6RdOGfJZNiQaGCQHaM9nas8p1W9nusLFwufnY2o/To/KxmxOT75WH78rgwEeZWQNj4RMt5jwFe27XfEOxSiIpiQNkVmPyhSkvU7nruvtL98bXlRu2w9UPrFauVTncH/2ZP8Nxn7sSABKMqZK77v59DJeGeQAAIABJREFUyz2F8Yf8yG/uCmb/9BYzdubrvnWjt17oOi8GKZRNsd8GXekp2C6P/Jh8fAjoJJk55eCd4OhLXw0PTj3TWSdVjN8M3OOm/PyU2IFYaIKHV9Oz+dqgxLvzlfU0jf68Pku+kN7pTbwBuCWfLFPOPDe28LlK16gxEVAr+Z6BoCr5ZIw3BRqwlzr3hZyGjjWaZky+wOQaMYqsVIP9/JlmxkwZWOyq0fEmfOcvjyin9KkW8xQlhK9Iipj6pIVdDyusUkim98939P8JPj/wfQAAmD7kcWfq4baJh67kpz1rjIVprPt1xwIpHOUn3nB//7b244gwEBCTL8zljS6V4enu4OezZohNzGHyc1jq6fdFPG/XxjXVkJKCDDfxRkp33WClSslj8ulfux7+7vxb/QVzOOTpywCet9+OwbJEP2pfR8k3JsJcd69+cI2/kAXfyl0D0TE3fGyJBcdqyqtwl4SL/TaobRISG7Osg7EM1LFzwawpcPs5p8CHTn0WzNxoT0DVqRPGlxsDU36P+hSMtU1TCKPfDjhO70Ky63KArp10hZp5Sa2FyOgzqKYcOLkUl/K9ZlPAyX13wjuvOhbgCcYYW6NGD6JW8pWA2OxVNqSK+UFW8kkzwvsvuAMO/+Tl1OLBkJ+wpeTDiWKTkDsYN8VGHb9chQsBAO6u+8MbHoP/vPQB+PGNS00+lt+xcui45L0nwjF7zUPvpYnJN8FPzBK0gb6g+9Tgj+HNA1cBAMB0zZLPZGdTyExm5Z8b+ib7jCPpFmIHrbk0jnf737iYfGblpF+JJ15SCjf8/LsuP9NxOdZ52HxS5lDlk8013nPq5nAl3qDA1haKnISwHO7DOX6bNwNigVn5h9YLrkhbm63b6nYvtMNsl9nTBmE/LIQCNXxI7Pctvatx4MbjDEOufLEF4zfKU5aPlfjrxlfjhmlwhsUW+YGHiYYQ6NouZTOx21yItqUvwI4zp5Dd+pNZ8vnYqY3rJ6gpUoUAGOy3M8FCJ9nKUa9z96VNrV9wz/hF8a/07UpEKHtGo4x0MNtoCugDAX1inCBZjRq9jVrJVwLe/ePb0hEjjfP4Cb8Nr3v27mwx7nxiA8qPAmp5fdzFAs/mRdBTToTN7KkDMNif0RZpBBkpsOtghHZZ/RvbWG8daU0020Y72Ve57R+73zl0tzkwd7rq7pzT9LkCUzAhreIT7/Zd8RinaYkIjPc/mS35jHZ2P+ttS9fBZfesMpSmbzl+LzLLQ1f/nlwWQ4rNXhV6b9d3F5Od2IY1m023uJSgWE2w3g2ysSgTPtHy+0KkshxErjHqd+ZiKr+wdUOulOAoBrsdk698xTYXdnkaTVwZQ6kLkMCaX7bkK13Jpwr7jas7MQCHBkzeGdAPIat447Z+ZXq8qH+HxuTzozjSMu9o7rqKG36i78PO3Q4smYi/Dqt4MORhi8KzdaChPn9/gnS7nD1Gy5jQXt5oac/BDfZusH7vVNohoIaWapT3sdSoUTlqJV8JeHLjsHGNOmT6TmF8oPB523NpG95uD2+YxaFrzMXaZ9pQP0wd6C8mzJC1PztOSOCCAItdpZxglbTQcNGtxF3Xda/nNkvlwLXo1WPyGUWt1h+Tr+18n+LffPsmeM/PbjcU5pzvvtE3xBdMAm0p6QZWM7XFq8vaKYXyPkdO6djPXZGMJgaK5SrPks9zP/HnJW9CfBYT+nGRDyGyvqrvBji1z35gSemOGNt1W0dhFbJGKujqmU+DLPl4dWxwjcuuuakqpQAZDosrITwhSTydx2X1600aBQAwNL34WZUlX46LFq8ofmNrb1Z23QrWKr3QrVAZUFfS+Ph/1CYVADCF0tfAZR3s+p4799503J40oQCgKWLcdWmHCbpSzLWHDHHXpdTheI0ID030HjYfyjJS+DqeSSbfyuHTC19ajRrxiM9dXyMpxhjxi2LQ35clcckCkAdP+uQic7bFGpo9zZ4sg2WFkElp0bPMEdDWckoKmYs6IkdYuzaIWY/RZ0eu+TLOcZYfMv27l2+EV379egBIY7Hgtrp4Zky2rnacN92ndLL128nQdmGbBCODHmOzMZ7FJenJWcfsbwp5ZTeUcHIGfAttW7KMBWNP8nmVvfm9/xJyUZYhH8voL/4Z5UyWeDZB/72UOG/oGwAAsGj4F8E09EOqLGslbVDKWNqu+I4wWi7LtITuume/YD9MMm+93rPksyNWKep61kveeyL85cE18Lk/3m8nINUfr3hLwnUbdCH1K6eu8yi85e8l1bfhY9yKjUkPt2OApLnp/HzefvPhKikZjc1aryn4B2bymPu6o3cj19O/DQ5fykHbkxuHYUxa5wvwxxCnzMe2Et6qWAgIPZNtgCUlykr67TNeFEJolnzSQRIIac2WaV4M3TZ1qVEjDrUl3wRC8MCL4JL3nugtw10LcK3slLrt/+Ri103/V4Dfno3y4DyyvNCYN8OuMHEpCTnQn7UI1O4RGttYY22al/LRe2qT2z2Ots5qCSBPkN+97tHid5rsugT08KYphTLNpo995RG7Gos2g1sPt01qUC07jaDVDB7NSCVfCE9rXeaJOgc8S75W2feu+Q82n9J75/Z1bT6Zl9nKDdvJZLlhMGKxZcQdA0h+JSnkcVEgzQ2WyVhNIJJgfmAlP8hjbsV/LJe890R47r5IQgrKJpn62BeeBXDzt+SaxIpchChjaLK4nvWAhbPg70/ehyxb+e66GmfPi+KEh6liFqa66+qI9UC0HcTnT42FSmg0BStp3TmnHcQXrI0lq7coCj43+Ept+fyd034umw2s76mZxf24a/lGeHTNVmWMHOSmNPbI5lPIUb+hojz43HURHh6+tMQbEo+8eDGFieLPZpObLqVGjd5FreTrcZS1cDBjdjAs1ZhChe5Hpm58FOCOnyvXXIM5OmlCBn2SJd/rA+IRlgWS+zU6u/H4+DZJlI2YrUQSY9AY7fAkgW3x/pKDFxJqT+I2Ysbky2HdDBGWb7HuuineR56Je8pgeZteV0uYBw78Z+JmRY0FRdl+2nnXkelZ5e7S55bPb8abCZQHq8fZ3HBdpLhiIsasKF3segoln51GQku+uy4AuOyjdKEigW1fhbBY05MPVNJ9EOPCNt4l879mFX9y4zCZcxXjnI2FT0bF6omb8dZ5z/7QY40mDFhcR7FaLz1kZ5ZcLTr8Rm9Z8nHrhFl3YdZ4VFfV0P7EUazaENqVKfV8hzCoYQMSykLt036+i5dt8BcCLfxAre2rMcFRK/kmEEhm1oRRFhu3ynYTTp26HXVZsvLuuOiG6JO443x5LWmn/PBTm+GXf13Gohbqrqte5z/tuOaWPCETbySGrRnlxdA33nw0q3I2CXV/1K6ihyLwWgOtfaT4Od7XfXfd/3zdYfDZMw6Fo/ecW1xLsXiX4cyum8JSLIREyX12q5TACMAtI2YxK29qqWOfq9yzFs6Cl4Vsbh3WcyFyFGXYkvh4hpcRou3axej2nbAc9Do2xHy/Peeu65DH767rfhZsrDjt8F2IgoEimzcmX+Jm9c0LD6zaTKZVyTu3LlJ91lbkokHy6MpjIQSMNpowyEgCEWZhyC8vAtz5FauxRJZ8PoT2p37vob4f1jEZs2gF97pMf45WHEEaP5mH6yLF7f4dP/yrnb5Eq6HN8zVqTGTUSr6qkGDMqHrY4c65sUojUoBXh7vuh39zF1qnL5NOf5iN+N4X7cdffGhMsJMnQjUA8CglJYofvfBu+NiFdxMF9PPtCJDzMi4pdTlttH1M3Wi7raueGZOtTakit+sBC2cCALZZx+tOjpYL25Gw18df6yhQG9mUIJ4F7/a/WL+mKobmTBuEtx6/l7J4HUiQNU+Gi5zNXTcME7MnGu/q87vBT4ffaykbyKPdNnvPnwG7z5smXXfwSNSc0a607DAWNH5GplCkvo1SPo6myK5rVQQQnqPnEm9Io5KO2BiGWHNYD6RwCh1Z6i2JE7Z+pb89Mwa2sJblwrJSUf76wfWPgRB4CIJWDDSTStJYgQ5c9/DTsHH7GKuO3O4cOXMlXz804JMDPwbYvIrB07ausw8uQogkIRLY3kJOmRDSUhPqinZ83YTQlft0ZNeRybfcde3jZY0aEwn1jDoJEDKk6wMpNjFQY+H03HpWQ8uSr/XbJSvaBgH8UraHb3KzYbtmsYIhRea4kGfVF6oxVgS9gXgZ7e6l0m+r99hEaKNAJHLXzfsY5dvZOrgDiYcNKSz5sKqpLV5d43uKpExVu+umhuJqKgBgbCssEivQ+zE8sgxg/51mwswpUtIBz7ivZ9oNsc6IlT9fQyTZVFoggBuTr/VvGnddKxeCHHFtIoSAr1/1MDyxblsUHQofn/ucD9GZuLswQKBuy67yBBHzOUf5jhPD1q/K9oZwqHDQq7+69QkAAFizuRMX2idiSB/k9pyRsQaMNwU8vpb3XalJGzj1Wv+e1Hc3nDnwJ4CL30+uy/ksUutHlfVT5pZFLWoWNNa2wuMCHuD1xX18PRNz/l1lmXbgXpHiuUaNslAr+XocWCwCDMVpN4EmN7OUIROTrgBaLCyWDMQJMMvaCT1ytwLiZKXQYMqOnlyBuUALXdpSnv3jv+1Y9XnjtVD6laVMyGbGzHxKqsXmUyokoVPsUawn9EjjGEWtCsIea7MEoD6TbhnJ+YZXzjqMJZOOIohzeJgr9L3Hjts6XBvDlK5nk6EXvvDAnUqjnWWtDa6vyeWDqrL1Io+u2eItYzV0k38TrO98j4Kxsc07ad11u2HJ16q4YsN2+PLlD8HfnX9rKCGNrOWQzqkUpR6oBMrE5JMG+rzgB7UblB3yBoC/9sYqcqU043djhHVrrBbs1mgm0LCQHtm442BoX1XCtHEOx/M6+S/hP3wv6gbK6l7PxxluhBgd6G0uQDj7K55s0OQhX/HHHe/gM2ccamb8zelABo068UaNSYRayTfZII3CA4yVbopNg9vpUldwmQwpE1BncKYLTNlAORiyD3NCT/Fxa0qZLp3Wg4xYMhTYJvKOApNOy8h86prwJ6oZEBO2PqM2jdWUL7U4PYRQd93utUlhydeWfXisAcNtF3WqVKXHqRTCOd5imb4jWE1I5HJ/+fVHwPP2m9+5rv1LpYPea//b1+dXrIbGhdJ52f4G6Hxt67fx3NmoPLl1ReuEsPU3gVhugdptS75QJbkeVmR4jK4UINHX/m5QlKKeR0kRv7Ms/P7/PK8yXmNarOEYTB/CYxPaD6LdkPtj0teVH57rLvYBVtwUt80y4TYAkMbeiuQKGUMEpFGMctdPrsNX/TlalsNuaqY80m8wDTY46yXvt6J8xrW6r8bERq3km0CwKlosA7IrWxUp/h1JqmoRIpOcXdcdzyIdz1TAT87s93KEnDbidDInr5AFoz7pd7ufVbkRwLByw3b499/di95DN6vGbj3URmbigWrJh50eV472qzvms1fAQf9+WUsOalyyCr4K18I4hbtuCHqpxxYWmSXzySBTkkO1eGPytP8VAvaeP6NzXaTbuHPI+Ky89XtkK1bj79aVtVtHO7QsdfNumyImn9/tkr+W6B5wgQrLxwgtXyuRQahcoDRWym9thxlDcPjuc/0FEyGVku+UgxbCXjvOQPuQNaxHpYsogfzmKeiwx1ASSzWbcOLy78E82OSRxGm2hvANe0eKuy4rzI29sLmME3DDkrUoTx9Sv37O8OUTU1/zND0HjFR51DiTdHr6mlpomtE68UaNyYRayZcIdy3fAJfdYw+qGjpkkK0FAumnACXhg28Q5lvL0cvKMfm4DRUyedp0MFW+I3kis7UtZ2L0hakIyYCYw211IaT/l4OgjcDwxmT8b1jytPWe3DL2ZprECxIzgjgJ9hiHBGvhyPbUtz1bRsb5m/4KNm1lx+TLwbEKqFo5QlFQUdz1guf3wpKKYG3OUJi9uu96+OHgF1Fetr8pOGL3OfDitusyZcynKNet8V6FaLt2tf4+/8alZFopLGFnTg2PrdZNS2IUxSGn2jC51UpUTD4Rl7ijrPnLPYbFhWDBMJ7IXTekKb3uioglVCy+8Do5rIWm5Mst+VgeN9Ifj18PJ674Pny273vOOmx3XVefcCgAkxlpOgS++sHVatH0LIJoyK/FdfhkK2EcuArMBdyjRMWMHSx7Et/3o5QtyIviXiPUN7tGjR5EeRFin2F41ddvAACApeeeVh4Tipk1gQx12HJtbt503J5ww5KnYZkUGNpW/mMX3gU3P7oumB9WhmVxkEkm3ox6nfrpF4StcqYZu8mbxbpTjyEPZUPidyUjCNXGPStUBRnNqrTHNk1Xfrr4GSvbnGmD1nv4gkTDCB4/q+farEJ0yxINoPM9keIpWlDF2tK1MU/Jvso3kdJis1DWlvguBABAplqb29CxRve36VeHvtn6wfS65Qak9yGFcgHd81lIFZZ8CV7atEHcZbLUmHwVD1t5n+pHj/tpwjRF3t6BwpekEEUVOiUqX0cTuuvaQPU2MRPr+WlQ+ea/d5kzzfvKba/Am+iv2XJTnwlpxyPXG8oc8fJYlnxrH5H+oDf2Mm3sDY7J53fYTUBDRdEkSDV9LeajTB25ZTqc8R6N/SgRa9aWfDUmEWol3yQB6jrjGPf0W9wJJcvMeDE2KxmKgo/Ml7n9zIAek88aF43F0T6JxS6uOtcIhOTEECx+ammrYikC7zz/NuXvFBuyiQzX5l61yMxdp7XyW54qRa7egO4GROuJ5G/tey8C6LMrWVODKlfp34QwF7tnv2Bf+OY1rQ3Ku07aO5rFxM+u21bWWmYA6kaIsqnpk63NCXTUhFw0SShlvNZy8niUZ9fVCSewcCzqI1YfLnRi8vF5/ele1RPDfrgXfyhWPXB5OolKpGdddTfA7T8mf7jNpkimCA/ZXNvWQ664oqItMCXmsf37V5EyJp8Noe66IW7zBW3L87d4tg8e9Hh6SJgXn4zKN5v1ta+525TbW5zfZdPOixWT72tHM6VqQQ5HEIpkoRu09xYS5ijHyLjarq0xne7G3LrmVthz+j/2/uT7auKNZ/bepMbER+2uWxG2jIxH08jHocvvXQUj4/ipU0pLBt9kFmMtw5mIXnLwQvjMGYcG188kKwmXggy7ExTgXLfQkyjPnznE4l/I4bmPlaeCdMpb4p7F7a1b/sK52+B6FRnFt65JKE16nHnCIrj146dUypO8yV5xO8ATNyflXbh5YveIH1L5S0thxOHKLZf6MoDpQ9r5X8QuvivxEBMgpSWftQXyvqJZ8mHzVMxGHWUdQIMWPoDHTwDARYuXw3t/uRgljK5DbLRIMeZwXPsQbxx1JYXqPSVfDtxdV9l0//Q1ALd+D/q32cNIKDSEUOOplQqzXW3zp3N9WsLrkZV8c2EzfGrgRzDENaVtI3dVp8IbA68EC2cA9zdglpauIpcVRXOu5PPJzfzOXOt/tyWfVI7R1TnibR6O2CNShSIWs+0J8OdxW75tNfa+7uy1uGGD+VvphxLBqcuuhbmgJh7kJK1SlPXPcAOEGhMftZKvx7FtVJ14bnl0LZz109vhC5c+mJRPyCIgZQZGAPt4+t23HwNvO34vpQwv1kfHki9o7Z1wnD/98F2jafDdh92nv82mgJ/d/DiMNZpGq4a0NxfO5+nZzVI6uDYjmeW3SsCykaio6Xz7u3NOOwgWzJpSjTBtxLnrximW5fguoahibanzyBUjybvNBP2ErUqp4sDIvBaCLMsMa3OMWmyGTL1OyJheRuzc25augw9ccCdcfOdKtW77P16819a/IZawcpU9d5huL0h4qJ6btizyFO66EeNNx103EJGNZRvrq876Ozbe4ffRgV/COwb+DK/qv5FNx6kAsdXhWPIFfPerNg2bYVZkSbW23nOHaQAAcMiuc8g8sD7kVfIx4Y7Jl8hd1wVHZd3irRQQLf1cZbhvZOuoquQTopVNnsMBte6TLhZ9p9mAhb97I1ww9BmlbBMra6HfUkr22gBeo0YYaiXfBIIQAOu3tTb0y9dvU65jv23jVJYRJyprmRZhfXFVycY0pI5syecoh8bES5jn0rTwS0MHLaOUd5e94LYn4Jzf3gPfv+4xqwWCcpLp4X3l/U/BOobrgYveMyE2hsvqA43rphe3LE7Lbrtla7fB6s3D3g3eAB7wicDgZoCHLtUu0p4p6lQ8Ek5LPuIrqSa7bn1K7UaurI1vJ5/jJyUmn3xQpWfi7UaiB1qr+OX6yP9zJ+7CE4y7FTvlGpb51xK9Z8mXy6xZ8kVYPso04j4RSYESMGfZ2hpX/oW9F6y/Xa65d8sx+frbB0Uxc/Dy9duNaymy64Z2zdO/dj1SHyd21J7zAADgrJP3IdNXnqHZ2udkHk8O7qO4dHwuXur6N9RPBpyNP0pU8ukkPv6KgyRuIslcwFFq+tjplny+7LqY1zRu2Y6Z/LUqP6tvubUsOp/koTmyDLaOyOvpeo1UY2Kjjsk3yRA6vlPrdeIUxcUgiZ2GyJvlrDWdxMx73AWwvhCzW4S4//bBOZkziG1oK443bB/1Wq/4sGl4DN7149vgqD3nwkVnP49Ux72IkHa2vYLEsrhj8nV+20NE2b7Dctvs5C9dDQAlbqh/+NLgqp+55D7l7yq7T6Hki9j5VqF/09skV/qlbqse+nINuKxaHlzVSmgTHpmtXc45TAspJp9Pyacq9rjQnzXWXTfvpMaZg3Qlti9x69/8yFoAAOiXTUU2LGPzdY5pBKHCDYnjGuyw3ebA3Stomd8vvnMlbNzeVqYEDjhXPfAUNJv+7LqXvu8kslxclOGdsdeO0ztJaCxlL79PjYU77ojpFoKzf/6/xjXbs5Z5YGMjnTkW1vlljhu3UrbtndDvsarnvnvngap2WPrPv1wMv79zJSw99zSWu6cqn52ffssWgsnHd6fZU5IfajqkZtc4cOfZAPeoJZ3OO8QxUC517KId2hf9RgouS74MALaNjteqvRqTBrWSbwKh6thGvoEuVQbLzrjsH1pDgrn3ZVnRdm4zdPNmlgFMHegHOU3hiw/cycmvrPjLZSkq8oXIms0jxiKoCK5OpJWfRibL0thLyr0cTXUxFiuhq7/gCxKNY9O9OCwb1GHgmL3mwZI1eCbg1Dhg51nw8Oot8O23Hq1YRFSiPMt5Ifcefor2/FXIqSdOKktZy/mEy5zjnK5aGhYvWw//9IvWJrvsEahlWa8m3kCNFIp/BatNB/oyNLRGkJJP6pgzR9bAILg3lzILuhWr/4qN1od+fScAtJ65wA/4hwUUpZfLSqtblnwLZ0+F9714f/MAFpGniIEIAP3o8/qf4Z3n3wZvO34v79hx0C6z4aBdZuM3LW11xQdP9vIHKKetD9l1trKGobAYb3QOwLsN/XWGWnd9+fVHwG1L1cR5Ki2bUsWtJsJuK2ud8REA8LelM8Yes7zQlHy/l8IHyHud2PdrG1uolnw6dEWq81VntPnMlWjkr4/p/QFvk0N2nQ3bRhvwd89bBHCFu7wsE5qRGfmdi3jh2SfAkbvPbV/E25BjmbhVDpFVezvUmOColXwTDp7TfjIdbWIIkGT6UD9skk6QomJQafLsPX8GALof5jPpy7LCBDwkS9T0oX7l7wFPABvbAgfZZjnppIJvfZffvvB/V8DnX3OYl4ZrA9TJ1seR0CVcDybeSCxTw2EBQGrGit113/Cdm2DWVH5G2t/84wklSINj3wUzAQDgpYfsDFmWwZ1PbKBXTtVs2stbsnozvPLr1xOrlr+43HenmcrfqYPndw4IekNRz4nR9eiarcXvJzeYbnMAoAyKwRb07X/7Mv9G3BZXy8f7/afsD1++/KEk5yVFDDdowFm3vQJ2HTweAI6PJ4ygxSrMFVSxvt+80l5QgvzNuVn6G7Ib7tMArX50ysELrfdtUsVYLsYn3sDp77fTLFJt23rrefvt6K0rv/NYC1RZEeROLeCRyWXlZHXXpfdYzrPtucN0Q8lX8CQQK8QaH4WzrjoaHu1/NwDgSnelCzVa4V4anohS3NcUapegKPkCXi2FbWhMPv0gOMXIw/Eykr+bTJsT910w0+ibAkBpRL09se/Zdei17/yZ0njvVzpnCm/Te2H7aOMZESaoxjMDdUy+HoRtvWR1pWQOSNQNJB67oCPH198cliqegrNfsK/zPivzmByTL2Dsnj5FVfL5aOgWjvJfcpsaEynzPbon3jC6B37iMuXvTuINnkwc9xFXfxzc3oOZYzWlmj/DnBtOy0/ZXdfWThUrQm95bB1ccf9T/oIlgLr4armQpYmnxoVtI3bKf11LppFMbGvfrG4Ry/k85kzjK4+p4FieyxuuMlsqt2ro0y350COhzhzGsmS3LChCnuuOtrI8d6M7te82j3WMrPigcdT7foiNmWLJF6Jwifz+1K5W4bcWcAcgzt2zKSLHWUpnzuybeOyMbM60Qfju247x8rL1SXNt5gf1EGEwIsuJbQjz6VhtBwTpoLVrEdanLdhwy1X7XwcusMqg9KG2u27Tp+RjPkqzvVb67zccwaynrOJ5TIk1Qi35lG+XOTeUCVusTiGE51CBRr+YWzLlokWWzm/XbJhlrdiaA33u0jVqTBTUSr4S8bGXHwhvOm5Pdr2YBVNVJ8hZBrDLnKnhBBAxKY8d4q6bZTTVgK3M9EGewavVVaGCV4NnfvTVIREm8c830pM6qH9i91i3JV+nHacOtoZrI/Njl911exGNaOuScORfSow1XvT389UjAX71lmp5RuCUgxbCj/7uWDg8d7sJhGuY4rj1yfGRjBir2r9emTwlW9l1/fLlG5WR8aYRgylk2g9ZK3D7SMiUx5m7bFDkJMosFztqj3n2gqSYfKGTfUmLBNEZlTCgimDiM3g37SUDa+v5M4dgxhT7us1naadYJ7Uu+OXIP1BPW1DmBVvT2/qV77sMtVLEkvTZj7Clq3lThPaLduINn6jc7yzkEBoAlHAHrJh8jO95mxY+45zTDrKUVNGHvCOnTASRrFtYNQFYAAAgAElEQVQYzKJOdA5f9efFXabdn8jQAE8toT6735IPDYEjVRsbb3YOiSbzPqbGMwK1kq9CbNhGyzhqG1ZsFmEhoMTs4G5Sq3AxC+FAcYUCwCewDAD20JUqHhixnyzxLXR2frfacjYALrpuI3gTIe66tL5czrOzvqPGOMBT9xqWfLFwWRjJ7bjT7KnwvbcfA99887PVQr1ydNtDsAWDr6SpYjc4oI5z73vx/vDm5zAPi9Y/BvDAJR5zX/VebLIBa1VCmRlT+uGFz3LHOo0Fx5JPtqoYGy+v0+Rjb5ZlyryBNfeazSPFb9mS1qtETDgv92X+92lT0oV2oZDvaCBA6yTX+NxrDuUzlaCEt5gArl94c9EP9uIOCOLaJyYmH0eZ4QMnHIANv3nPc92ZRy0sfK0f/O15b1qUfHp9bv9I4J2AGUvIWVTNm3Za8thc1mHY5u1jyt/vPomWmdiIyZdgvLFauBKvAbT6nC2mtMuQZWei8QhmyGdPBONW0grp3lijGTR/1KjRi6iVfCVCgDqgnPSFq0n1bJOITVGVYiEdgjIPOWKsGS84S40R1HLXBbjsnlVw/o1L2fQ+/epD4NzXdmLV+Zo4xWIvFDlr5eSWWMcF6kK6c4JLf38xJ4sh+OTv7y1+s3rZ1Z8F+NYJAKvvTycMABoUP4duYXHqwQthznTNpTGx0nEyoNFULflYw0mizhYzPMryfuDUA6yxMv2gP4s7i3jMRtpftwqrS8xg1iaa7K472ij3+8rddcuYNt514t7F7xAXRB1sS76Ah8pZvO5bN8L2tnULN5FAf6S77tTBfn8hRwt2K/GGHW55sPeaNWkZO5uWzTwZyiGo96QT5S/j2XvNg/PedFS4PDgbL1Ik1z1yD7cls83qn+LiniO2Zypd27MnKfqFbg3tXZTSGtOZ4AO5FxozWg6pYq16y3cAPjlHE4LOY9PwmL8QAnneFpBm+aIcUsjewBbi2Hdrd9f19E/fEJB7K0kHZJ2blsQb0mXfux9tCCnueq3sqzGxUSv5SoQeO2fzCDHNecnjCvUQy1bONgZHWxdSyrSZXPfw094yOVobKAGfuvheS40crSebP3OKcnXGlAF4I8Pt2hZjjWu554Oruqr4tUzMSNBZCj3XeyoCs/fwSZis6GUpk5ff1vp385PajfIsEUjSWdx1J4IFCRfUt9UQQskYWfaee9voOHz9qodhvNGUFp/h9JLFElz7CHpZIGf1To4RyhpKzSpOzjmHL7K7ri1WErdJFoCZ/KWzGY60SLK08idOPzjpYVx/n2ln5JyHIvltG20EWSIqCmuyuy6RD+E9bdwetmGPhdeii+OuK9Vy0hQC+qJ2EnG9RF/ffO1NR8Ehu86xlSZxNhXifhlzS2FVvZx24rHH7/W560q/GeOM65NQvkuNZj6WBY89bWWNr/24Q2beflyltDx3WKve9HXjktD+xe+2sGk7cX+ooS/LWGMkP4Y7xyFANS6wKvkc4gqLctDGV7djxOCLqai46zaaMNjD+5caNTiolXwlItRsmpt4I4Rn7BCGTSplxapQ+bbwhcseaNHB3Iy19X2mBTX34Z9euC/80wv3NWjl8L0HWwwnnV5o/ETKpkpR8gVxaQM/iLWiE5OPwaKL82nQXJ44Bt5Tm0as90htU1Hijff/ajFccV+FCTcyijUNjmVrtynWWFPa8QwXzJpiqxKFr17xMHz58ofgwsUrpMVnDywUv3Esfh3ZtJSWpIQwdvTHaQlICHXXDQ2ILmPR+hvh1qlnwwv7FivXhQCArKVkCc38SD3tMebcAH59WnxbjuVVzDyk98yxRjcPMITyD4Y3f++W4ncPjALeCTx0T5snNUtlyRcC/buhyJIrO5XEMBZlRZap907afz5Kk3yIQFFgkJQVHfgTb6jKlxgY8QrRMu37FpdZrwy5ktCn5PPR0cs71qcuXrK7buy8bqs96szAZke/1sYpRkar8hu9ZlFhCtwtF3XXlUg0PZZ+ehWFlOX7kMcINPxofjALGYw1mh0jhTomX40JjlrJVzEaTeFV7nAnkRTuumg8Oq8Y4QOgvEgvK9ZcjgxoVhKp3J5tm8nYpCg0izs+D1K8QiKtiZZ4I2jBllip9oPrH3PcpewIqnHX/e0dK+HdP7mtEl4AANAXruS78oHVyqL5wJ1nwxdedxj819/ysupRsXW0dQq/ZXgc/nTvqlJ4pIb+2bu/2TgrsyWrtzjLVGHJxxkb5SF8x5m4Ypg6bwkBsPOWlhX5kX2mZWUGmZIBvkU7EYQ9xUDIvFvsfRx1U1vNYt3ykrtWcihE81NQpllw7orGZHHwLrPdBZbd1KJrua0rCqjoyzJoiO7O97pyLYUorvY/dDfcStCIxRyALMuc8tvWluU1fwa27ydzmXgZ19v9mvotFpZ8bji9IJBGaYJ9fWp/FEGz5EPr0suGwjgfc5rdxfGykcaaxOaWqyvx9DW4ANqhAX6Y6t976f1CCCGFGWod6jFzf9So0bOou3KJwAbEff/tj/Cen93urOc2ZY4UykoY4O7lG0si7oeexcvWBGZsHrOMuXmlyVCcDCn8On9d8y8vgN3mTvPScVmM+KcjP3Iarr5w35ObOnzoxhYIL/PE24V8McRZCFGKpljDjo434Zzf3h1PWLfkK3ElR+q7NqVjz8WFYiLCkg/DG47dE+ZOHyKUDG+3r1z+IFz94BoAQBbfPYq8i/W75I3sSy/7v9c671cSk4/xCPLj/uML9rXeSwUjJh+Th10mm6VSGDC3TuqekjqHYD2Ba2WqlI6p6wT+PHqCNY7LZuhB4PQhx1j59MMAN57nrK+2L70F8gPUuM9XtpKyPP8NX7XW1pVr3VI3yoogsjKLy8Om5PO56yY6wG4RoBXBLKyKLKxed5iGUt7Kh/ksefNxlNJNEaPApSkrKbC9YyWZn/C/Hm6byWMDWtWmHAXLuC3cQzI1xicaFsUyidos+TA2iiVfb9hh16gRjAmyDZlc+NO95bm8UcdvffD92S2Pwyu/fj1c8+DqThk0doHdErGS7Lp2K+/ONU2+vixDT/xmOBbGGN1F82fAnGmDlrsdGO66luKPrNmilSNahrDv2d5XCymDhI+3LTR7MSbf5fetgp/dvEy5FiSlZjlXZuw70ubWFox7osfkQy35evuZto7KfYPWu4b6+2D/nWbCl19fjpWhDfp44x6/Iyz5hDu5DEAaS76T+u523udYrsllBy3az1Qb5yyLi8nnrOWIbxTCjmvxFWu9DhCm9AvhOkZ2l7NT3zY6Dkd++s8B3FVwLbNy5Ssq2bD/8Fb9/DQqTmup1jvupruuUd05jGlrM1sxz98Yiph8nraIGelsY4Sv+XVX5E3DY3Dr0nVefu6YfC1aNn5RTqREbwnu+JKX5xzANYVQlKv2NsEVWmWjL+OPFz4olm+MevKaU1iU/wKEc63RtJkAFnTVf23oh04f8lliyqRGG6KOyVdj0qBW8k0gWM3JAeDP98cpDm9/fD0AACxbt6245jaFj2IXDMO0G43Jl0m/23FjEnlYUp7b6q6r/f3xi+6JF0jnQbBs1EFRzio0nKdw7UWUpaE4cbFSA9vEBfXjxO66s6YMWO+RxLO46y6bfkiYQL2CbOJMT7EHHH/+4PPhb569eyJpAsHYHHMghN+dsL8/fkJ5Rf9fvXLkeOvx7kRKVRnBCmm8LDsja4jiQkc+txYWOZyA/0weLvD2YLTCP79lGY2/4z0Nj1UTH1WHU/kqj6OWcqGHckK01lZx7rpx/V7/bmiyZE7W+Xe55w7TIYOMpPTArL1CW8XWxWzrJ/2ZDZdEzaL3H35yO7z+2zd55UDdML21wHS5bK+ZhFM1KBPgJ96YO33QL1fhaUJ/sqbhrst/q2WO7Lo8Nz6ytjRe2H5LgGyh2blui60nPJZ8IPzje7Mp4Et/ehAAdEu+zvgrK/lkuWdNNfuJfH9svNk5dJwgYYdq1LBh4uyiJiBaAwd/eA8ZVjYPj0s8+bjqgdXGNZ9CRh//4rPrphlQ52mTfZZlJCsOexBmvZybTqhytBe9K3N5qRvQcU9MvhRB7EMxjgRrD+pziRNvTHVYlJI2LLrScYd9YAwGYMvAvEjJuowuKflSWUD29vqwMyLSNi3h85gAgEGPEi9VTD6qlfPJ+y9w00kYp9QnUwZgJIfixsuzl3ZbTXChGzW2vhUH9wSfUpaZT+AbF1NYEIYAU7bxxpMwuXPLJLRV+uyHSEWRQHfd8WZLAZJqrAsho69NODRclnyLdpwOf3r/ycY927NSE2/4rOPcrowWHky99L0r6aF5bPK05g2bYgxXvpC/Baoln/T7jn9/iSofKpd7fYqh2dQTb9BRxSikP0pusJESrvlICAFzs61oHTzxhirzDsNPwAGwtPibksjnodWbi9+2LM99ipKvU2Tv+TOsdDNoGQMM9MtXatSYuKiVfCUi9TqTpKiylJ8N6iBMGbqwBUVVS2frGO9x1331kbsadVuuUBgP3KFAtwQkySVBV46m2nDoZLD+IEAYijQfd26/ciFfDNncIbD2q0oZgimtwyz50in5Hn5qM6zZHJldN7HSEUNXNs2Iu25WgRxhFk7ItWhJysMXL30QPnXxfQDQkdMpb5Qln3+EqSK7rrJR83xYxlBx70UlSJTLEp7dFMDzapSDK31e4vPKN1/OxBvKb9rBGV+OdLRCgbZBl+RyboolJZ/N8lKtzlQwCxEXniOyY+jfKsfSypZ1VgiAOdMGYVr7AI4iYe4tUuZ6xh6Tzw3dojbGqlboDYXxE3bli7BXk8rTlIJc6+eO5TS9jmnJx2JoXNrzsf+Bd/X/Idl4qOhSScYMfppKEfw1kvigrrGaEu//3Pu38P+yDxd/YxaAfxj6GLy1vxMGQZbDNnYNQGddLPeTmYjnTHE7a2XXHejtE9oaNciolXwlQgDAsWt/Dy/tc7sRhYB7Ov/DoS916hKrUt1OOzKVD52HPulgMujuFgAAt/zbi41ySoylgIVuDlton5BJnSvHms0jcMA5l5L4dqz07PQKa5xElnzBSLAiGot1p6EGA2Fg8RMbnPdpRlblW0d2xTAmceKNqtGXZXDQJy7rthgo7kD6nXtzHN4BfnvHCu9mrIrsujm+9DeHe8so426zCfDrM80yEfOFTiN2vLQ3r93KasvIGJsPV84U48bWkXHksI3uJpw+8Ybjobqk93Mr+fzjaIySjhoo3w56J8FKmu668bwEgNJvKP2YaskXA9vakqfYhCSn9S2Wtj2CpqnR3HW9CHDXNWUwkbcf2l8txBpCKO0eZxkt4Ig7/gM+Mfhzds3Dlv8Czhv8GnqPKlHoPOWkSVX0tuGL2COQ8eSQvsfhs4M/QsurOj7ckk/m6VtrjIxLiTdqZV+NCQ6/HX+NYAgB8NoVX4LXDgEsGv5FEnr4dWW3geKI7BE2P1+W2NTDX8h4ik9akiUeZNDXZxpfLZw9lSYTlnzEU8ftrmt/SBddatu85fu30ArKfBO6pjU9wadxVtVMpA00Jl933XXnebK97j5vup+I3rmLRk63oCs7XhgKwua0V4D1or4MYPtY+VaWscg/gQwADtttDpx5wqKk9P949yo4ZFdPTL4KlHx5F6Z886q1Cs8yGoPNAjUPQq4/frLPzWKpBADw+T8+EEw26G0Rn0mnjb0uX3/hup82KYH1m02AX78D4An7HFvGJpoC58aVcFjibi+/NVXU56vEu+O3n+muG6WmbYvUUUfFHopzkPMS2rX8EW3zsK/9lS0CwbK64I3RAoUYWu87f3lUq6StvciWfJ5invtG+YBDFaG76zL6esrR4MQlXwboBxjWrsvyXHm/GXYpBIqlnHwdeyKrkg8/YBLgnoOber2GeRilWvKpEuYYUJR8nevY3CE/11ijCQNTauVejcmBWsk3yaAsDqThWR+GKROVLyOijioOPbyZy5D71KDmanBli1sLYQFpuOsW/4ZP+cK/rgIAgMfXbjOu+fg6Lfna7alO+vY2yPuMLQRXeBvEL5ew/txtd92RcTetVjZnDwylY/qNZlfypXTJki/VMNbXCz6FBLS+59YLvvi9J+KFmuNRPHzDbxWWfPnYQzKOVf7QN6hCoecn5i7ZSg4V/vxu6vi9j1/kzkTs5Mdw+001bMh9FMCu1Dh+nx3g5kfXwYn7zWfR3yYp463z2+hmgPt/76QTr5wVQXTcY41d0ZsjZu3WaOKxt6qCkVQtkShK3giFvENB0czHhgi+Gn2559sUibORZAIyQtddvviBHQY+bwImf7Iln9sQQUcnJh9Cy8LrA/9zB2wZ7sx/1ibhJHxJig7Nc37rT+aXah8iU7RdRYcl4bbnbCkHpRJj5p5Ghs2ST0684QrVIaRqGQCMNQQMJEgEVqNGL6B21y0RqU91R8abpBPIGKzcMAy3P74OAPAFhZt82MCI0bQG+w2si8bk02lJ9c9+4b7wxmP3gLc/dy+jHpphSrrkctuIM/Snb6445bgK0Bha9lga5SNZZt+E7rEjKbIxVhA7sysWKmictu5YyvhgO1yYSLCKu/5xgA3Lomjf9+Qm5/1qLfkoZRVTPuVeBgHfrMT0fS/e3yuTbsmjQ28v6/epxOTrXM4zyYYi33wPZeYhhc2zIFjRgMyaNqXSjjOmwL4LZqjtQ3jh8ibeJYkP2BOmSuTjgjO7rtKX8XJ9FmsYCq57+OnI7zeufXjuujTOLVdDHK6mbgjhXW9x3T3lvp6vLd947B5KmUN3Uy2lDQ74J0nj77zLW4TOzza1a3nqJXDXxVAo+Rid5KoHVsNfl67rXOixab11SFQNL6y9ba+gKfCkTy1LPgcP0L7hse0AALBd4F4vWIIXAIB+JSYfziuvKj9XKyafXb4aNSYSaiVfiQjVt9kWsP9mOX2nsJHPTlynrt/+yyPwum/dBACOmHzFxiTdSJhl6eZOnU5fxl9mz546COe+7nCYoQVppTxy06NM+sk7j0Ovu/pLxtuzkOgWMfkIyi9q+3mV0NLvUw5aSKTKkcAOrmWqFQnddYcRS74Fs6bwiFjdddOBS3JnWAvnD34BZoL7FNaJCR+Tj18ntausHYppbusfm7xPPxTEgTM/VGPJ14Ks0LCHwLD9IW08hbWIAdld94R9d1TvBTw61ly7Z2tgyZS3wgHZE9JVnpUVxXK4le2WNiCkOBzIMjAmdZsCHc/oSFDyyfEJbcUJ1qypEhRx+0SRXRe1nPErpZU1QMAzpHLX9QNRGmjVnf1dK9y0KMGNfkSUkXKQGGU12RAojQwyOHgXe0gEfTij9lNUSeMYG+0CMA5GRjYD/OULAKBaY2FImXiDmtSLo6QtI2GZ0d+TcwCQe0yWZdbXfNvSdYpAmfY92UIfug5Ajey6o62kkduho+QTmnyY3P0Wd90CY9vh2uY74Fkbr1MMPMYazU4W+Ql2UFujho5ayVciUg/vlNToVJ4kt9PA2HJlwrB48Ex4eebCKrODmtl11X9DTr2p7rohcJErYsQwedpPyM07VXWl8QbjGNIFw103/IUMI5Z882cylXyVZNfllX/fwIXwgv474fT+m8OZTqCYfBh81gIXnHU8AKgL1k++6hBYeu5ppcplg3VOyMKWCbYxF2uW/v4KsuuKzgbZN+Som4Ly5o6UlF/edwsMZE14ff9fJAY8E56QedJVI2TeoswHtk/LZYHlwmaKJR9FWYY8YxWWfFhYjQ6kjbeldUaVeLV8eVOtB0Oo6OtUGg2PfZrUjziPRvOKcEhVvEe8UE6fvR6TlS+J+mNrHOVZ8uU/nfKvvr/42Ze5vznuk4hiDqhq/6JLmH4s4CZd4fYd1W1drfydax8F2zMJi2xNj7tuU1cOti35httKPqf8sruu1HfQKuuXwizYBqeu/LZyeXS8CQOFF0mt5KsxsVEr+SYQaG5G6fih7rqOSaqSmHxynEHUV9ecWDJiTD7yesVzzXmYm7CN6NZ17pK0poloHMvtCvWuAJAweYQRBCgcwykSM1SQXZfbdgfuPCueadcs+dL0E5+77rwZhEVrBfAOSYmVrfkC+tzXHiZdq85dl11Wr1hkiiTSAgDh7AsZ24VPV561lBJtBYAWKp1DmayIo9Jj8LbyQphZLfkw6xHCwmTbqByTzwLCYUq3Em/Q3XVxoAdgHQLe+tWFHzVlMd1144XR+xH1rTaawqso94pnWOl14E7q5qGbg9FFbYkT2EQD1yg+Sz69Od56/J7O8rlCuBdCaeSibxoegyfW0T0e9DGm7CdRZhN9KhQtiYq/lXvtBBpaJZuFn3pfKtC2oG4IyjrEYsmHbcra43kT+pVvdrTRhIFaM1JjkqBOvFEmEu/erOOiclhvO7HjTwXJYpgRIQ/s1sQXuiUfsrgw3XUtJ/wGLby+izYFQvvXXs69gAvtTjEx+fLnlfVa7nga7pPmbio0fMpZOqF0lnPrto7C9KF+ZZPJtqYx3Mjap9WRsiksGDK9+8S94ajmXICnI5kiyiXOM73qiF3D+AZ0ClwR4a6TbzS6krkYgfW7DlS22iwM+voAoAHwyiN2hamD/fD+C+7wZt+lwv3t0K04hOMvqluXrQ4tu6+bh6vPKHeYFjwp47PqZWN6Oee7N9cOBK8FylqHknwm9lMOHAvc3gF+mmOyJV+ADFEx+SLHP70/kg7DPazzrNdc6Od/mKUbl6r8PKEhR2LsyeztyViUsjKUy/sA3wG1ev+zZxwGW0cacNHiFc7Y3Hh3TT8PU5rnVV+7HpYiCfM4NEP6KpcHANZCjnko33PpSj4AcH0FQujZuhGuNrbSjT5JyYd+N+11vMj6lL3feEN0vDB6QBlco0YMan11ieBOGacfvgt85oxDScO1beyhDM6Zo74Me0w+fJhOOhxaHkQPk0KZRFuWfInkgrg1aehkXKYeoAzaVpK9oc+IQ0L32NWbR2AnSgy+4U0Aqx8gy8Oz3/GjK68t0pLvuL13SCRIGHzWAvk6stufhHcuSGzJN29624IRAM44ajf430+cCofvPjcpDwz5OEf5MlRLPnWDWljMcZRd0u/5D/8PfHfwKyYfS3kMhksTdJ7LsORjhFygzJNbRxvOzXf6/oy7fOG845Ug1u9BO9zB2qBb37LzGxay2xpecEx58fyniLOMkhXgfN68tR2tsBCgdDu5u7nVqe2xoST7Kqsy2sNO99qPWfOFhb4p58vgUu3E5Cvj/YSZB3AUfADmM3dTDyUEQGbbJ0LezpqSTwinzEbCjtxFXStjkaj4NSAl3hhvIBah7XWzgL6iWtaO314rRmpMFtR9uYfwpb85At52/F7W+zQLgHTyHOQI5IshbZwL/EGUoOlIqQwyNUlF+z95YWITk2RRY3UToi+SrXuIincI+UTKtSRyLnI9pMIzLCZonNAG3vwUwDYpu5ruehLx4jZsG4W50/GsYQp++hqAbz4Hv0exMIlEBR7BJtDsunRUkbE1B6a8920kcvm6YciHSWYdvwNj8tme/ldnHQ+fOP1gmNlOarTDDEL/BwD4/ikAl58TJAuAZKnteC03P7oW9v7YH2Dt1hGpot8OxvsKJab73PgReEn/7cotl0y2W3n/6lghyPYIuWA8dQNlLmjFr5Ms9BxVlFkxsKNj7eOyFA9x15Wf23oQR0q84S1SPQhCzSUkXHEhyzK45l9eEFaZ0WjYm9Fd8WjLUPfYK+v4WmlmZC2fywopUp01PgoDzVHtUL7DT44rqkO5ZvFS6chIk9I5LiGumFbolnyuesR9hBACfnrT4477eB2AspR8JsPOmJxuYDAsVxkj/Lt+fBuNh0zf4bfe2oPhe59m4a5rLh59ewifJd9vbl9uqSxn1/VZ8rXuN7X1TSG3V9IaNXoftZIvAe58YgN6PfWCr+rh5qBdzLha1S1i/Uo+G/RJr0/LDlW1kq2YlGMsADWhv3nNErLAXsUbc71F4mqztEAud3MaJW08v3IAwBf37vwdqVRbvWm4Q0oIIx4ZKtIKx+LMIk/K/twVl9JIS75wHR//Wbl7hoN2md0TcYFkoNLccB7AA5ck5bPXjjPgXSfu7S+oY/mtADd+LZhv3oXVgyL1XX/nL4+AEACLl8lzur6pyi35GHxL+H5olvQC+WUHpcym7WP+Qjm9gMembFrtlnw4RS8ochJiseLhQ/iNENpd8CHF3Qd+eOYxcPIBC+zMCcL0ZQCL5s8gyVjg0b8A3M8bWzBJ9P07Oq4+vQTglu96nqV174MX3AF/fWxdkIWUeejMfJFfPQK+uuSl1tuhIXRSJt4Qjr/slTinhBnyy8SNj6yF7cyYxoW7LrLzLWOFU4WXDJZ9vCyeet8RQlj7uM1d18ieq6FlpW4qFvPjKgECfmJT7kq89stWwkJoHc6jMUfzmHxZv/JcLbl78bSmRg0+aiVfAnzrmkfQ61UFYZb52Be/6WeBVHtUTOI+S9vJG3ch/EGOAVoTunJSbxG8WDB4HizmrYa2mf6YX7zsQXpdj8Qp4zD5Lfnsf5WNZNwiY/L9868WF7+bTXNTwh43dCVfBQvLVGWdQGPyMaw+uqxEc30zl77vJG/23argVKr8+RNRirVeQmEFA/5xWLEOSrBTc9j+4FcJLPVxo5N4QyXEy77oZpxlAAcsnBkVm5aL1h5WHyNxPLhqc1DcMnV9YCukjrNYW1UcxpgGj4LlRQcu1CuwWQQlzvnJqwAueEsQPxn6+gWV5PsvBrj0X425G3uHFy5ekUwWDM7vcfNKpIJEP9CiPvzbsycFUgwHmSfJztLEriTHMZbxhmP3AACA5+xjhutwWvL1pBlueoQ85nO2Xw/fG/yyUffuFRutc4Yo5h5dMeief0133rD3ct7Q1+GWqf8HAAAa2Icj8sQbfcZzdQz5emONVqNGKOrEGwmw02xCTK0E4GbXVSZn+WfkuGUbcmPImgsJy5m8z10Xce3JAM+uGyKv1QpQ/m1pIG+W2wB5KPBN6u5NkfnETlN77V9Tlu4spJy+SEcAACAASURBVFZvGoavXbXEuM6SJu9Ykdl1t450FqdNY0ETACMmX67MSNfWHEu+qFf8hw8BbHoS4E2/QC35OKSdGScrgG/DX4p8914EsIfFrduCfFzqNcvC1Chi8jld0LAQBvqLRJRpPt4OpqGGGJ1Ye+phl+1AjzL2Uvbq573pKHjsiQUAP8/5mZu4soE9y6qNw7AMy1JJaFySco5wuIPJFTIOcz9Fp6K+gheSKvFG3la3fvwUtCimTDKUfFjjDW8sOFAhtynPapfMwkFIlkO+bFkXgyqv8ZTamUVUlxDmH15ysqKZNXDaC8vf2pUfen7x+/h9doSl556G1skPb8o5XzOJVmEdWMa0LZT+IuBDGz4L0A/w94+vV8o9vWUUhJhupYG56wrwKPlAW4uIvI8xN8ASxhzZdYXVpASgdtetMdFRK/kSYJ4lphZ3IvUN1hl03E4vvWcVvKv/D/CJwZ/DBeJOL+2+gMEqQdgMFiinN/rkTPEsyTKVWqzoMYqqLFCAYLdZJs1UcMW66QY+9Gv/N0LGmvu1C7yn0l0D9A0Su3sh7rrJE28wZJo/awhgU+v3p151CI/Rrd/v/I5M+BAZ0i8aPsVocvnGRwF+fSbADvuwqg2PtRbh06ekTbDRa+h8d26FG0AruUSnoqZEQN6r1wIuZs6wpFY3x298w80Z5xsEOWdNHYTDd5tDJ9pGjGKBEpNvo9WN2N8ASsxeW6GmnngDo+Nl5ZHDTeftP/wrbNg26q2vXZV+d6Tu78tw98+AZxjojxnMTIYLKMmo2jDddV2s2t8IJexL1vn34ae2dK67yJe8wgntX8qag1HPeyDC0n5a/9Qpoz9d2HfBTFK53GuzKit/NE5qG8FjhWF11l1FlHyIkWn9LAOzj7RcfF3uuv7sunbgZdHEG3J23WLczZXA3dqp1KiRFrW7bgLoFhD77dSacLiJK3zQ56WPDPyqdb3R2eSrk7k8+GoZAhPMC/pAnXLetI2xSjwli0JP/TszY/JFyDkRjV1ipivseZeu3Wrn5Vm54O+MtwkLwbAldksQ2Ueu0ogE0GijKYSh7GGTa2qb2xIsN6jt/9x9doSzTtqnkGHKQMQUgyR84FjFhFumBVjeINd+dwfieiUhynLugrcCfFJXtLTl3vCEtdq2LINlAwNoO86eOrnP/CiWfM6KbbBj8jn6U7BMYFoa5X8pCn7Fws8P3zNVMv8Zc7hZBNdNhY97JEu+ChIc+XDtQ2vgruUb/QVlWJ5trx2mwyuP2JVewYHBRJZ8ISBZ8nWYOf5SIZMZlZQEDPJJIPOjWtTrbaBbZoXwLuqrfis0QiVk7gppatGWI2bunTZEPwwrJyYfcuATWT8O6qF153eurEO+OYfAZkw+/PAKF8Wi5GsP8G95zp4So467riKbLN5E3PDVqCGhVvKVgBcduBMAgGUBlQ7YQGSbVCZeIFGbJZ/srivQYqaij5g9r7jPhzK5EcqEFQhDahfZW5eut94766etrJG2hUQuy7+85AD74z54WZR8ON9eIaJivOkOQkxCBZtPaqypvz12d82qI+LZIi35uh2Tb8WG7c77Ue66919sXiP0z39euABO2wOfl2ZOicuwaaBH18cusXyJC/C//aAqE9C6lveax5DtbEywcqJwUVu/1W4BxkbAWBizudSb6KGnNjMqUzqitD6wlScoKtBDLAJ3o07Kb8ciN/ltEN51nCVfHGjrG9xE0umtEvDmmu0lqfN7Z9IMkUOvYXi8sClaeJDHAV3Ro/790ZcfKBEub+LI+0ro3Hvdh19YZIUn8Sth36WvxUpx12VYftq+v5ayDgAW/0wn7lzz2mLyFYk3XAJZxrrcYvkdJywyygop8UZx6OZgUaPGREKt5EsAbLyaNpje9cmMNWeOdpTNOHXRYCVlVST66Z60/3zrPbU+zsRw28FOtbRCuiWfrVwMXCesqawIw101qodN1n//3b0AADBXcnE3mmTpdeUI1SOQ2+aOJzYY/XCXOVNplXMYMfnSg7pYTRrXLTK7rpzFuGyEPHbyxBsEJcQt0+x9a2ZqS74uDDzO9X9hNec6FMNMV3BLPgCAY7IH4AV9i/UaQfKlgm7Jd+3DTwMAwH/8/t6kXCQWleP8G5ca12LkkNdNdndd/2FKfBbyuPo+JbXiTicsQR0CnmGwnzmWPf2wzJBR0SyLeeJ562OWoAmUJ6R5kkDXelDsICLL64t5Rm3yDDKSq7z3kRxz05knLIL3PH9fC3/aupqKZjEH8OsCAPx/9r473pKiyv/b994XJjAzMENOwygiQ1IEBBRFQVZF1+wqinmNuK5hXQM/w5rT6ioowQCKElUQBSQHCQJDznFmGCYwOb2Z9969t35/9O3uCqeqTlX3fe+N3q8fmXe7q05VV1dXOPU95+y6De1/biyhK9WSBJi08WnMQiDD14HrHl5eWoZAJ/DGpZ9Vrrdt406WT1cCBr1oOu1ox05bCRCUR9eVzHWze4nxRw89bJHoKfkqRiWsqc64MmvqgHZZM4/NTx/C6fOxCq7ipCMuP7dcmw8j3VzXWJiRZUL1PVPGXNdy/VuX6H7aXDJoKaE9p2xX4zRD1VPcZfcvTeUmwMeP2hOzpg7g4NlmFLR/JsjrjpOOez5+8vbn2xOTSr6Jw+SrlD1Xksk33KzeRMiGmOeu3Pl3SZOoUqbVWwCKACP2TR6HyVdEsRW4YOB/cEb/99hjt+2Vu+ZTljsDpV4SnrwOP59/DKZjA0bDtCF8GGOSpTXG47SJCZZyzhLgyH1lAlhRKFYMRR8Q4DJX/cj8yv7u31+Iq6UgCFacdBBZvxgEKVazMbLz3C6l3DPrhoPr0g2Ft2quy8yjjSfKc1ZUx4QwxbRCn5uc2bgTo6fsdhu44svAyseLHJ0sYxXZfiy+/AQJXnPNv+D2wY+w0nP66CaLexuLRPJvIejxxT7upDCUgBnjrkTgjcwnX0P2jSOIwBud/OM+ZvfQQ0X4x17VjwNSxVM1A8SBu81QfptMvrxUpXwKNVS/wKfMYstguFlMLInFKZ/Pf1mSqEuEBJRy1I0Yv3KX3rfUm3+8po0quiNHhOwQnFPm83adgdtPPBrTJ1dsJhiAaswpwmTobSMrrl+z/04Ky5EFQ8kntH/Lo83cXWxd5l0aGntTyRey+BqJVPKVCZIQglIRKSnkG6m4+lceXXeCHYJzWBwcZtPfH18ZXLatSM6b8rNk0n+yrYqyGbrhB6ihjf1qT6obnLLo4jdCH9SVOZnzHxYoj2MrinGYUtXar9LmlRQs+vhZ1Sea9a3DnzULc5hBEArQTEN27iAln3D9VPCwxSTcpZAfrzXeXjtspfx2se9E538ceD+7iMAbetlGGVXNQwtuBG78EfCTA/NL2RxATb2lFTtEvbsRv4Ha85TJT6FPMr9XX7GZ29ZuqZ9Ys3Ky/1hbPnUtwqjxfb8H/n6qNW1G8qhnjGMhgD9/EgDQluaHLHdB5Jtgi5geeghET8lXAajhqCvRZ63X5QmUIadL49a0W76P5yYLg/Nli7Tf/r3IO2nTMjItZyNaXtloVw7YNhuudlcDoIzHpBG/0gjZXL3xZzex0wZvYv6BD9bCdD3jw+TjYvbMKfGZH75E/V2Sybe52X0z5jKoRKk2tKr4uySTb4zIDeOGdZ3oq8FjsNau6zenciL2txaml7s+4UMfberUCDWpZIKrMCgzhG8YLjHG1fxm6Cw2mPCPJxNzmpLWH7KyxVbZCA3jFQ/Q67XKMDlz82LWjctuS7OHj5GJ4Aeta7cFWJGaQ8qXfxDv5q0H7YL9d5nhVpxUUTZZhTgmn5yr8sOlvBDie3UE3qjCXNuQWV6EKVOvZ2D+1598ozeNjdVPKfRsn1Rqrmu7bq91W/fJlwfecDzpBe9LzYItY9doZ5DIzXXXLQY2pGNWG7WiDIcSuIcetkT0lHwVo8ogB1SkWBlZmG9O5KyYE1zbo1CX+zGK6bf+AH/o/zJDMgGtQi+/8lgymTz4zl+5kdXeNsahbZ7plplfDENQP4FV7nE3V2O083hieRF1d6Jtdircz5SGeRIbsKLg+OTrwkNxNsEXfPiwcj5rNmkBXYjouiGYM6uEwjEQnDd44G4zlDpVsrn58fOKvwM2sNRifbwDlZRFS7jr/65f3pr+4WLyMcwXa2UOTYLK5SM1jaImkMzUqWrmqDD+3jzaqjzIUwisRTOUfIrfXFuidmzgjXCmWbWBN2imnH2zrdfXX39fkCEnrvgSI5G9DmF+EPU1lAcbV+LXi16Jd9SvDCgDKKveUxSF0juiFJozNbc+tDz175Am47kYkJXH9nGIgjksJdJf1Y4nGaMrb9JrvmlN2+3ZMNaKxMuErAAyk0/x9UikTSxWYikjj75uq/MtT6zE5Q8sw6PPbJBzqPktdQYAtOjgUq12Zq5LKHeTYkbP2rbod1v2mqiHHnpKvoohEL/HzpVP2W/CzJSeGAS23WogL59CmY2JWpJ7EVpHd9kz8oLnnT+/1VSYdP6nQz1d9TAnfMo46jTLkeWBxeuwaSRtlySZQAzwMYhiZr3ftZK3PATtvTetBpY/ol6zmutWB86YdlBZ34qGHbPJ5At5srcetGu5+lSMl+21HS484UW44bMvA1CR0mWz5Gy7pIKlcvPhMcYQHAFrJCQA+oeW4n31S4l7ll2JhD1ri3xJzPu265xDKs998/CH/kVtcEKRTxlavRev2YTn/r/LcNYtC6x5u6X/87bh0Aq/DM7IwhhnJ6S5rsMcdov54h1Kokzxtd1WfmVXkH8vANiYBiD4TOM8VnLOeyvT5lQ/vejOp80yjLUdg8GpYa9kIQaX30PUQSqHvRYW1jSxh11RgTf0Mq/7jjVtNYdeWiXHyndOSdjiXviUrvL91LeezVyXbtvLOm6PFB/qBJPPGpzu9JeRl0eaHSYfEQW8DSnwhsiew8ME6aGHLQQVh9ProRgkysP0VWEvtK+zgB/Hg/TSSBJ//eUFQcocoOUUPzLlqKQ8LfFyON5Y9CpdeNdiDI3EKT/H22TgH2WKs346Y1qLTplapw1a5J7yYmDDUuArknJnDKLrlo8aGYi7z7H45ONhq8FGiUV6xLMyihIApg32Ydpg6rewa4E3It/VFq7jYyNJEuz/t4/ixX334cF1HwQww5NDbc9f9n0PwGfppB40YJqdVrWPKPRvFnPdCnzykV2r3caClUMAgD/fswTvOXx2qTKCfUx1mCGkom7+34CmP8q2TNL77pv3txQUZ65rfRyyMbswzkoKMtk3c5XmulUhRImQIfMX+5v3vxDbTPH4s3U8G31HOO5Rqavx8iuDsFxUsD4zZZc+HJ/pLqeOfx34HHABgOfT5p1JtrDW6kWuFRxKWmOeZg4AMe2crr0SDD58EbDwGv2m8rOWoMuUhTiYz139xC0r2VSCBPVu7V8Oaa6rm+NKoKN0h5EqKGT+3gf7sjlQer6knj9jIXcL3kj30IOEHpOvAiinHuMwOCQoJkouM6BbBxT+RZodfYxNSE3ZvxDTHVFAzKYh9L4vz7wFq709I5QhyMVEkdFdgZ7ixrS0MAQp+TYsNa9ZFtBVmroE+T0CELkEL/7844dK+RqsgrVUNUyfP5Y6fn8v4HdviyigrE++iddm3UAtARoj6wAAAgx/W7ZNDHMMk1NNRQmzRmcZIjebUllKRen1Sn3yaYywjmhXk3RrbeSU+gwv6r0s46i9t6cTsQJvsIrzJg5tK+ena+2/sCzIJuBs6WirTKk02FfLLVrscorv/abHV2D5ek8EXY4/MKUurGRhIHT2PhNKI0CdQSiLr6ial1izkpl0n3zSuKQv+y86If+z6hkp6ytTL/534O7fOdM6LX7abeDJ6/0FdsPRn4ZuTNuKkk8ui0jbtKRND18IJp9FDkAz7YpvsEDoofPwaNr/+i1MPrlugFzvf441UQ//uOgp+SqGEPFLJMM8VxtfjrYsPgXndDYCNqVTt/zucByDyxvR9ESSd65lm3AouJ7OJoLbIqVMNYyFGjdf/PuKXUBU3UW6FtK+CrElH7b0Ik1X7gjBN0fiF0Je3WGax0SyjL/BBy7k59VALhZ9+BBj0W4BJ5gDe+O+YSnwiGlK6i+Ar+Sj6lKlku++p9cqPjq7io7vRu4YkSCRnOlznlmV267w2yozchyxZxqMwJgXlB/Fr74qzXW1EnMWYQUDqummhOdewxiGhQAe5n1H8qbRWpqHMT3cbOGn1z7GKg8AMBQepTkOdiYOxzyd00vPeO/BMRXjw6nkS//ljV+FnO/99eGg9Bnc+tRq1ik2Kfx1plpLVfnSXSUuzeSzpzfe25K7WOXEPEaQL0LXi/77z4AzXws8fJmco1R5XBh6w+qLUJR8oy17dO40rTkuZn3Mvhejy6XnKKYi2YHhZgsDjRqtdEyKpyrqPQEPOnroIQI9JV8FoAaOKvw56DKmDTaCIvG9bK/tStfBBp0NWMWgyGHfOFyN2PNAZTlmi4pundHYJvYYU+6JQq6ZKPX4R0Rp5UpJBhcHXWEoGKiuEC9Dg0LJQB8+vPfwPboqPyzwhokqyY9/JPxEdQ3SgMvyiaU8p/tgLRWqs1A684l07THFUThRJlma/RoHsySH+6mfIwpFLaMU3zZoDV1YErCz2NNVNQ489Gfg8at4ZcpKPtt47FHy/fyGJ3HRXYuN6+TaaOMK4PvP5qVlwPntCvcmvQqUCrgkgeMORUemVOLp+EKU+yCZfK5yBNR+XkWAo5jxQS9C7t+hPUCvL5VfCehCJRAqEUFpo1iffBF92c0A08c0R9IVj6b/rpPnOUoZVcGAqEF/hm4EzGpZ6kb15yblP49gnOZJQPvqA9SAH4ZMaR8XqqgebratEYPbScOuOO1tfHrYwtFT8lWMKk9uKH0WOWkQhd76xaPwFofjeQ7zJAZlpJIDvAaFySeAO59ao5avu/fI0iqn2eUQdYKYF+5YxvJebXzZGjjtUChIQ8ssUXELxX/iolztSilXhChdPge2xXG1USD9SqqkmyyETMkXUYavHY574W6YPrkvolIEHvgTfT1vv1hFQXUvc0x8OP71i8CfP4nQ53U9JYfZRDH5PnzWPFbZdUskQtdGzXZLv5xtwBQWr1T3qQPdcsEs8jFsbEYjvXQLAphyqmLGlkhV8ulj0dBIgHuBDc+w68PBS56zLUuYqoyx+cYKf4P1bm+GHQwb/fDWLaf4s7xdB5G6y+Me7aLGf2itMvmq+UZTn3xczX1IdF2pjIpHkzLvR61npByPqTNLhPa7K0y+Fv/g7P6n1xrXsizUNymEaU6egT6IMue1cJ98bQz0ST6epf7YSupj0qY99DAe6Cn5KoaAKK2YySZtfYC0bpyEunADgK0GHBvKCTaCZbUPN9cV+NBviM2VruhLMoo4c1nneH825ag3mmwXFn9ciWPs/o6FCVilMYP+PkopV8bg5Qoh8qhnOiodSsa7o5Zg8o3ZkLphOXDe8fS9jR3FQSSzs+w+fc3QCN78s5uwcOXQ2LzKm08Cbv9leD55c8p6ZnoLwHeXIOdsW+8FI1Os5Y9CKPmkup9x0/wShemgWS+uea5b81WWvkxbymQU63fgYfLZy5durHgMGNmIKmfAOdtO8SjZJCWfNjbQOr4IJV/XfaDa65RZE/KUfBKrkZU++6Z46Ma4lyRu5Qbph1r7XakbH0WWKZjepwjrL+W9NUci68FD22GSoB9M62v96/7rZWbhepS/AIhWXFgPfYztik8+qYzhJh24J8No03yOnF1LyE5fQefOenVdSQbeIF5Z6CHi5tGWyuSTxoE2zMAbSZLJn2Cb5R56CERPyVcxYkwyM+iDtb5uss9PxYlspWsMilkmXZYXH2VP3BLwov/5k1Cm03SS1UOjpAQfC430VWj5OwT/zIovP8q1jm3zOR6BcnSUMrcgIz5WsOuVcNPjK/GjKx+tRJYVi+8C1i/pbhlejM+CbvqkAIZfy2GGfOa/lqpHWbOfP9+zBLcvWI2fXfe44tNnrBDik0/+pd00wWDy+eDaMiRJHK9eziUk2ekcrWkAq4bMDpM2beMxmmZjeBn2qML0t5rrupl6rNJPegFw9tu79l5oJqp8n7xsz8BErSIln/UbzjW55q3svfOCRzuemjSlCDPvTb89d/sF6Bat6evK2ju0kPJEhLTchOzHPnNdALjg9kX534qCeLPJCqsSZRybqCbpTCWQkzAQVxvTtLT6dYtNGUrOX8Q1l7muvF/FgpuUO7Q1l6loD+2+w802+i1KvlZSMNyF/l575ro9bOHoKfkqRpVLN/1k0qaoSFCMRVz/JGXHLtMslleuC3T4dBW6uS4X3KSzkyX4z78dDDx1K53AUkVuXdJ3ZQqhoj4p+ZJ4NWoZRVa3Am+MN2mLjctPrFyk/j7KmesSC0VRbeCNTSNxJ85sDK8HTnspcP33GIkr6jjPPAg8fYd6rYs++Wxv44IPH4bLP/mSagoZcfuF6zYumJdu3KYO1MfGXDcStURivemsCEb+widfhDJkTNRgY7sxkR2pu3S77GjEkUy+Mn2OpZPWDlSiS3vyOnvuQOYYC9IcsaE2VSmqqj1s1811Gcq5ECYff3402bGuwxCWT1BWse5DyQ+9dI5RH1fEXUVBUrpzuZl62Xf4xVfvLd1W1ym/uWVB/rfSnK0AJh87pZTH6d5EZxs6B7P0X3Ywv4mtLGqgiX+t3YSsDTI/e3O2naKkI31MkqSHbD9Im+vmV4fXq/UgmXwmC9TFyKTQbLVVn++SzDbqxuVe4I0e/lHQLSct/1RIlLFDVMcO0sY7q7UuI81YwTc4uuZEDpNEj65L10FfkCUdZ7/+xeBLavekf9xzLrDrId76jBVKmftasgYFPg31TROUevzgbdabfiIltp28lnvaUua6o5tKlc3B1EH7NFGJ0+eNK8rL0DC5v+5O8NNDzWu5ki/8fcY2w0Gzt4nLWBKxLAIX7ur4R53c38CG4QD/ZOOCzuY9lFbTyRk7HuvmTlVEPi+4B+7Nd7VQeRV5K1awAIldP5n7PvXdPprsjj1tZXLqrZnrcj/5rm/vhfynuw8sqe+MnaQ7vOi6fnTfWtdep6DourEaZCbaQqCGNl5evzOsHA02353Zs8r+NWl2sPr7+keW539XNSKkZVAKHqIOWjvK/UV5byFKvoh+KgSwI1ax0zrudv6VNauEQitP1b0VcRVLsI83LsQnGn/AyEgDl7UPyc119WCI3v1dfnhmr5uQr3fWryvFNMwE0EfScc22Dm1NAW2sk8bydlI3iMLFY09s5WwPPfjQY/JVjHzwYYwNVz24jLyeDYBRPvn8xZaidytR6BSZ5cGZtGMWk/pEcyxuAE46GFRreYwgAHBYavaFjzsfI1FE+vFQuJXzQdi9ifWwOTOV31tP6ednbm6uuDYpnGZGG5bbAy0AwJ9OIC5W+8b7LVHJqkP1PfRdh80Oz9RFNso/k9XHrKn9OPvWp8a83FBuDpWLVFoT0XXTQyNueUXCWkJnckbstB3SaIkKc92ka+a6trVDtlYRjiJjfBiGoMycw8oa7ZMvPHHIs0yf3OexznSzwqg7oajKXNcOl5IvO7y1JJBNQAP8lc6eORn6aOGDEMDxuBQ7JKvZ5dAo+pqqK+soXx6/Ai+rlVMkxkLpTg5zXWUslaPrQt3bKO/NYxIfBKJuIW4keDo+mh3GK0Bg5Qba/cafTniRK1vl2A5pf52RpNYAGVNO34PSn5i0/xTqv9R8IYQo5Hb6ReYCg/TtmbObVSuu7acN4AMv3sP6THoVVaWz5JMvqedjYTHubik0hR56cKOn5KsYIQPw+8+8XfmtD2/6743DLZzwO2piF+SfLvDMkizXiRtVnFQ5Jax4FFh8p7ZwoJPa/UCkN74ufgKseMRdZ8uq2dZuvkV5NmlSYj/7yr28L6QSxtQEgru9qp9g8xM6acT7zpv2wx8+cjhfSEWsOcOniuvd/vbN9kALAPDYVZXUyQWXaUQl3bILq1ZrvTaudGhNygTeGMfv894LgGUPjF/5HRywy3QAhalPpaiwjwhFExU+EwokwaahSf5vdc9hc5mhlEBskMogVwypO33DXUi5MgLTi6zseMmsepP+T914we5bBzEs9auc8fXUd77AnSDfSNeUEiaiua7PJx91P3vv1nn027vJgqSy3DjzfYfkybnfj4DAzknBmgv53k/u+1H+t8z4VQJvdP6d+vvj8Kv+73Xus4tIZVQ0BFmZfPm7ki+qylU7k4/2kx0FQqFrjygNI1o279X5Gl+rw+r5ys/1m2ml5v67zLBKNAKEdGHpkc3hutKNNtc1kdWR0tm1hZSnM6aKjirihkeXmxlyE3vpkhB47f474cTXzLU+g1ofu9K5ndSVdICkGPkH23f18M+HnpKvagSc8Pugn6Jk5lA6EtGW6MZjcwLRDaWT3G7Ldjgy/aOv4+z2pIOA045UJg2faUqGpCP76TWblBSUfyTOooy/ZKfvm5szf1sKIaKZCrH98cjaXZj7q72A4XBfX5X3woo+Krmt/+3g3TRnyh40JSXfge+qpD6Ah526ZqE7M3XyvcdLS9VHR1QQhQpZLTL4wRUIrHoC+N4c4OaTLZnix7RxXQv+/v3Azw4bxwqkyOYEORpfZYiMGEyKkvuQ9OLWDI3g4rsXExkoJV/cGBej5PP1LaEpQAQSKVP31wOpB9AOk6+C4kLnuSoCb7CGuGys/fwidzoJ53/oMMw78RXuRDvsj43TaUNiziNtN21QTXfhx4DHrpSlSP9Vr5J9y1B2+CsxXky+VlvgVzc+mdaBpa83FQY2pH06sC8yktvWzsfWCz/QVisZUn7izqMhdI+gV5c+RDAuae9DV0zJTD67uW6pQxFSycd3j+JuJ+3eaUcCqx73JiPTBCLokDgS2cGubq5L769cindTdqpopZl8F95FzL8W+SGP3RbCqnROo+uqRfVUez38o6Cn5KsYty9IeUakLAAAIABJREFU/T1UMUjwooUB+unseKAaJl8ho9k/rSNYbYTGkoLJGOLkeOXGYvHQ7nR7yu9J8RwWJp+VyuevSyqVoK/ni/DuvDybXF8f/UzjPNSam4CVEZFVPY/ivl3tFCuEwJ0LUwV5qfXQyMbi710OlgsIq4/2O8onX05f0ZR8x/8ReF2mxKqmP7Ucz5dV/Zi529tSMEqopp4jknKJbNPsFP2xK2gBnbFmQi/wxnKAJ8r62+SjrMmzU//h0W4o+fzPncDuD3f5esksStDv+Ot/edBWuPYrCR6rs9TqnGPE8guCHl1X/bvb5roSI0pix1Si5NPLZDaOS8k3f/K+7jJdFf/atsAVXyrMdWuNLJO3TrVa4jRBAwDs8wY8tc+HleuxzZggAe46CzjrTUZZNBuN5uGoKfy1IZ+xSlja+tqHn8Gi1Zv4dTjPfjhH9jui7WJM60Nha3Ou+LLBQVhlIMm7j8yCK4L/2U1Z5b1NrLmu9zkI8/qWg8lnZHfq+DQt1mLadLrQHdFmq+yl322/MGRm6MaXlzH5dOU9g46Q/rfTPtOHTLcdSv0776jlUkVo36AQ6cwZsm5u620tsbJFUpPeE1nLHnrYYtFT8lUAebGtm+CWkhux+WedUCZgzQw+33KuaF5BEFl5SuHERWDSmZ5TcaIu5klkdvLpaK3Atve2uyeBrzQyShVzIopd1MkKz2B3I3FFdgXzFkT6x2lpC06Z0VjrixI50mxjSAtKUErJp59Wz9oL6BustP3bDp1NggTzv30sTnvXQfEFVMTSkp2LO5t01RP09S5G160MESaDFFiHMoG+RTOTveFmF6Ixl+wj3770ofzvdP9ibt5HbAxErR3agT755EYj2eNeph593WSEewqvElql5CAgtnkp1iefL59lqaA0kGySFVy31ghw4/+ZSj5dhq2tSeHSNe6CLBpZX1fHtyr1v9WZ61pg+f6nSAEoQudRXvJQZX5FR7XSy9ED++kIbfqfXVueTZbDMUfYfKABmk8+WZHUdXPdNru9eC5l3MIq+8b+8ilJJj32VolMUat/13rgKIBeS2RXdlpD7IdlFh4r2jXN5AsZMg3XBIqPyESaQzQFdc9ct4ctHFvArmZiY9NIC+fdXo2TcV2RYxClLTNGIoQxSHUT1DKmlBkd0mdXqy6drSwqpzjV2XPZhEJPWAxENDEri5boSxfdh592FmRj8V6tiJjofDm6/ThCCNzw6HIIIRQWZxD0U2XZXLcep+R766k3Y/FaNYBH3DrCtvOvbkgfabZx8+MrnUy+StAVn3yORrWZQE90Jd/Kx9P/VwKOki9MsZbNC10x1y25dTbNjjqQ+om9y5hMvnjTUGnzXvKZbNF1hSx5LNYDQjbXLV9erATXO2mjzjJh3GHaoCNRR8nnURiyoNS1uo1kQn2zVhPVACaRB3yrkwjcerp0uKGzDKW/A5/F5yZFZfIVqLdHYYtOHmJdwgfN2DVSyWOZI90DS9YF10AGvU437zuZfGNirmseNgnhUEgbbhlcsjUmnwe252D58l2j7i/HYifQCmDykVzg7POwRBzO3/m8MwEU5rokKHazCFPqpzpBuT/qjHq1jG5GQ+6hh7HEBN/VTHx885IH8eSKjcb1Mn4Sspw1bQNiG3bk6y5fCFXCqqyraHDM5QgBbFxBpim7zXL75KOfcLjZJhdJ3M1NklgmSuLir29eoJbBKsFEaL6dsRwvTB4stYAeb5xz21M4/he34k93L8aqWCWfizVVD4jKK4Hyq+lcrFgXpBZFSq2CDWgH37r0Qbz99Ftw7yLaFyhQUb/gKJPe9rssMUskWS3fN1oq8IbvfgUN9ZMDgbPeWF4OG9T4aEe2MZiITD55syIQQsOTyh6c3smvsgFDIJvrJpII13rBfkt6JqEq+Qp0a7MSziZhc5ykd7MzluM/H3grdsBKb01c5nWtpOFsiizvNZ850pGoCSCJ0Gh5mHzU3cjXRq6/sk1yUjPuVzWtV8XkI3HJZ6y3ZH+xYXXwp02SxGAa1dHCh64/FF9s/FZJmw0vhu8vNngvfMHKIbOeUeXF4+qHnknLTUBW2xd4QzdTVdY/bT6TzzueWHzycfdmO7oU/kwmn0JWIMCqypVfViVqoroR9KvVabrsm2oLu/UTzeSzt4/yjaxND1zbFlXEa2s35YHn5C8r9DtLI/rKleAq6bewjU8PPWjoKflKYoUlBHoZ5EoobSK0DzfhK8LYiUEeBJOKNxOKokz2hDqwVZCcRFOO2sx1KZ98SkYCdy5cgw3DTazU3rv89NQCvSzDgVyYsPdMtgUG/Yw3DPwnzh34WilzXV/3coqjvfUGFb9odboYXrhyCHfEmusSPl1yRDL5KLhdCQUq+Spkoz2ybD0AYMWGACWpEMBXpqd+ofiZ3Le3ng0891g83djNna4ssrarMMhDaax5CthcjnkRjcB2aHVe4+ax8Ml32pFq2bV+5wGTzOSzKcTsRL7qmHzywVJWh8o2aZ1xU9lijwkDvHvRdd/euBozR57Gm+vXe/OZZUubwoQ2sdXzOjfdoq2Nr1p5sY+elF9HHfasmdh+2gA++tLZxF1K+RvQNRgJu+6Tz4LMd9hXXjsXjXrY3OdTEslfU9Z2DaRrgg80Ls3THbLHNvjFew4uUke8Qv3AeSIxifTx6fd3yIFnKAVPilqSANd+O10PtNT1ssrkk24EmOvG+ORr68oeB6ZPdhzkBjL5qkX4AQsXxiFAot9358nn1c4lmztSZzAXCR9uXExeV9iADAgBze69bd5HMSaM03DWQw+Vo6fkKwlb9KBqZKsToU0qf0FAnLDZUkomwBxkMmPPMdO8sjyJyddPR0CNjjaLzMF++Al7hqGRMLZKzrB01qs7CJVbS7TTx7FeyFTw/dQ7bIuWEBhpRSodnEy+QslXtnVK+eTT0YXgEUEb9xgFmUv+jN2AE8LN9aO6bBkFqafA6E/oR/sCp788MnNJhCr5Oqfjq2OZsyF10Rydj9anOLPLSoi2xaEedeix6zaTkI+D+T8JRDtsmEryf8PHNp9PvswRuSK729F1lQO5QhHgahNue8njDcd/brYOMNYDfzoh/7OVuA9lWHt20a6OKa1UlTiJDHxvMyb34+9fOBr77jiVKIv2e6Xve8vAyY5a/jDw3TnAuiV+Od7nVu9nTL7n7ba1V7YqJTGYn7TrRPWizcVLQidnow+qaxBBfcoW5O7DiGvdhK0IRWGeRbIfUa2d5Lyque6oNZ1ZjqeCxPwVEpW1Cp98+RKaLoBXEU+2SowpjN/6wVZWlrvO2d22Y0AV6ETXHS3c1uiprvzUS4l8xZxmBNLwwBVdV34ml1/XHnrYEtFT8pUEdQr/278v9I4N10lO4gtZ2r+SjL56LT+11NGNw/rYxXqZE0hFLsNnTWxJmf8Ht0++sE17yDuoj6zDkTV1k9oNyn11qL5uzgVUBc6XMzODVlug2YrsKa6IE7U+tKbvHidXF+U6Ngw11+0oqsoo2/MiOs22cJVpIpQXZ8sUVJCjnesDUazJuO8prs3uX7wWP74qIgI1FzHRratA4LvMdOlXdcy5dBy73474y3+8OLYyzrujDfogKENdMdf1ywOAv/zHi/HnE44wmXwiLLqukEqrQd1c5JuniK6nZynYidKyrmtMPprxISouMkSUa7gGeIpC57gh2ogaI2jtEStrcL+gDqby8k1hY7LuuPU0YGgl8CDNyimDn1ydjo1Wn5sOtDxaorTtdYYdjVqAUp16p7pVidxXlcjglLxxXDsmRN/O9TvKRWbgDdKPHt2m8wk3SWpG2j9l1KGqDSVlsQ5QvW5GSlXBUmZHtLnAM4t3GBlRVUuZ9AD++nlr8dtM6cces6ZghoVNGWJ2ndXHFQjGCGYygZi0PfRQBj0lX1kQ44xNGSfj3b+81ZtGnoz6GzU0bYwkIYhBilrQFWm8Ef18lTNo3OUHRXXjJLPJwmTrj/b3J1Zp5WRKPpe20qfk00/FBfk3dW3OdZ/AGf3fw/ZYZaRzoQJf70HIn3CsT7Oam/1pPGjU0zrPW7Aaf7nXzyAg4fHJt/HNv7PfD0DMBmUszHUzXPuweSCRF1eSgeLN4wlGxMzWxUzAxXdH9q9xBOtRyT5mf1ctj5blNfvviH12ms4omFuXAs3GFGffUMwJRf4fBXr+Z207FdMn90Gdi9JDonNvewrn3BYecEs11+38WxWbqvNvOmN2mcknQzKBcx3ecGtCieAw4F0bZuFb73SyOofix65yjq9BLb1Z8nPq6ADBc35mokjU02DrVKSNPfN9h1QiJwZ3LEzbMcZc2Pf8CRIek08iYi5avSkqUFVmBlyUzQfVfcZC8UcpQeUrijJNb0fpfSmvLqDtTrrmMXcCiskHwVbyuYl8vHq6GHACws9GdMjM0I13bTL7MkY1J7V8rmAZhxIAS++1lp8g9Y+684xJRimkEtmBHbESQuj9kV7bm7InMvmihx786Cn5SqKbQ4A8+Q00alZG0pieOljMCNjRdZkLWsVc1zKh+syYMuiO4N3RdZnPwUollSmdjA2uTSNkDiSjyr3QRTd70xTZP3xBSNx5A7H26aIhKlDyZYv+mx63O2z3wskwq9InX8woYnmnFQbeiNoDVs3kk5MF9KqocTlSQbpFWnQklve0dpEUfTiUydeleWjVk8C6xc4ko3U3k2+l5FdSGQ/VSYdG1j87vmGH0Yff3LLAWZ4N9iiLEbLyTAKiLfC62t/MREMlxj8XLP7vqtAbqZbA/JYp0/2yvM6xeNl9kWMEUbFz3iH90BXQZQ70On01MdmcQlvuG+yWey9I/acF4oBdfHlCerf04MMbgE1q0CfbsBVzUObrL7ISSwj3wXCmaPnEOXcF1wNIA3rEIq2mUCMrl5iTnrWt6vrAbcJutkdb8U2Taf91Jp/8t6LlY9dzUp9nrUP45Gu1+f7W3Cw77trYvrZJYJIzODDMdbuw/tDr5XKbQCowHexhZAq3WuEnVZdBjcN9aGEm1tJKZAduHvy4WRO5P0quqXJGd9UncD30ME7oKfm6hCqIJLJCrL9Rs/oWE8KixjHYfaZcG6wyLSgzFArtX/Nq/OqdOuUqzHV5ExYpVxMbpdtwyKsSVmWoJ5/iq6K66phYdj/ww7mpWQ8AjFLmoWE1qCTaHzPwRllGxERn8oUh5kNw5eGNVWs3ab58EqT2ew9ezP84Y5V8UbkmKH64D/Cj/dK/SXMne1bepjkCP34e8NNDnUnatc73SNRhaKSpOIvPzYU8MCzwjv4KAOAvrUOxaPUmhgQTdd1ct8TYIc9te66+DlsnGwBkGzKdad69XiqvFJ5YsdHK/D3jxieZ8iLrUaItrYE3jB11MUaUOlxVDrIEbD3S9c3sud1U/PYDL1QvUky+znfcJoQpV24+yVISg/HWDfxwLvCd3VlJw5l8iXIocex+OxIpQKyhLUo+y4E3nd6sa8NhrsvCN3fCV1Z/LiyPBW4Cm7JqLeYIYq5QmVPa8yX039z2BoCj9t7OUVOAZpgJ+/5HK1v2u73TdEukXc+kVjDD6OfQ58wj9pyFr71+X7fMMSB1FCq6RPnttH6i8hPNk/vHq6nBkDYMF34p8+FLErBPbQHmDX4E7VbTKtsFp7muXu+euW4P/yAY7x3hFo8QvwChMmXR/Q6ffNZp+d4LVLmEqZANnDMsCr5JwBrpFdocm4+6diYfLZ+Qbbwj96lsJ5OzHP0USak6IZZWYE4cHDZn5piWp7TRqifSf5+4Lv13tDomH4Uz3nswTj7uQL8Ql7lurQ9JRcOn0yef7Uu963fA5rVE8gqZfIx++pO3P1/LFHPMHPI9mGnPu/0pHPDVy/HoMxukYhPg9l8A574zbSsOusjkm8iKwKOeux3NiggMvNH0OUXrIlIlH92PNmlBkoQAOUjryoridydtfxrUoBX43ctzHqUAiO0bcr8bbI5NBGZOX1ejbxY482Ye+9FQJMC96cqSl2HyFZtS/fRO69Od+23irVmVjL71i+M7c2XdYfogXvTsWVqGTMknfc9//CAAoAmVfW7KjuuJ/GEze1HMcYKa3yxo1MLHbpmpte1WA2YCmcmXX/Ir+WKgm+uGzIlpEIMh7DNqN38MgqNok61NBbcgxjVpLSWgrp9VVwr85/b6s7P4C+Tu2WSl002fP8ou21GPOzvm5FaSuJb3jPceguMPdSu29c+H8zT77RzG0LUTA6h5k58fkBjE2sBx6nWPO2VmyJiioXp9ZX63RNc1y5/IK7ceevCj4U/SgwvdHALkQWmgUXeaQ5GD6oallddJ5BsTdUsUc/Lx1KohnHb9E9JzUZuhMCVfVjcX3Oa63DICkZ3oVeh/x8dceNle2+Kah5ez5L56vx1w8xOqaVde025T1rMNSbYQJJl8YXAp+Y7cy3cC3IGTydcPoJoook4mn63tL/sc8OT1RHqJadI1p/sF9jUWjxUz+Rh975qHUubQdy57SL2x7un03/WZzzwf1cwVdduOSh15jwN+8Z6D6RukIszeNv69exfZZIl9KeM6kJGf0WTSa2ki+4eMI+r30TdK9iGZqUcpoGIkcpGUZNsbJcv756wMm52mBN+m3913Bb1ptCj5KgWh5OO0JrkedPjkG0lURZbBaop8Nu/4p99vN+l0iB8h6vXwnHp/IddTuamzy/pDXQ3HsGbrSby5LoUqLGsoyF0uSWD03fsXr82ZvIlMKM5dQGR5ixrGmuv65xuaycdRDgkhsHHY3k/z5x5eD3x1hl8gXYhxMMGqm/Zc3SCa6C0Xaq5bMKPpw5AECbDDfsCT1+UyZBKLa0xxyXZBSd7WmXyd73zi8S966KEUekq+krCNM1UMvPKA399wnFRSm7EE0Kf6kM1J6GDHVo5J7fLp8+7GrfPT4BPP23WGVqaQKlJu5NXrlm2CkkQYotl+6AyzHndydWL2mM04pIQgmyhZuZIEA40ahptUaPlqFhGbR+XTXKlWmR+5bIMyGmcKJyPGEbcBr0++apR80XV9+BLzWqctq4yu2/VMXVtZddrg6Xmpr6nXnexJnkRVZ0xUfLFttP1+wA77AnefrQusvExetMruoF2zL2UMHYNlTqH53ijSlniAbjy6rIMUupKh4gLnJIuxQGxfXJD6RlUBHHJ5TLaIlAGAT8nnbpDRdhuNOrHGcpjr+jB7pttPZFFGHAOWVEIIu5JvNOn3fPaRTL7QDC6GvD8zebUrPvlS+5JOqW7rj5ChgUprBt4YP22D63s2vjFNWXLsjwu/oIqi5o5fK+lUn3xy4XyzZW+AE8pfoOAF3hhutj0BFDv3Vj/plFOwtWlZenvG7Bu7SjShDr6se6YCxeGMKTNn8k3fzVou5x0lCYBNq4H7L/SmNepiOWjISSxZ2i38ALeHHnrmuiVR5RCgD/B6dF07Qplu/slEQHT9VMM4SVV+xDP55JdCTzLpxTrB5Ptc3zn2jEoROjtE3vQQZVIKTKNe1SJ7BKuJtNxOAN5+iGXSjVl4EHnOvGm+JXHG5GsDt54OLL7TTDIe7KqfOEx6JZ98ZRfk9VqCcz54KP7w0cOJu8znmLknsM8blXdVtj/F5XfkGtkILJpHZCln5klGF5SvZcrQzBy8arAWpCX7Y2wbNfqBN5wCTNm262Vyosp3C20Hk08f/oSA6qS+A6tPXIPJFw6qZZKS3pXkPiUSPahCuf4mj2m7Jctw9cBn8KO+k3HdxtcDj14J5YmqVvIpovnPUaYazZZAH6UoMph8ZkALG4o5yKeMkMvgM0rWbSY2qi4mX21Aea9GEZHuCthzrRDAmqfcDHm/kPyvtjTehB6UiUTNT0FmqhXHnXQeuQ1i1gPGWjSgM9sUKbHQ8yq6EcX1AJwKW9c72XpyfyFH8d1XyF8zuLN+Kcdoq40rHlgGAHhr/RpgyT1EKbTyqWZlBRfX5QNpJxoWX31M8F6zzjhV73I+v9CZxtwz2McyufjMesQWeOOOhasLk2nHw6ffniDXIIU5eAJc+FHgz//peBK5nlJdRjYo94rAG+pz9NDDlo6ekq8kukGVzqBH17XWIVXJeeVVVlNi/I5Z2Og5bL6LqlZ/+U5lsxq4UMVrlzdi3elHfJmkoqTidpeVAKrbpU7hIxuBSz5T8qQ/hddfi1eAR8FR73ffD0C9luDQOTNx4G5bxwuZ81LgLb+qrE7RsLX7/RcCPzsc+PnLgaFVeia7vOWqCS5pLkL23e6Ny2feNB+f+32xqZjQTD67QPPS9d/3p3GZPHrNJSOwaTVRELHwTxrW8Uq/evizZf+jMnNZZ75nArKIpXFv2uffKKmAeKeauJbvkbKE7ZC+g9fWb0kv3HWWknbjSAunX+9mtISA7kf2vpWtfcqM+c2Wjcmn97WMHW+2sVz8C3bfGqe/+yBe4aKQq8PV5fRgQ6kse19tQ/O7KbRkkf2bne3vPwN+tC+wlFLIhEN+33FMPsZ4xWAtAeXXg9yABtyyb31Sn2Orga4XLQ5LzPq73skhe2yT/60GSeO1Q+brDgC+23c6cOoRZqKnzQPFdluAY9ntPbDK+g5zHWiLQMsas5pui5Eq1zq+Ywn/nkA9pND75ht/elNRjuMQMUkA3HEmsNT0M5m1WS0BsGGZpz6azAzD64vr0u7Z3Nr2mHw9bNnoKflKwjYEVDI0WP1WhOcHwpQ2Ph/SZnV4sl1PYZVQ8Qa3iK5bnaN4r+WA41e3ygwyIXG+mWomOqsZSGau+9Qt9rxVlcWFw28QACMyWBk4WQghLIkJAUs9zn83sHp++ndTC6zCrLtIkrCOYLQdd4xyp/vyn+7HObc9lf8eE598Z70hMmOnbi/+pD/p1V9Tfwcy+XzmusFojQLfmU1cNzc9LnNdeSN15adegmmD9iAdMorXqjP5qnnOqnpNqiPSpVXXJ/etzTdlS2363csewrm3P4UqoI/bXIWlEAKPPbPBn9CC0bZAH6XkM7Q89iWznPKr/7oPnrXt1Kxy7sIjA298nYrCmbHksjnVIUBAn/fVth7ZjVCcEPAPf50E2fi//GFHSv635TXXzNC/VVT+RLJNdPlxntl6BpNXP2Rct8olrwnn77GEq1kUJl+SOPtuGlCM7hxyGUrMFN181VVRHzQTYSAz1/Vn9c9lnfsl14GsLtxU3ddYSXalylGF3PCIGiE9y74tzGA4tALTfg/IxgyZ1KGiliTAPeeRea2R0D1Q0mtMvtz3ZqTsHnqYqOgp+cqiC4NBJtLqt0JH2zStTRRJ9jKqQhXy7CatYQseQ2liVC5rHYfcwFFera0pl56Y1DIiXIw4wTQWStO6mHxJUomfMnndpIirMCIsVVYwhADu+q07TSCTb8WGYXzm/LtpUZWsKAiG23hsFDgdxbCfrD4qa1yThmW65YmVuPKBZWRZZLTGMqCCrITgsI8BX+FHqwRQuZIvmK1sU7QTSj7hGEPkjemk/s6mzOrHVv6daGmrn+xZn4vlumLlpqSvgslXSPxKn7ZZ7vIOiHpeX3Tdc297Cj+9tojOqEdm9bXJaLONPori4zLXdaAREghCKiM3G2OM3a/eb0dClmau2yLYfhKSBMC6JcBFJwBtd1obggNvMM3rfJAJvVMGHMoWS/18ZP1UD+FXOp207Hjsc9Gr3MKCwZ+7Xc/xqcZ5mD94XGDJDra2Psa7zHUt7Z76ENWUhfnNLkdoF669VFEnr5IvcEFsK5LF5NtqJ+XnsnWbLQnt4FY3G2d1VwCDSTo2nNH/XWsepbzOtb0fOTW/Nl/sKOVJnJVyjih5dF23DB3KOCUx+dL6KqKVvU8PPWzJ6Cn5SqKbZmGKn48KBpsEAgMYwVZn/yvqy0watI5yHoNsMqW/9VN7+ZRQsT8Km/inDjScStGMyUf55Cvgbm/Tz5O7rbysO/ftKIQx+Ry4+2zc0nwr+uBht8Ui0heQC1FMvnlnAH/5DHD/H/wbEcknH2dB/u1LH8IF8xbRoqoIEnLoR5WflXy5UUKMD6NbBTnhO+SoAm877RZ84Ne3k6Vc9emXdrXssvAqgIdWBW9kbBujXbeZ1CmzIjx8qXGpXeuzypcfwySgy5tNW4Eak48R6ZWDwlw3idpL2CJ6CiSlFX3u/lEwnPxpI0AcBvme5t6nNQX2Hz/MLu6J5Rtw/rxFWLKW2DxHKvloVqAFom2yRrPiQl+j7pMvY00/x1RC5XPkJZ8B7vwNad5IVMnAeG2DMwXJF1793LD2RvqNbD2lT7tGp8zSA9VYf4yFa5QM/9HgBSWQ4fL5ZgzxjjnCFfFYYfJFmOvGbodagmeum81lA40atp7c50jpOdhypmOa6zbUA8MP/Pp2tYwKPkDufDEtGfIISp9ntJV+J1OHaIb3rKHHgAcvtopxHRyoa/vI70YO7ifJM30J9pR8PWzZ6Cn5SsIeXbe8bP7enz/QHZA8jsaimzHp6i+4JTJEypuMsosU0xIvXskHAO86bLb1Hs8nX3dQpVLYV/usLNa7dC08b/gBGmhhBuLNofR6BOvgAjNEMfku/gRw2+nAhuX+tLVG0EfuSulmQjDK2GYOMGtPdl24iPJxZeThUJVKBt4g2ohu0u4s2BYTyoEJszSMnYh+/a/ke3FJsyn59tyONpfzwtb//vhB41IWeIPcSslKvpjm0HxUhIqwMvFKurhQiYYGDTGqLF+ZuewuuQdIWT7SbwbrXoB4r6ueYJd56X1L3RWSoTAE7XXqV5ROPlZQO3VFwCjeC53J1xxO/332UWZS0Gy1DMW46q6EMX8tux/4/b9HBdhIIPDAwHuB899rvZ8hM7eNdZew384zCrnU+ieByeSrSMFvlKX9fmZd+t4aaMLX/pVHt3aIMwJvON4x10JBSWcwJ6t+Nnt0XbnobC779pv2w51fOsYlsFx9WInc6yNbK1/a/984rn4VvxxKduS3tXlUrfOmuroGeMcdb9Pc8/APNzIFYl+9FtT+yrNoVgKZGIPJ10MPWzh6Sr6S6OZmTh4vB75OAAAgAElEQVSUXAo/XtiNFBmLjYouqMukr9P3ym5YcuHUjwAlADXm62WylHyM6MPq7wBYElfNnCzrk6/qvm19vopNNNptgYvuerqEAAZjMTCKbbAD9RB0acPN9nmkIILJx/XJF9AjaWU693nCnvvsWxca18bET183sfTe4O/S7+MqsA4h474juq68Ma3lijqqrpYKZoqSerUm2CkfLv7bVfkvYeNROZRTIPogvy/uN29871WNiXofHN2YXvZkC2XyGZdiq58z+TKffKrSz/D95jxfMm++8JtX4k0/uyn/Pe/Eozu+1ySc/x7g3vOAlY9lgtx1rjWw6g1n5z8nJ8Mpm96DrNlix1qfcizRGKvpNf+LkdPw18VmOVOwCY8NvgufqLvbYizVEaa5bvoSqOfkWigoyZhjfuzs2ha8w/0s8Ea9ZvmOHQFHKJD1FYKnoPUEorMp4vauPYVv9v2iU1RcL+G0M/Xu9ejEiyfvHTRPuJSLipIvRKb8ozUKDEzPf+bBmwotX1YRtvweepiI6Cn5SqIbY0Am0+W/Tb1DKN2ShDzNb2evnDHou5J0Y+iTi1PMda/4csXlZBu9eHNdp3xP0+bn44K46JIbWo7FBIjMPwYmJAqTT1Hilo+mK+OCeYtw96JA/2MyuPUpqUTNMNJy9EPOANMlPzZRgRRimHxl+5mNhaG3neuDec8llQ7m+sZzzNaKR7oZ2mHgvxchhLW/xDuz5pffrtmj6xrsk7RS6r+u+mURfidl0a/pcq5+aFnUZiqBe3zgfUEJ+XdRhl/KrKmFn1FuFMVugD6oczD5NGf6bz9kN5jKmcj5LOB9ykkV/35eGRW2pR5dN9eE1Y1+EbPvX7ZuGPMWpN/D+160B2ZOJRTfoYL3eAnak2b602komHyehJb6eJnqBJNv9+QZZu1cYnnfZ2Y58dbGtU55XTrjIyEP8SnT0b72qNfMPQgAQ7mlKnS6+zD6WKHdzf/K5jIvG/Gq/zGvESb9H3zJHmR21hLLx+Qbyw5AlU9c05l8okKXPKOt9Hkb9TBGufLe202gnh4MJih82mdr8UZPt9fDPwh6Sr6SsC/OE2DjispkOxcyIsasw8Pk865L25h8yw8wA6kD0ygrKOnvnTY/hn3FI3TCJXdFSE9BnQhli12nT77QwBsCeGb9Zm+EPz2qlC6jSoSYBvNShpzEBeSuWEm1ZpPpmD8IHCafBM4iy9WdNo2UVHLWbEEHynWoZquCDkm2Dc0GGFdUrIUL0S9aEWHuhuk7q781B9NBCHgvrs1KdivYVUEQk88eeIM+zAh4IZvXpP9O2trJLHvfGbfjD3fwGcRVHaKYrPJYVnYi/cVnuXdzP8RtIXme//QxzzH6juu9uRWzce+orzFOTD6bko/YZAuISnw929HdnXKuiInyaZsoY9a7HjkBv8GJagpprSY6rL6z+r/llexjn9KWMFQq3nNFudUIkCf3EeOeK/CG472oPvmkGzedJMnupHXU1Xp38Z3k5bYQqNf87eXtW04GRA1r3n+zcmnaIM00N5iRoWUBXTMh54Jirm7SmHxth5JvU31a0FzYbKWy+0OZfLq5bs30tZixBIv33tP29bBlo6fkKwnnGumab5aSLc8vrnK4Jj8Kky9u6swXxLutm4cpN34np4OX3bD89/wP4JyavMgqJy9R2k5tvKwN3LR9z+BOZD3sW1fj6P+9jlnDoNIsVWBSBjnlEx2seofqgvqzciVPqBNuA9TpLIlqFgCbmy5lDqOM57/TuFRFhM1KfPL53u3j1wC/eQNbPMlaptJRA+bq+U7JVaKSvfPJh4TnGZwOHPD24vfyh+LL197l6trW1g2Hi/UZr7BgZJy6PTD7CLYYl58x6yvb1FHyDU7vpLPXaykR+dD+GAWDICrwhkwW0wTo3z/LzJBbh4W3pN9tFyBAjzueM06l7pWZyv/f84Cz3+YpmUafzcyPFFOMkfqBUbiJeyf/moVp2FUtEIde44S8yoO/buGHgn6R8hqiw+SLDFwl97M91t+B50E9ZE6k8qqYT92IZ55yUk3DRswfPA7H1G4Lr5oEeZz3HdrYFGS6e6E83fplwCopQrbjQLzoL5anP+1IumyHTz4ZfgUyd+8gOlfow06vjm/TGjKavFJaF3V8nPFHfrbsL91cV8B+CPfkjEM7cnjImHypTz5enlruqKqD1qgSPC8Tk72PkODoPfQwkdFT8pWE26dJueZVouu6zHWFAD3amXnyoc7DEnGdbgsB1DoneFOxqVMSU9FIPMabateb6bpIQS+YJdkE3ManG+dhW6wuJTdbGIzVuZoQwHZYjWNrt5D3EymdD6SihJGmErDeNb9V+0NYFGMEF2Ni1pSSvr5e/CmzvHISARR+acKg5fnLp+yBTFYvCFCoAuFPpaV/+nY6GaANTOW/4ErMdXOfVgFIasBhJ/iTcZ5R+y6HkinWpC6F8OyZkwEA051RCqnyGcr/ua8H3vNnZxLFXNd4D657mYCOz8x6H3x9MMr3kadvhLLwhPY7/cVQ8rHKAbBmAfDXzwfJDoHahAmrDPl7qyWGkDisfhJYZFOKuFtLMdf1tY8QVnnhjyFlmPcriclnbrL9su0KcfmuF97BLwmRlqNs4A2/u5Mkb7/0DfEP0933zfpun8SvPTl9ZE6yBADw0cafSsmz3iOu941uKFwdOBLHsEmzpUlo8Dwh7Jteec+R9a1GFJOP35+dB/XNYeA7uwP3nu+UUVt2D6ssHV9r/BKn9P3QmSZ2GNWVfO2k7vguwo4aRjtMvnSM5eXqx6g6DLWbafA8QDHXzZCf0fR88vWwhcPurboHJuhBIEngMKNjSpZPp516C8vJN2FWww28wa1b6AKfmjR+0H8KlTK8Utw6dNqg1jHXfWHtIXy8cSHmJgsCZMShzJRhkKSEwO/6v4Fn1xbjys0HYhj9yn3f4kkWR0eXq3jzJv391oN2LX7EmCU6UJrJ1wVQ7bvdVgP4n9fti6P33q6M5K4tRFimJDqoD3zB38xrm9cC/7d/uHwmEgC4/ruBOaprx3FbGlbo+yaEYetSCH/h2L3xkudsiwN329qapgqE+OTTD5Fun78KZ91iBlAB0GmH4jtzvduQT4bbR3wihVDTxJrrKuz3EBYWOyXwyXPDXG8U/nMdaYR6X1bOyOh2xER5wx5kQtocNlbi1Ob/5sdXMiohPff6JYa5btYGC1ZuxHCzjRUb4l1bTJvkU9p32uDRKzzp4t5LwbrxtbUp38YYteWlvqldkmewSLjnbpplbl77Zf/3tXw8c916LcHnXvVc4GJ3ulZHtVWHf63lUjy1nAcmKrZ50l4p0ly3zQ9A9vMb0ujZoUq+ttMnX4FWWzfbDAAxB9vGHmcXHB1iFde4/TRvGqqc4xtXdsqx59PfcVvw5hb923KZ69oii9vQVAJv8NCPUSg1betMPrW+xZjSU/L1sGWjp+QrCTeTr5ySr781hDpaaKHuZvIFnDAKJnnTu7HQTtir1zNUsyCntu7tJKV59yUtQGQTANAHyRfbGJzgyIu4mNNMAWDnJPX76DJj5LTkWBxYyfP+tltJ7LXKzXW7+DBHfAaYvE1l4l657w4lJXRv4+pS3Fx8wovx0NJ1vPoYK8wEGFoVWSuin1Om5qFdYGoZRasJg8k3VovFpFbhx8zvWzZz3VoCDDTqOGrv7SOKZ4wLjMV4W9lQ0qPim09RfSgZ9egc2IVuSFxIpK1FWalGUAUi6Ja/PrLlgMBfpkzGJCHw8qFNJWtX4I93un0WCiHIjfWHGxfjT63D8ICYTeZTzXUB5f3u/zZgoYVNXCEUs3DFltrT9qObgMFOUoPllP7bbgt84Y/3hlVCiOK3tsn+ydUpS/iKB5YCe/nFUth352m8hCsfZct09dNRqMF1skOo2E+SFSDNkej5yWNeJV8suBF6H//mq9M/2Eo+/zjgapcQFx41B8lB7uf5mHzu8UY626v96/3L0rzO4HlEuYIRqAWFP+Ly5rqOVMLTnsy2rjKohQ9t4tmo/qnHknOZ64rOzMP9jEcUc10uk6+lBd5oqT75NDET0Biohx6i0OvKJeEcmKI3BGm+d137Iny/7xR/QcKkG9uQD9KRgTd0ToRuJOQDd+1bdtPj2lSvw1YAiuhlWQCOlvI5dPL/cD/g168zZMT66nYF3giF39yEL4ujhAhjd5jyrCfE4+2TTwjguu/x0j73WOCwjwXXiWpdVmuOo7nAqCPq79ydpuEtMhszA5cdMRquOHAbe/ivOTFj99AcTozba9MPlo76siUdQ1YAG8qm5Itzig/g1JcAd/0uLq8BO/uE113bipLEHemVvk730Y6SogIFsC+6Lme00dvmc9vNwie239afr+KDhmPmFgphWfLBtYfJ9HoACcN886j/5yxvnINTAqNDxRvTKpP9vOLBZXhyxUaGMJXTmUeK19pkydp0/N1568l2UVoz3vKEyiSc7mPyDa8DznuXO01elv8b0Ps1P/AGfd+nsNIDb+hSVnXWkd1ClX4Aw5h8KuRahLgjSBzmR7KY/P0tvoMtO0Ookq8thNXfmrw+zRiL9sAbvKB9+dhoabYojyh6cQyLMX3tfX7/V6LK4ir59G/LdTiWEU+4TTHarMBctzVaWNoRfbqWzfc9c90etnD0lHwlYRsDEqC0uS4AvKF+YyEvGOZpPlfJ54JAcXpUvQlMNilWJ1d/R2uS9AR6ZpIykRqdhU+TOm1auxB44trK6iLDYF8EPrMv8EbOWWGIJc11dVZByXc9ZIsi64jQVqThlxOs5Ft2P3DN13lp5W86YAEwHmuFsu/LpeSz76sYZYo4JV/XcMSnVX8IVSy8x2txqPsDmjwzXtZ9v1d+ujadNiVfcDtsXgvccz6w5G7gCrdyxijLcl2uWpRCTVHyufPLY3Ifmvh64xeYPLoivEwJcon/87p9iuudthVGKmBwdK1VBqecsr23jhZmIC6yc4xi2Ai8ISu7PRNg7Oeu1/KMm+bHCRrdZEjTq6z7uLJCXteJdvFbW4vmBD+nLDXxDy5XlaxTB2xKvk7GeWcAD1zkq3H0Wq/tU8To9bEWaxm7hlYDq56wSt0oJnnKjUeV3yJQrG3rDKWYPpa/7aGPS/f4ZVqVfEJduZaZKsN98gnWuqhUdF0kxkuzPaNzzc/9LiIsxg6uPUJeN9f8WpW4lmCU9YYNge8/M9dt1GvAMw+w8tSTljr3t5tk4I0i/Xif/PTQQzXoKflKwrlpKEGjTrTJOMS/Gg+e2Vr4lE5pfWpJtsUIrwGV49r+T0XLk7Fuc2qCSykq1mEqAJPJpyj5vO1d7SQQs9BRo0aa9fH2Gc8iYueE4QMoAL+6cb6lIkRfNBhI/Pb2++jR0Brmp6W+acZijGQ2ytmaI8C3dgXuvSC8fAJVsAD0/jNr6oD1XlEwl8nH8zcTizDlUpJl6kpduixaK0jrH3V6I84aX69nslthZ8UEf4sXfhT4wwc6P/h5s/5OVaOtDpTpPyGHXAFMPnl/fFTtDryzcRVetfB/yRzKRt7xqHLeyf20hxWZYTEI088afz4VOLJ2J0sZQNcwxTcbv8Bdgx9CQ3aBESGJNY4JLUhZAk2pVf2GLWx89ZTvGAujo+tmf2s++XS5bdmk1yhcr4t6YatBj7cfy9jjQsi6L/uunWP92kXAyAZnfltfHzz14DwwlCDqVpVymE5TSN8xiXVtUaAdwOTT18x7rCuCz9ijqBNsLtf2UupzrsApaVAEe0tzlXzLd30lgNQFCcedC58lSoB8HrOewmeBpcxRjnpwmHyeZuL2ZZrJR6QjBLpaMgkYUTM3MlM2zGfm6JQhF9BuKua6eh8rInb3mHw9bNnoKfm6iRI++fTJyznXECN4SujQmXySXM+o/74zb8OZNy9wFJV0/kvxCOxotEcwf/A4HF+/nLw/u7YsK4kpkcYF8xYBAB59xlzgZYuPrC0yJh9prhsBlwJQ93sEFBOMYmhDvJ9VG9XNm3IaGl5NtV6M3USQ2iRoD1StuS7V+jtOH8Rhc2bi/71mrnmzFbARlb/pgGf0tsfQitS86fITPYLGbsjee0fV59J7XzTbn4l8l9obESJ91ghwm3yiWVl4dZ+Lbgf++sXyBek++er9loTVKj50/42/ePdBABzRCW1Y+1Txd789mm+BzjzkKKYtdUlXdF2nAInJ53qii+9eLNWsOACj8iQQwa+B05oDkUq+JEnwqtqtOKP/e3hf/dKwiml4XccCIUxZaILbPHK71JIkj5wIAJi2s5FGBu1bdAyx1Y4VCpOVfO0iqFVueaHCbbJqrh9lTBnwKPlqXCUfZQzrR6aLch4k/HAf8rJA4lXyJWRkWOm+v4oTBtlbjmHyWWUKYH2S4L+3nYmhtnlQ2rQVpYmPjY4MmGQIEv/2Wyzf/dj85y4z/AzMrA3s85ejjZIaNreGcfKM6Wh36qcHesrg/v6kew6FecJYE1Y121NKPlLBq5vrOr6W0APp7N30DfMjUifQ+llLC7yhPUI33Xr30MNYoqfkKwmruW6CUhtyfTL28spYbCKJqu5RrMxb4B5A9YGZewI7tZ0uqD/W8JhxRJpwcBh2Wd0PraVU73qu5CtvXs0CpZRlZPvwWfM0MYUcqv2fWb85TcfZ4DHK7xoqdopELZwm9dVx9gcPxXO2J/zotEOUfNQ3Hdu+GuuCg4ojEbvQ1E7033TgLpg1tT9X4NBgMvmW0/613KBbkTY138Lw86OAm08qL0c/1Z80o7xMBZbNirYx3H3mFHzopXNw7ocOCxPfJyn2LAwcG2oWExt5/Eukq2yIdqHc93Qs2W+ab07MuQKRnTXLJoQaiTO27ycJsEOHNbRLwg9UQSswU/SjifmDx+FNteuj6sTZAAqoG7haAvUwpuYON3DJvUuj6qZjx+mD9A3f2H7s9933Q6Cs62QmX530i+UKrqRDz25lRK1Mg3qwmXzMw3C9L2TzfKzbz+zReX7dKAZT90z6qpadtZ1tjJTh6hPyWCog8Jvp03DJ1Cm4pnW/kba/z6YEVlekod5VZLCYfM96ufJz6ymWfil9p1kbWBWQTofiCX795O9xytbTsXiGJWJ7B87PT/6WXXtJh+/DXFTkGls//Kcjt0vza6ccrqI4zR8G0WkXV2AXb0HtprJW0mubt+hEOy3uoYdA9JR8JeEcAkqZ66rDjotpFRYQofMvxw+aBUKIfMEY6vg2oJTKJOmnxNlC5/B6quTrT1IlT1P+HLymrpF1kcSWXcgJobJFgFQRk2H+Cr5JJGcu69qillRcxZdFLmhczxei5FMWFvwFAN981JcuRDnBT0ohizCXYYfpg7j9xFe4I6VSba8zJUUb2LymXOV8KMUOqL6fs6tTVuGd1KD0oWcdxcu3ej5DNv0QP77qUbzxZzcp1+q1BJ9/1d6YuxMzAmeGRbf50xBwMgWkJs03bSHtLNrKs/Oj2Rd18+XgMpmse064zYlt0j/7Sjq0alV7m62T1C/fZ/vOYefhGwQWUHzyrV8CLGNEoi0LrQ9N6os8JByQD540BkznJ3v+8JrrZuuFxEjug67wIE0ZF94iJWAq+WoeRqAF2fzUiNQSZesEjqIo5RqGj80vq91lXGNZTSTVcq2zunOYfPrhngy5v7RF4fSnpo0wx8zdHttPszPmrJGoA8FS8un9Vitu1ezXGFnaZcx1kWBTxwWMSLLvj0on3Ew+ZV3sqEcJi7FQ+Mx1s6dxjyvU/MTv7bn+MMQntr5CEC3nuJOL7in5etjC0VPylYRzgqqSyecohu8fTiAfYCvZTBaTLH/jw2MvVbnR9o3TA0j997VEnAJHhzfqreXZ9PV5CLLaztnWNHFjBd4YT+5Txea67EPEJXcDS+8D2qN84RUE08nQ/aiO5QoYabVx6Jxtypf5xw9qSUQQI/GBlQ/gNX98DTYGOEMO6s3Fii4kV3dwz7lx+QY6yjTZXHfWc6yDn9GS574zrlwA/3vFI1i+XjXXitobLbo97FsEWItweSOlJ2fNM5JPPrsy0amCsprrcubufozg243TMBNr7Uo+TzvYnnOwUYxnsQdQVFqh/cv1naWPiSwmn9CUBs/cxyqrDKinaZUY0PX3p4tif05GA2ZaQnotymLc5L7v1Mt9FINoSPLjyzXXlTbb3ueUnm+0nTngjxu32wKYNtjABR86xJuWaiXON3Jk/e6ImlWP7PvjKPm46ychBNqWpj9iz1nO/Y887vl88rnwrkN3cd7PpMjydJ3w0DbPNXJk33KcuW7CSgaYDHi1CN66OCGVfHySSAiGYXP/oYJWXtJ9MNh/dDYeBa7FlTaQ2fmgxtuuL8576GFM0FPydQkJkmglX5KYDDmnEkaYc0mSJOQkwfXJZy8qP0bp/Df9/dLaPUwJ/gH9tL4fYMaKOyJq18H9f8Qra7eyk2c+jNTAG+48VUwByqYqYhKmAm+E+DeRn2FcD6xYi5mQkz4zrbyZzXHqS4BTXgRsDAgwwvYzpILfvtUsLqqQMtpqY6rP75JRMHeH4Hnnx52X/3nSnSdhwboFeGCA5kRRTbtiQ0AwlTGA9fWvXwrceVbx+48fKlcCM6qo0Y4hfikZiPKztJFvIhoCKrpu4Q+VIWB4PeQvyqaw41xTkR2QWYR28NrazXhb41p8rnG2dR0Qe7AkK0hU2eVGELOW3d00KYrcAEVsrBmbrw6+crkI/4zkk8I28sj1MUo+T+E1SgEitwGbyVdnPahe09GO07f+Eky+2bOmYO72ft+fIV4Dq1lKiXDlhwMFk686dx9CsoDQWydJHPsfLeBErLk1ALzheQx/llrE+YZWniAYXdl3QfZxwDPgEmw3S/osQCBdBvNdsQJvxI1B2ZNc2joYAHB+66VEGlO2MaxITdIo2QdbnbVjiJLPZPIVB3f6YVstqU4p2kMP442ekq8knGMBw1eCNaum5HNNhNbNhMaWSZS0JR1iZ3vKzu+v9p1ZSp6MY+rz/IlcOP89OKX/R9bbentlTL7mGPnk6xtaZlwTQuA3tyzAkrWb0t8MOW0hT1yZeUFxP7dMY8jiTGldO92iFD5lWBFa1l23mYRTj3+BPYPONHOBpPjHKswlZC9r/RLg6q9Hyasao602Gpwx7Ib/Bc5+e+cHkxniY/I951/MbEhI+dSC7NX7VenMvjyOnmsxcf7dW4GLPlZdQYq5bkC/bAzY7203F3jjz1PxXMZ21Bo5IhPjIM3F5GPhnnOAzAm/RYBL8Wcz1+VWpS4FhlKKVwT4mHwW2dLCIt43oL1P5L7AKhgj7ao7obHo+K0d4DrKiXZb4KlVm0rL0asTPA3KGW75KfCLV6R/W8bx9NuIY1laUoVmiDfX9QZHsKMhRtEW9sPw8UZiHTXi5QHlA+EoaI7m0vQ3UK/xSQ5lAm/oCjsamrmuXq2k0/+k/ppH17VSpx3tKK1t3OQMgdVDLiWfM2IUAKC/UUNCzN1VRYLOsAlpGU1CZaCWlf7tMkP+wOEc9qUdwS4MOrjqQWnfJUSuHNWZ4PVaMhZmNj30MCboKflKwjmIVxhd12muawvFTkxE3MAbLqTLwuIUZKzx7O2mlsqv13kwSZl8Wos7ZbhOxvgtUqTMJvuPnMVnMFIboNgF0/gy+aoNJqEvMP7zqOdg120mVyNc3pAE+QQx0yp7E7nO138vomKZGIGVmwKYiQ6MtgT6Gowp4qqvAg9fklWAIVkEjT98dwQpvvSauXSAlYASY/HvR+yBHaapDvjnf/tYHDpnJp1hwzPRZZFIkriPuWEJGgAAx3wN2P8tQa1SZuMWhJf8V+cPe3nuAIYx7zqQtZc4FFSM4rONuTswlMxks5tK6ZAVJFb9YQQGkmzzGqbkM791Xk1mTZE2ugHvNMRBvIki75wvXJL/beicAupTKIaZ67+n5wErH5fKsoyrSU3hsmTyyj0/AcVpG9P0XlHyuesj9+2Rju841hyloa+9GW0h0nfFcB1Bq40ngCLg0v8GrvOvF7I3XzratcyKbm2W7HrUDpoGv7GwjsGwIjnw3Vg444WWY70C/ZwtlscnnyAYYa0yPvlaI+b3S7o0EFg9NGJnorb972qgUQOmm0ozo9Zd7KpUC7nMkKfUqX0pH6IjO9Rcd+OI9J23W9b+Waxdemy+HrZ89JR8JeGMrjtl22i5IZMxudhojQLNzUa6sua6WXbd7EnGgckjvuylcMSesyqQUmCww+Trq9CUgQOq7TZ1JiIOvd5nrlv4mK+Gy9e1ffviOysVZ/p1CsSs59jv1eNYB1TbqaawllpuMFmfLlz0+EU48rwj8TBr9evGaKuNvnqCV8zd3lBckWiOgMfkk8zIAsDtfu990exg2WkB5Tr4rV84Cl88di5u+QIz2EUVOOpLwH89gWJgjpzSXUy+CJljouR7/jtZ0YOVKOSlq2Vj8lGKP58kkdfJlbZQ8tHvIX08WclHlWWRLTG8ZGZETDCvDOr8L6zyDqvdj283ToM+Zsjvi2uyqL5Xft0NxsmTN7DmI1e98nZsDvPN4G0M0dxiwlLe6S8HfnKgUjNakIvJx0Mwk6/FV/IJxmvW2zwLvEH6BgScFe4XwymZR2PyudZKXPP7KpAqt5RTQDrh308BrvEz/zPLoFgl31/6P9+phfx+R/I66q/Px5D0+p8+9gekGa2OAdYyR1fyaf2FIGPkkZsjlXwFsvGPgsDmkRYm2dZqjIPQgUYtL2NYxLmSccH0YcsDpePLvp9Goq39kjB1ea5YDliXmErPwiefXnqq2J0ACvweeqgAPSVfSTgHvRJO+vWT7+BN08kvBK7+mnJJMQEoyZ4qJndzMHxWbbE1H2PZ4i+75PirtqTIffKpviLchcRH1zXfoxppjC9LSGeiuTmGwszwsRFDyw3Y+IU8yLwz/PIC2pu1cXnsKvs94mQ0R4Rp0dBIE7/425PG9a0GpUWZrc7KgtGPW5ekviif6K+XOoe89N4lWL5+GP31Gk5/10E8xdWmVeR2pXAAACAASURBVLwPIzDwhprXn4Td97JFYkWmGVGn/mXLnrQ1MEViCdZ4PvmMhnQx+aKUfMFZwkHUSxAbI3nDETyPEm3IV+hlrCybuS7v3ctKPlURV/z99Gq3qajVJ5+FyVcGc2sL8r+zulNMvrP7v4G3Na6NNuXNoJtbcfDTax/Dh38zz5wrznwNcNqRpb7LvEm/vh3w85dHyYgu3pbRYlXC88kXUBn5++NGrY9cJzdzJl+4SWVfe7hg8jHWwpRSV/+myvZjl2xZORczz4QE3qCwj/RN52iNWJl8daeSjxFwiDnn9HOSSacoCUxz3YLJV9QpCzBsNQV3faCixe4Jo22BPlvgGLlfTtuZTDLQqOftrLAjK1ZSJdo+g7onw1yDF89Y1ief6DAcSz2j5JMPUJX7ad8V42ze1EMP1aCn5CsJ5zhQZqGoTcYK1dgsyDyBXFWYcFw1eRL+YzuV/VbbsAylTivySdOUMSQYrJ9xhFznBCL3yVeTNUljcJBDMi46HSr00DyTtef2cabMnOlsbM1TBPCaH2L1AeGBCPR9C3k6f9Yb7QLqDlYToeTztcvJ1zxGXleYfLHm8/+hsk6K/hO/QFmydhM+8ts7MNxsoy/EqTn7GUT88xKIftLt96msDgCtXDx6b4svvqqQ98dsQI70yTfFxY4Ob+GxcVwtleEoTw3KoMPTRkY/tUfKtYMfCmKbKWb0Qtknnw0PLFnrrIutfgpTRfkzhMmnMfEkQTXH5pCCi4XtktD2UoPUOnz3sodx2f1L7Uqur/oZojYofX/J3cCfPm5P/P4rmDI9CS77fHpw4jDXPXnGJvxtshqQqNXma0hZbhNkWWwmX5ySLzPXtfqNlQ+StEOMZtKX+jROEiWdqyn0Pqy/En3NbpXDGBr1sl5Wuyv/2+onjiGvbOANuX0Glt2ZR9fVa1SvJe6DPO/naicRyOiTX/3q+cDd51hkSWOSbq6bUIE3OocT1rb2kQCycc/9HK2WsCtt5W95xq54bIdjsVlj6w00apKSz94vqlq5++aWLMCIEMLo51mqhjCV/yHzTTZkhygvDPm6kk+6FcXe7KGHCYqekq8kbJuZAFUNmVc/cVuwcqM9g6uY/q3wn9tvi2umTO5MdUXi5yWP2/M5IPtqpp5+IxxKki4jVK8qT//KRFDGZ6HL5MNTwZD5RT+5e/GzZ+HwZ9Gb9afXbMJl9y2114uxcBzzqe+g92HjnFcBCPuS9FPE4K/QFRVQiq7Ljbe3eZTuS+8+fPfiR2x/22aO8jOrUxkVmlzfhu2UmQKXUiOqVfJFd8zsXRqr0fhxW8aj33gVTnMFfKHK5kBWQutK56TGkmmk2HoPR+JsmcCv63gx+SiojGWNx+17185+qh4W6Sj8q9Eo5p0krxfVbA058AZZC3V7Rysh3XXgXuegrSj5/KyLsodHAo4AGi94b6cQ+okYbq9Y6Jf8wp3wsmerN+ffYM+46yHqb+OAilmBW34KLLzZFJAhqeGayaM4ZZsh5XKIS75gc102k68Yw/z9rpCfm+tymFC7H57+e9D7AAD3b/VitAUMJp/tETmHZlUy+QD1u/h5/w+KciJ2bS4mVgiy3IfWHsDOV3xEWmeo7ZMkYHde0vIiMeKhkqjLB/OnH0VHp9e+fTO6bsU++cBbcyZCYLTtCG4mK0mTOoYGZqKtbdn7GzVgo+mD2Th4KWkxwD6k6fzbyliyBAxz3cDZpp335cjBe3QzsPYpaf2g+rQvzHV7yr4etnz0lHwl4RwGSmxka5p9otvMyHHGqvkskhciM5O1emovstxZdaiFjYvJNx4MaH1jpzP5it8ab+DhS6uvi7IsMtsue888XYmQFm/AHrOmWNO+4eQb8eGz7FGLOa8l27DN3XEaIzUTXJ9FHFFtgZOufhSfveCecoKcSr5w1oFtsfPKfaUIsBUpveS+HruskxeEVqfQZMY2s1QRba5bKZvUeM/lBid9bOur17pzKtw3qfg73yBnpy6RU7qr/01Un3xkGWb/UHzyhZaht0uSsDfMvrJIxSCRqSYF3pDvK3976+L/bhR/rhUx+XJfsQHyFFIe840Jfe7O8NofOfOF+KQDoCjDs2f78VWPYqSZvqP+eg3/cdSeYTIJ6LViVbPdsibUv+4smTvwhqYQZ1QhKvDGXq82yuLUZzQz17UGLpDmmIxVODgdGzEJIkk666dE0fSGKENMZh8v7ySxCVPgM6+344TkAlY5qrxqFZCzE/XAWH8Dqa9Dyxwv1K91MuVYjzt/5ONzAgytyOW7ZBk6NcKUvaM/dkTX9bVnsS6Xf+totlzmunJ03ZpB0gA6TL5rv9kpQR53qdq4ass77uGy2NvCVKbmbFKP8t+7ZMoCbwT48FHq+Pv3p25wkkQ5kCrKz+ipPSVfD1s+ekq+snDq3oqB5b31MIWRTv13DXzOCXxwupJOTjsNQ1QOJuwbHZdZkSKh2nUHW6jKerBsAGsNt9+2SGy9/lHn/RCdQFsA/Z1TsQTmyVkReAN4Zv1w5+/ieQWR1oUsyUGzt9aCRpRA07LYjegbl9+/FN+/nAj6EiqrbprM5YiY+FkKj4o+hozJV0aanDfYXJelnW4HKTWzLUHlw0WtWkfVY2OiCk3Jp21QEq5PPh2OtDHmYQmAm34C/GDvwEwhhfD6pqzHMIsIY/JRpqiAew62baBsnvoynNr3v5g/eFxuYtdEzcoe1g+uuFAVhb668SBvnOoWJl/mB7dsWUA6pyldPSS6bsg38rqTc/cI8jv93yuKOSdIHgN5FFxN7sGztyZSC+u4OiRohZtR30M+qMqD/acX3AO8Z8cFKhrtfNhWtrmsZDrgbalfsxe8p3Mv7Sa1mprOzuQzoZfKNde9YvNxuH/w/U7pLh7bCbVwJV9hNl8O2foxk1d865oizeGTLyX8p+acF33sRdhuq0Fg1K30pOqQ/uiUIQ9knrWFweRLtIMyFNFh6xwFnI4Xf4rd0K22w1z3iWuLv7Xo2Bnk9Vk3tlN58fm/fHam8kqkujfgNtf1raP0PhiMRy5L/x0t9r+y2rleQxo0yeLHtIcetiT0lHwlYV10JxnlN8U76mEKI1PJZx/4ZPNZA1LUxATq3LNVUkbJl5duXOFMBGV8hrlw0BpTmaqXZFPsKfVuDAIbl1vLcfrdtVx/VvI0Xn/r25Tyvtk4HUc++k2pEglufXIVLr7bHryELkewlAxWv9wsJZ9wylDTMjG62Z8mLdUvKsT2yIXQ4BqeYqn3kqCt2orFMPn2OtZaVrvE5yUvosN98rEoqPZTfgfsZ82RD1siMBJZj5hqxCgFZOWW4ZMvkSoSIJvJ5AvyrXb5icB6/zjWbSg++YIDb5jtQjH5QvzgKeJz300pPtb+LeYPHof9kifwL/XbARQmaW3UyD72yXPv9nKuuulP1fS9VFz4fONssvx/q18j1c0O7joh25Sf8s4DYev3lKR2yJzBMIUPZgZ2cOqqy3DMrjsZ1zNxOsPs1OMPMoU4FoIbWqovvkwxduZ7NXNhxcF/0vlvyDgiM/n4LH2f7zK1RimGR9M5ZKBuGcdlJt+M3YBPPQBsPbu4LQQRXZdTMo1YpUNcbNG4MnQLoVBkubP9SfZb/yzqNTjZ+kIAk/rqOGDXjqWRRcmXQBjvRPlkZSafLNwBXW9Hmes2MyVfzKR+9JdNMiGRTAiB0ZbF7/GlnwOu/HLx27JWkS0FKAZ11fAy+XKffFLb7f5iJX1N88mnj/G+Fr/36TWsdFb0pRZPw5vX4F923REP1leo5rpJAqx8HJj5rNgSeuhhwqCn5CsJbuCN0NN13Sef2+zLxcJQX7FMcY6NtAVIiwZyQ8EDy5Ezlc8xic8YtSvmKMjsxgQCw6Kzab78i8ADF0bVz4YdklXGteMa12C/pX/If9cS4K2n3oxPn3+3ku57f33YyKubodkUwXI725gGHEWJlfVYBqMeRXPAIssaCS0Uk2d6kyxctxCH/fUNuK/fwfrrgFrDXdj/JeB/JDZGjJLv7b8zLmXvcVgyKwyF3EXCfPIxmXwQYea6AY/xjTfsy09sNcuO3UiNEWQn8jobMakBg52N07NfodxaN7JO+qWzdPxzSMjBTDdZBUUhvFK+delD1nveb8TxXSaWv3XZIy2B+YRPXcoq6L3tdC64sP//5dcyn3xNwVNK03XxQzXXjUdbFAPekfW7O/LUduauPVijiSg2/sfM3cEih36isHOh+EM0Eo2CkXvlhruwpNHA5tbmjhxVkG5WO9hHTCqO8Xe4o3DrE4W85+06A4c9ayaUVh7YKv9zdM7RqnjX21i7CFhwkyrLZa77ym+rvzkHlNLfl923BF//y4MAgEn9lu9CM3fUpbUFzMAblmdsw5xPdXPB0CAzGXZJzDWrbnWTIXbNXrW/wO2nputkW+ANF5Mva2UlT2vETGbzo9np45855jnFu5Pf72X/ba13AmGy8/L5zSzD7v+Q2Z7C+ENBsy3oNdbff6b+Hkhd5BgRnS2fTVVKvnyOcsizHZ7k/hLnvFS5V7cc8OYKZNkEicB9i9d1yo3cvw6nbqoWjazF0kYDFw48rjxdvZ4AQyuBqdvFye+hhwmEnpKvJDzE4hJytQWFK60QeAnmYf7gcdhd85WhszDKm8YUZQI2Jd+YbPG8OL5+OXndxt5LwK87R0F5YPJIzsrw1SNDiC8ruQakuS6Rx+qDJ2BXF3O4aVXMNofp6xGwmT0EK5O32tGb5PpF1wMA/jzV7gcxA/VOD6hpfaJin3w/njk5WsZoq2ivIJ98ALrJ5PvqrCb+Pslt1rPfztOd9xUYQSvKqenGzFx39ouKv6ds2/kjG5hrwORtgE/eb2ykX3R2kY+MNmdF+HOx9e0PXwZc/InIcoi+RowzDy5Zp15Y81RAERSTL0XN4181w8LVm3J3CZpwa54RFMrbbFPUtgTe0MunzQb936XafePZRVROfZ0gO5CvYr2QbcpDHP4Dgea1xvddst6fMV1LNLVxMTfX5WgjHQGN2llgl46YtrCYCUruXZq6kk+qwrH7aXPkTw4CfvUqNZEruu6hH7HfY+Bn1xZB4+SgJwoUJV+hCMwUviILDsCYiygXNOYaPW4O/37fqeR16rvoI0wdOahqTS5E6gdu7+3VdY9+SOz0ydeRo8yXAea6rdwnm6RIlJV8t/3czCSVZUTXJUbVlo/J99TfPbXktXezLVCXNYmtUWAVsV8Y2ArU/Ci3uzO6bmVbMt5+ryWExJhU61XXmHxJkqbIWdC5IYL7e4rq07K7ouZmonaddy5a6Jnr9vCPgJ6SryR+Ki02ZBzWmgest0czdeHzr9ob/TX9xMYxgEPgWKQR3A7QI+ZqJ5hn938jqk4y9k2ewHOXp34NaDMlF7qrAJSlf63vDADU3l0/ke38y+CqvOfw2f46dMS/op4Gujiidq83T4YgMpq08E8gDLYn1dI2cyJOsWUWitZFRsum5Asvy+rAOBSDUmCRSZTvozDIi9mTjns+nagqJV8FfDJ5QxnUpiFMvshV59+mmObdql+xgPrqpq4lMSYRZQFgynbAic8Ax50P7HqwVonOwnT6LkC9UGLalOw/eMsBnQSO/hegkD3zfYfge2/eHzMm+xmuAICz/w2YdwZbvlov9ZiDjR8VbE9vLyQCb3DhS5lIxw+62E0o2i9T2jU1c13XYRVVFoUpFt+qZQJv6BEgqTT6AVV+Xaj3+IE3OrIsDCKblCBzXUk5tDT48IPAoD2AlV4rfd4mI3I6vuGsHbN2aLb8Sr6sHrnJsHT95HccqObLfevKTL5whZS/32WKAEa/UKKTmu2VRtfVmHyW4lvwb/hj2XJcX34A0B+p5KsSAkWdC2NZTclXA0NJI6FJu23R3/KtT67KfWDWa7KSjz8u62/ynk3z8ZpddsSoxCZ0RtcdMVnZOrKR3W2GLtBstdEnl3HpZ4EfE2vEgalkOfIw4Op9sVZTGVzBpvoSs08KIa2HEijvpw6Lua6m47P1n+KQLWKOUkzzO/5iBXDhnU/n12u1zphQsTuXHnoYD7BWKkmSvDJJkoeTJHksSZLPEfffkyTJ8iRJ7ur8/wPVV3XLwvdHvw5c+62wTMMbgJ8fjdfvtBaXfPxFyi1XKHfnwkhZ3FSjYPvzwIl43tLz/WVT1ZFrE1mdKtWEz06KwZ3DdNx/l3QR7PTJJwRmTR3A83dJF/G2QCS0gjREU1AsTrV5NL+W1TXrPgojINBXVTkTLhuTjzDTiITVXDm0w8gnePu9Jf33sBOAd//ZSHrVlEleP0zyp/uKudvj6k+/1Ew0gZR8o5KvwP5GwEJHtIHbf8lLFx1d130tiExn8b2YRA5M7LZffCewen4nU8T7Gpia+lp9zjFEJeixpqlvuD0HHwrkvJ622Wn6IN5y0K7ONJWB+GaCX51Xp2BnLviYfIU5n/sdU3dl5dYHG38BkM0jtCybws9VBgDsuR29eSwDMmIhQxFIy5Idyyf4Zd93MX/wOCNdzsoCHMols12CfOh1vtWzHzobr91tOzxdMzf7R+61rXGtCsjT9u0nHm1hr9mZfLprgrYQNEtpoFA83rHqfuy/x26Y336GX1Euk8+AqoQkRUt3WQcqwqfky5h8/rm3JWretWHWxynF9Ctrt2KvZKG3nAwJBDk1xDL5QhSJbohOVOJMyacyRDPsu/N052AsIHAAHgHWdtbfJJPPbIC3nnozTrs+ZbrVFNYufx7V/RL+fOXlWNDXh2ckVz9Z3yDXxfP/5i2DW6tmSzPXffwaOmG/RclnWT8YhyoCmIYNntr4QZU2iQii1HYw+XSffBmydnjHC3dPL1TM5DvzfYeoCn1pnTs0UlzPmXyhvrl76GECwrvSSpKkDuBkAK8CMBfA25MkmUskPVcI8bzO/wm+9D8TeJRmAwtuBBbdljpcbdNmG/YSixOj/2qcI2WUzWLKg3Mi5AptrjiHja2QY/FAKgG0i/KG4+z+b0inVP73xDpAFsDcnabh0D1SFhg32jBXfp5WqJtMF9szD8hgmTdjmXwnHktHzzSdsVtgZfLp8Pc7kt3AyqmhRkQonbEbsMcRRtKljQZubNl9fgG6r6sEc7alFmyBtfwwvdCswmRUVgRPHQxY6Ig2cNvpjHQIYod1DZWb6zITnnYk8H8Zgy5CoeiKCmxR8o20aWV6EaPDsaDuKE04NSUPo7oSRh3qsxbH/6XF3vL5o3DpJzrfutEuCfGXG/boutLfxnhp5nErxorn/kDjEqIsC4M7SbDvztPyv6m6+aCnpequb6xjAm8lEHh5/S7jeuqTT5r/KOWvpTyWGaxUA6Bw1bA6Ueeufz1gJ5zyzhcEyDNhW1/J9Zw1dYBMAyGAJXfTtxpqHiWqp/x9SoyhG1fcBgB4rLW0k4zJ0s7weFiwOQA4vnEFKx3LrYmDyZdA8snHCLzBMdd1+eQ7pf9H+OuAwY9IyyT6pu3p4s11q4HokPBrOYMu/UdunefvNiONmGs9yEuDaZyVnFiw1iLctih+/zz9IbH83alN53pxp+mKevu7t3rrpn8rsqRlu706/3u0bQm8oWPKLEOOUabj7iubV+OewQ9i72SBvywCrnInwXx3ipJPeze6uW5+vZMu31NYmXyd9xWwrkiQBnpRImlnTD7t6eq1JN0oWdZSPfSwJYHTiw8B8JgQ4gkhxAiAcwC8rrvV2rIRG2kvB+FbxR1dV10afqzxJ6ngqqb3omoyXM+6X/IEGS7dJosLV1sKxuPKSdRFE3/r4ap6zg7oLHLsmzyK/cGsQFqQIsvwydcRJjP5rOa6rHYT8g8AwNwdp7GCHSxbZ4mi61ncdSsKsxOMyf3/s/fd8XYUdfvP7Cm3JDc3vSf0ThCkRQUEARVUQBAriBXwRRFEFLGhgAr6WngtoPIKvoKKIEhvoQSkS5MaYhIgIb3cJPfmlnN2fn/szu536s7uORclv/N8PpB7dmdnZndnpzzzfL9f+sVt4G5/MvS9WBUILpKlTfEz1z4amDzLmFSasBT8wI69+MHk765cJJ9veQ7FSQbU/YOeTUOYvzLdoU7at09dusxO+v/j4dphtviRGcgi0z1IPh80RPLlHauk9M3rJyZ3t2OnKbGiyTu6rqF6WRtG0CNHulBDSTbXpX+TdF1M74+sJB/SPiNLDegLnz7bFgmyqFlZYnoJ5GqvRZR8vUORgq+Ny+PE1NEdaK94Kp+3PcRyIlaDKdXyqicPgQd/bj6lknzc8q1WOkiaqO2LxbeowehOxyZDg4T+7OB5r3T5lXzpBaKGkSINXoE36g5/mEkRBdVy9u9Fr4vJNNIHzQy8waGb6wIMi1ZF38UTr6yLE5qfh/SMxbhkMdd1zbTl4B5+/b/JWieETvaEoWXTfMCghtv7s84StSza44AOXCHbXeiaYswrkNq1uU8FgH3DaHNke5bDH60BpnGhnaUbiOJsPaTfaCpAAcxKPuG6ImAkyGTGHDGvexQ94rM5/8SfZMtct4XNAD4k3zQAtGdYHB9TcQxj7GnG2NWMsdfJXuc/E8WXG3SBnnZA49GD7Yb8IgRqZUuERbMk+ylsk4dt2WLc0PYNfJWqCpuEcf12sweHiJBATZQu2lTFgZa/x9sNefwe4klmzcOfi0CewBuMEJSMua8VhF9j0XUJcs4Z97tAMUPgHJhzLrDqJfMFBRYLtoVQ7qyMg7v9+WRN/yWFjO0duSY0qkLRQULS/Newxk2hR+VV8nml47nMdbmjfzv24gfSRQUA1r8OWL/UHK2P4qhfSaZpzUCeb7chlFwkn0XJpzwPPfCG4yMJ6VLODeMzaNQUXSW5m4qMzsHhk8934ZzZ/xiemWkxxb16aUsRjqJNUX7z5a0umj3GSMeClD4vH7KRgyPkxLwxB8lXL9A0RQTcsBHi5IP/lyu5n+LQsflZinw8ikdUD0MzuUAidwsT/yBeJoj3ogU3oi+s6OaNZ9tLCHYvJZ8ruq7Y/GSKuseSVQ4lXzNga/dFffK5rGvygCMiRwXJJ3JlAAbVj2mdfZ6uqUIN32wSIMX1XMW7y9iYdRFgoipi7Hp4wWqs6R1E2fR9/HAb/dh7fuQsWyuTRPMdqnPFAsVyr51jjYdpFeV7lJH61Sw6gthVqu3Q51qS+wRNyWcJDsQVYjVDyZdnPRuZv/t96yXhk68VeKOFzQDN0qPeAGBLzvluAO4AcLkpEWPsRMbYY4yxx1au1MPGby7IH4zCANLB3dz2NXxj6RccaR2dd4OS40+Xbsai9o9azQRsSr4JLApTPitYmJk+L7Zf/3frOVPuvkujZi3Rq+EmfG7V+cCGpQBSUw+f/L129QSoTz2E2sRX/BI7ZIDsaJw+qzxKPiYfNKd15HfeUbtGPsnu+xFwy5nZBXsiV6REF6TBPZsUdbUvdTJrV/I56q5ONlwkH6nL/45cZM/TEzbH/Eb4LvCev15WWUzezbsI9fHNWy7vrO/yx72BH++YHa1vGCZwrxfH51by6ZUIeWhV8o3cuBBYcK9VARRlIPp+D4WWsRkrBwc2AH/5JNC7OjO/qFgbMW4oTDn2yuo+vzJsUNq0bbHoVJd7KLm9xyhPH0z6dfZ+zLQWa6Qp+/jba7ZCO4rWGf/IYfr36pqM9jHrg8Au75cOBXH/u+P6e6Tjub5/q/9YMwFh5Pgeuli52N7/hrFPvpTksxDypF+sx310Ket9vpIqv3MH29jzE6LgzKQcLHke1tRhmJJ7Pj75FHWP7SsyzytlFCX5zOa6Zv/QFRT1Zdskko/HqlmuK/m0uY3wO2vKREWBIC1/fWJx2uZzrHOYUlFB1gdgGKqH+NCvH8Jf/rFYC2QHwKE4jPG5B6WfxoAP5Bur1UO/4GZBRSgH5Pw9Ox35/dvbQrb6XIeJ5LvvpVVYudHcDweWNhxypVvM8nWds03rQXZEP6GmQ0vJ18JmA5+ecQkAqsybHh9LwDlfzTkXX/RvARgdk3DOf80534tzvteECcPjoPg/AU0ZUElnNJGtcyQEaC+lOdiVfPLlr9cXytcCAEZgk1KSPc9Ies2s5wVcffiyUgmPt5l9z5THb2vPMwdZ5fo7Cy7/NAcP3I3ZffcAz98AwL7oabidKAobbU5CfosJvV3Jlw1RX+57gQFnHbYjjpu9BVBSom9mkS4eBJ4tUmJuEzA6uG8XBzdQophy6ZuzP4zv3vgcfnjbi8nvYko+leTLVmwCQL0J/VB73sAbPnjgImDh3PT3p+8ARjRnPAiEYi2T5GPILUfNwOvF8eX1yfexmz6G9177XmPyd971XuD3R+gn3v/r9O8cC7CxIw1RddV28Y/LgWf/Ctz3336Zeizg5oav4owJ47TjVz++2HldZrfy2O/kqoBudmQtiMznLxrTjYNmTMvdXlxKvqJ1CRiS/kTeuPD/NvTIuXmVfP55mxD5CSMqEMu3b8rrK9c8nfx99uE76hdN2F5rf5X4+xu0bKp5IbNNK+pI09h261fl38uf8y4+DDkSV2CvPZGeEGPNrseglpB80Z3tPDVSPn/qbVvJma0hG7l5yJpzeoD3/cw/PYFVNX3eRODi/aIIqL/YJz2uPG/OOV5asRGvreuXyEBhDTAKelCV7MAbBc11LZNW01GX+xsXmkXyCXQPvCb9Nm5jO9qCVpunzBY/rm/q/XtMJySfI6Gap0qkJ7UJJMVsrg33z9wFfOgKYNLOcZ5KmdKPUnyM+5vrBmU1l+iwpOSj5dneN7OqxE1ISXV7+2lDGmBHpFuybhO5Ru0nLWtGrsyPmxx4Qw2yw235CzKwpeRrYTOAD8n3KIDtGGNbMcaqAD4M4HqagDE2hfw8AoCfY43NFKbuMvcgm8MxfdRBRqVWmHIdmdx0MXnXOs9ueiKddziUldL7sG0OvG/6FJwwdZLx3Dt2md5Q3rYotM4acw7843JUatkRqtShI1XyZbeBPIETAim6rj3wBl0AWc1+cpKj8nH/Om8SUaxU4kol/RKzHPm3C7Z7ayi6jfE3pQAAIABJREFU7vbvAr65Cpi6hzW5i+T73d8X+ZXpmtCoC0LH5ENeqDcOcxRHC4qaZVbajaazl3Z3YW5HOx547YH8edYySD76dJokwSsW9KTA4iunku+Z1c/kL5fmM357azI6HszeeixGmpSfmslr3KbUMc72oVoJkTT9xfUncPvIEVqKtoz2m9knz70w+nf/M+ISbUSYXb2vjrO/Gd2NVeWSWRntgGu8LtyCeR0VHi3SZCVfcVIgd3dLrvDx+WuC5JPPqLYxP6GeTekC9cQDDKZ4hrYnSL6BhvoNX+Vm9O+98zwsXxY/aj3FlbZWC8PUTLBvFSkwiMa7o3+b+OQT49vYzioCBhy040Q5c0rm9K7CcIJp8wIF4RCw4lnglYeUC+l7ZFgVq4yefHWdvPCP/7yxerZSrr0uye+CJrE+KkEBm3rpojHduGFkp7WMZpkSi/5ekHwh2SDQ3KUoJN+C6akiVuvqn7vOUJr7Gzl0p0nI65MPUN7T6c9J3watlxZ9ev6d9kyn7wnslG6kqRvLAaNKvrQtegfesKrKiij5iqcznbe1raRmSRuJn7MlmJVk4gs4zHX96ipfw3UlH8xKPibKbin5WtgMkNm7cM5rAD4P4DZE5N1VnPNnGWPfZYwJCcCpjLFnGWNPATgVwCeGq8JvBJh29LI7zxBJb7PwXmDty4XKblOk0w+QoAC/rMi7pudUfp+Zn6TeMp03TGyoRsZ231ndc78lUiqQf8c0r78XI155CLjhVOz29HkA3PVXiVCbj6JGqYW6pORzU7binD26bnZtbIvSPEo5znlkTtWrLFrKBgWQsTQ7cjlRd0FteyWHcgrZPvm84FTyKeV7+uQr7sErhdcENEEDz1/5SPuG+vDTsWNwymR5Qel9R0MZZjVNVjYA+R1B4+mrgI3L8xfk8snXNJCbmbgjOSo/t2dfW5+dldq2xeSZHu9dBax6EUbkMddVUPVsv5nvrnMclvUuw0NtlJzza0M+qfyi6zKFiPNX3k0ZZVbFj772OFy9qrE4ampNfcx1QxK0wlV3ac1nU5sAsk++LBUvQbavO71Mq5Ivz/efoeimTfvpxetw/3wP8sy0MVwdCZzTo89JOMzmiCyIxrsg0Mx1Obh5I4OSOff+ILueJng9vDRNtv9T5b0qC3bpPsikSMwhZgby/GT8yKpO6qlETs5+4a3BM5iM1XhrSVdg2sx1bXn/ZnQ3zp4wHnuyF/GuQCd7i6oMVQhLkHLil1LUCxhSffKFNWBMqvpcP2LL5G9pzrj82UJ1KZUYUfL5kl2Qm0b3NIQkyiqdQ0oKu/l3An84xrtuasTeMg2QkWzSctTq3Oz7r9QGbHNw+tuysRdI4wH926wl5HDPoWznXMSa6dikUW04Zo9pJAX5dq3mukSrvmktMOgWVBQy16V9pI1EZIiVfK3oui288eG1WuCc3wzgZuXYt8jfXwPwteZW7f8fzGDLcV/b6cCV8YH6IPC3/8qRQ7InIUmnAeCkekoWbh0szV03VRekrqtshKbNXLcpopkGHbnbJmvWyVn76MiPFIDq4NrM/FVSr879BwuXGbCKWo0E3oD72SaBB0n+koNzcu3PR3fjppGduGWx3F6ao3cC8OfjgJdul4+VzIvQPHhu6Qbj8dx0jodMny6wg2ZMBlxtetLOQA9xYm0pb+7iufi/51KH7kEBHuuaf8jmjfmUfM0jzsKMXdxMZC30HYE/iqqYciv5/uqKyueAS8mn4LnV/iZ8Egz3YiJZvF65TclH38HPdgcGzd8v+gmR2DUV6BgNrHgO5i9bPubbfsulAIM1xzfIOY687kj0jR+BL8dDAF1kFFHvM3CcW/5fHPLUUjxyyF8y68gVki9PWR/YYwouuks/Xl2UHpQ3CIojb+ANFUUi7CZBFIBUybf1gbnz0SCZj0X1KsWEUUNKPksfbrr3RP2eBccYouZbDzmMrsACg08+ESiAW8jwHEGUbFg5sAb1IMB42y6kgmyOT5XnyM9butxgrqui2yMAlZ1IM+d5ZfV7uKP+5sx8KbJu+5q27wAAtuy/UjrerLmbeDyVMPrG0s18hlrdoOQjgVxoJtJjfuFm+fx7fuxVl3LAUoI2cw5G52vqJnz0OwCT3NlIJJ8jiIgPSqRMquRb3TuIMt2ISiJk1KNxLkYdHK9iI3ZU2tKbe+9L/nZHew+N6VSo5/beeBfmM3PQj+Qag9CjHiJ1B8CYVKOA9lMz3wrEjr6k/uWCLT3qmEfJB93/puVbb0XXbWFzQouqHgbkXSSeWLqpsQJJZ+WKvtU8Y4IUtsUNt52nJFOh+gDgIX7yoTcZT715xpjMy99Ru9943DoRYizZreZxpFzX4lZdDNeHySdfvS6b69pIBo50197HJ98lY7qxuKKr18y7eNR1u3zcVheN4AMM5rr58MySHlw0xxypd9+t3JMUDR6DO100NWUq4FTyKQsMy4T2lDmnyMlyVqFWD3HGX56SjmWZO0poiHwntd3rUw3kEyPLXLdiWHzAj6D4t8Pkk+9jV0dBAhSz5/MfPr9YGb7KCIuyTIKN5KMBEmwEHxCZ4Al86A/AvidZk6o9FCX5HvvGIYYLoisqWVI+HqKvFrm7WF+OzVvdVxAVvD3l8eU7MWWjrmQx9bWhpa/1AfNQ8sg55zOFovD5hmyRINVSRyPbPQbnXFfytXUDx/3VWU8/0AWqeJ+2MTSXlM875UjfCOcOsk0saMV6PPIFFn8bdDyhQQHi/FjseiXklnvM44evVAV2OFw7fMyDn8dBW/i4YYnvI6t/yiD5JH9cISX5PKogslR+FzOJtVl55FPyZZfSHCWfeKxljeSLTMAl1IdkCw2qZnOXkl7iSFkKiJIvx/dks0Rh4NJQJalFcwTzAWjzizcGSFalOAjW3HkrMFgL8fRig891Hkrf4iUv/QVnsYfwUjXtCxhCnLz8HPI7u21EWjm78EI9d9LK83Fb9Sxnnia3FSHnKAW0daSQyPADv5qMBZLLBQ+ovhWzoEXShlBwKvkCLZ98LWw2eD3sfv6/Q97ouov45MYK5KkIu43pkY6aAZ8Igekx4CuH7QTcRVVyId4XPIQAs5M0Ppjb0Y4DNimmdzxEtWTugCeN0hfvallzOjvw++4uXLZ0hXIPLuYuGhy4BwGkThR9zJeKQCX5VND7FoOnLTiFjwrJtlOYz1zXckI111UTZgzoy3rs5pnbT+ryqBkBKwEHfcNNPBAEzXi/LoJMVaVZ3tXIykhsHMpeFNtgahqSue6LtwD3/Rj41G26STPQsMIWQBSdbtLO4J7P3oosc91qF7DzkcA93wd2Pqqxsl5vmJR8M/eN/lNQDYqS500kO20L7if/4Hf9ifdEGwPjt498Hy1/xpyvAdRcV/OxRBCpKRyKJNK2/zp1MbBAXji7+l99CWS/JhNWJV/GZTl8/DYK7mOuK5F89udweuUarzJDGnij1g+0j5I2awq7CGYsddcgNvq4IPv0pLnyNcA0Eyk7XJdIMJJ8wkxPNh+s08AbErNBlHzQzXWNDS0PyfdND9+CFtBnkmler20sUIKJQbIs5dnqHjATGWC2CFHnyq727XLlUkQdLDCDLcd+QeqLtVm9uVA6VupyID4G+Cv5fBFX2vZKypTkyxNdV92UEN8zh13JlxVV11pWlEei5Nv9OGyqRGT2E6+sA1DGs6+txw6TurDXlmOAl4WSL5Tu6em1kSuLFfG6hzGgrLRxJv3tUuvJCDIU3G1siKzj3MSzeHz1kKfkQuKTLy5PfG9nLgBGjAMQuY0KuaV/sdQ/D6ke+eSDrOQTE151esLQUvK1sNmgRfINA0ydj8sp7wo+2nrOB1Q551LyFcs7gq32xnsFx8Su9uRvAPhg6V5cUPkNbu9JfVD5mKaeMnki/rlQkcrzsAGzX47TJk2I/9LrbYWY4BN/Gtakym9b4A1jlKuMR/LB0t24vb4X1qELvf0yoWt9JkQGb1Xy5Rhcmw9mN0H0fNG5IqFlISgBbz/TmUT2h9WEsnMF3jBPaEe3jZZIvrz1Mvmnkp7rXz4ZKeRq/UDV4OS7EXNd8Z6ZvChV4T2Vz1LyVUcAE3eKIjw2iIuP2xMN6JLdaO8G+pU65vDJ11bUDN7VmRB4+cG0Kfl8MeVN0X+ZdYPWBstEQmH0QRajYrRdpPmmi4OhQFcAuBffacoeT8LGaBrt/J4zFPce76lo4A09um428gT9ysLq3kGs3jiYEj9DmxojFyQQki9W8+RxqWHP1u/7AnL4mg2H9GOiP1XyqHNLVE+Dki/ZxOIWci0PydckZCp+7jxH/q30OWIe9PtP7QMMpZYdViWfxzuw+eRzKWSHSzl+VfVcTGFrSN2apOSL/y3x6J2HyXtgqGm723WgbBh/uPo4let2NEeCVxFIJJ/7Ocp9m1qdtHw6B5J85eVU8qn3lBDq7zofuPsKAPKYc9vpB0R//JRcJJFMUdpTp4wH1gkSWG2Ttu2k9LdJyRdY+n2T+tO2mbWJMawmG2oh52BB2jZgKsMwFuZS8llGmg2f/ju6Ln2bnp4xeS5idQnDIjPwlpKvhc0ALXPdYUCzw9VngQ6Yqk++RqFN4NVdD+O9cm33Zjyiher+628SKYqDh/ad3AG3I3gGnix4VDG5fXhhyQ4Q9+j4VauFPEo+lypuG7YEF1Z+g4sqPwcALFlLCR3itNYAodSzORr3GVqbEV3XeH/ltsYIIrgX8LlhiPSqgt5Hw4E3BvuAOefaz6s7ihaSpNqgyfPZ1/7TnSBrQt0Uc90sssV+ageq2MzyyeeIGJu3d3r3rpPx7l2nZCcsgq0P1I/l8MlXsQSNyVbfmnUkKoZU9YaxMCVNwzvkWVtPKdrKaVkmUkOY7GQuLgxt262WCDPV1VKJnko7Wy0z/f95fJvWuuWEz3gnk3xp3ect35B7KDj24gdxyzPLojFusA/oWWw1x7fhtEO2M59gLCUM44V+GLcZfZOwGdDnW/4kn8NcVyWXQm5u82RsSQNvxFYA3DLHaNAn39Mrn859jfiUrZ/tyufl3/F99Q314W1bjcVjI6JAJqM7K4q5ruubdpEqdiKNXldR1MI2km87tjhTNeXCGMhK+KYp+eL5o6oMZkgDb/z5xNlx4iHF17L5m8fDl6R/f+YuoGsyucJ+v+VC5roc1d7XlCM8+ZeS4dKcMieRzZU/SuIP0mAzg0JJZvSKOpSxwsStTkKb3wttq67ZGQPHFyeNx2EzpiXHQpvPT5C2I9ZSLMojVKPrZsDYNvb6tDVtSY2u+65zRfFq4uh9t5R8LWwGaJF8w4C8A2qjAzDt7Fwkn2m4nILV3nlHebgnOsl1Sth0kaqD9znL8wIxT9bwwEWmyiR/BmR5oQdydyn5/Ek+NZc80XVd83nxbiewiDBVHb+rE15GVFHJspjkT9+ln5LPpFDNvo7CeH+lNiRPbYSIpppvpWeMUFYUk3bOlbwhkm/lPOB7U4DXHrenUduc1dQr61t149onlrgTZBEFRUi+cduqmUT/9/AdqZ2jJ7NIvhHjjId5VLj72tcTJv97pmMWNF/JJ0OLqGhCo0o+FRN2iP6dvndmUnobLnPdTCLFEBDAFnjjqOB+LGw/DjNY5A6CElo0F9qfHvm3WehEfzJW2Bb4UnAMhzpFvzabiMkdOMaCRnzyHf3LBwqXGzAAl74TWPIYUO7Ide0XD7aRfEHqL7beBCVfuR044CuZyaRx2rc4R5+nBd7gJKon7SMCneQTJADnlm7BpCDMgTtevsM7bfpd5GyrcZ+zuj+a7z7UHf1bLQdJ/xRyZu0HfEqzmQ/Sb7msWNrYFK0/rf4yMy8X1HybpeQTzycQSr6kXmngjY5qPF8Ja2YlH4ipJAD0kcjR4+l36H7qpRxKPprX9LmylYYIvMHBZXNdqYPNR/io31sqaku/r0pWlg6S713sEXygNFdObvkbAByGZDi19BfsHLysXVf2GDOiazge7Ij72/h9REphc92TwBsJiUY2EWzvcaRO/Grfwjk9wHt/DFu7kXzyHXs5+OTdLOkQpWtF121hM0CrFQ8D8iv5Gl1UpkNKG3OQfIa+78H2LzhzztLX2PQe6fEmLphPuDH6N6wbd3yqHipGlx8H53vjMsnnmnirSoYwMdfNxgP/spOuoWL2Gyg+oVz5BxlKvjyTZi//fTaxl+kg9ce31f7e9aDII/N3Yov9cl/SkLnu83/LTvOmD3tl1RQzMoJz3qeQnWJyppXD5PO+KFWBUx6Js5Cfoc+9bByQF0zvmUXUdAV96LgwVA/xuT/8I/l9/OwtMPfMg5pejgSqzhSL8RwqJZtPvuwxKj/JV7Zt3TdC8r354/qxGfsApz4BvPkEU2HyL0mZYS+muyODOM2h5DuiFBFV27PFWo3o3+r13ejN6MUVLQynxzOUfB5qK3mRWLwv8THFHQ4zxYAxYHmsRjZ8I+4NAscMRxAVtcg9hoj8rc6npCdm67++sRx4x9cdNVHrlYPk61lsOCgIOjFniFCv81SpZA1CJC+mOcg4u+Rx4OWYkK0Pjx9oFbRd5d7Ti/ucPzwn+wGtloJkAzcEy7W/o1bBruRLoSr5XN9KXt/ernybZV0UPR+emuuSeonAG4kPSQvJx8HtjTqHSr0cBClps2Gp1zUmEQQdI+j8WFLy5VR1pVS0OjdK30TmnJX6FFVO/bLyE5xX+Z10TDa1NT9f09HPB9dgr2Cedl2FENJun3w0/+h8GNLNb3njKjED1oLhxN+1uqF24NnGe7G1obtXPIx7OyxzJDEOKu9zOluJRe0fxaHBY9E7C1s++VrYPNAi+YYFxWX2jaLZ5roCyURPuY0AISZirZZWJwcbm9TPr1Rwwau3xEqb0DhAHlu613itunhJlHxMEJJcSydnwFKffB4dv82CzccnnwtiYSQmkyWV5HM84ub45POvby6+qSM7InLzl4Q25P9OS43Uzqe4bd6hXPP69CXHzd5CKVe0N6V8pk5kPcFK+kQqvrfQY8H0v/cvlM6dchBRBWYp+Qrg+aXrccszy5LfB+80ETPHGXwTNhOUAI+VRKj4l9mZI60ES4eg9gHU2foPP2COeC61lz8fB1xrj46r4Yj/MR8fu7VXp0X3NExKPuG6oVrO6NcNbTsg8gj6XMRGkonI8v1CrIGuyGFKsGaSfDmVPI345POBb+CNPJCemKbkK9hHS+a6IqKoub6SCqyg6wKRAyUevM11HcGKTEq+sYPLgJ4lRpVqXAmpTpG5bozfHAT87rA4s+GZc0p49w+knyIglPdQGBMKV75wZfQzvi5SgwmSL3Dkx6GOe5pvMw8l39nlK6RzeYOy+Sv51Ouag5BziaikZJbwyZds9tSVwBtSgAdLm1NIPvfGNXJ/ZyYRBP026PuXrGkb3UQWGRNiy0jySbtAuk8+K3Z5f0bbyCYA1XNnlv8MIFK4uq6VAm/E/0ZKYXHPsgQhIcPJ/UXmuvFmS534P+yeCRz4VcgPJv7b8u6//sxP8PnJE6VjDMBgvZ6SwqyUbNYwDuzCFgGI1pBJ4I2WT74WNgO0SL5hgGmwL6wg80JqkOk2180/UGUvHoD3lR5QjvFkELdFHMuLk6ZMwh8W3oDlpRJsgTd8TBLocJNrt1PsADERac6R1FCmKf+iJF9KoNKBW1eUJfcpBk84out6lN+MxZhxEj1zdnpij+OAUdOA3T/mXbP+oTo+8puHGq4bgCjiak74muteffJb9IM+k1SNWG4+ydfTp/cbug8zMbmyLSHy1kvWNUlnLKutIZKsQ7F3kXbeXQvP0TOlnyv6VmD1JrfbAkBXwTY12IsNJj+LFX9TxLHtYwsWrN8bN3S6gzHRdO5Ru2Jyt2X3nLbx528oWB8zamENvUO91vPyos0ui7L1i2lGJiVc9qI+SkV07YymM+XoVvbsf8W2CVEwUPNf4Pr45JPLGm7IC8e2cuNTUUmNl9MnnxOe5rpSEyq6EWO4zpvkM+Ggs+NsFZIv5PjCP48GfrKzpW3rrXvTYF0PrgAUJjSTy7Pu74tPA7M/Jx3qrCpjYn0IOKfbnoeiGmLJXIql5rqwm+v6wMcn34fL90jn8ipafVOr5GERc90jp03B/jOnScdCLiu86JggmkZCXoW1NGiNQpjMeX45jCAkn2m8oWC5ZK7xNYY+OzXXDaUxvkSl37kJnzSf66rfBO78dpxPek9dbdG9nnjA1pbK2s11NQQVTcxA0YXeuFZ64A2pSPL3ceU5AIAaAqcQgh4Tm7Mhpz751HWJqqZL1cYMkDdo6SJGQ753P1jj6YZGUCLkLsMQorqUUY/ulYctJV8LmwVaJN8wIC8Z4j/MZ+frMlm1XT0NK73L1pf3HEMw7L5xmdhqnJYgBJdFyeezK0onO6qA3s9cN9ucgHN1UIvyLRWYaEn5FlTycQBL1m1K/k7rSerIWOZkOyEXMxaqLhhVEF1T05p1zwS+9BwwZgvv/B9/eW12Ih98fRmwz4m5Lws8u9FdpxkWIV4LpOFfcn/iskf0Un0jQBZV8o2YaDgo+gsbyZceHzfSEWjEVpdKJ/A5eVPi4L8cjAOvOjAu1/6s1fWty8db09BuiLyeQ50XFPUr43lvQsm371YOMrFBEsCFb/z9G5h95Wzrebpo16LrEWgK56FNwFUnpL8zzHVNCyzRliSSL0PB5mp/5bi/P7EcBa8alJR8bvgo+YqO0WrZPvMfWcnXuFAGUMyxmxVdl3Ngl/dHRF+88RRa2lCmkm/7w7KLM7aJgjinB5h9spSveMzShoXFlFu9zz89+io2DRnSNhh4g0K61+4Zytm0kYxoU+ZhQxm+npV7keYyd0QETIjA4ZNP/zIZOLZhqR9b10a+tVoZpL52zOVcTcpXzSs/FlQrWFeSiQ7OuUzykXNioyTZ+wqHItLunecDJ98n5bOuzxKtlnzEK7Epu9fKObaY/MzRdy775COJbB2Uh3/c3YN/kXyCJK9SXMDxksUEeaKG6LpWlCrWfvfptk9jH54GVXPlZCZBs+YQ6TUhj4KXcA50DcbrSibLDxKffEZzXSb7p94gyOBGNxc4dprSJSn56HqnRki+5DtuKfla2AzQIvmGAebdjsaVfNboptzPJ58Nd7TZHUGbyvxnuGXydwCOQcgDnazkGwZYlHx1L5KvyJSZkcAb2WYiKhkgqqrupjb6bGTH7/YKXft4OhFtlpIvyy9frqAPnUXVRhHoLU3sKhhoAIgUUgVWm76dqDlrj+ekEjWWxpc30AbFE6+s809s9cmXo/yDvgF84kaShfxwbAvpGUPpxOvl1Y6FnW0C2DUZaOsyn8uASoI3NaKzDbseox/LoeSzPUenJ+4oAbDPScCh33WmEn6YnKrGAiTfQ+1tWFHKnmTftOAmuSilLO0u1brECbTF/Qs3Ac9dZ7wu7c/NzzAg20cqsp4EV/6lEE77h3j0XKaMSomsrMV/4BG9l7bv19Nct1mQNv6Mi+8C/SMPow2nb64EJu6ERT2L8OTKJ+PclD6LDkSmNj9uG/9i6d/kvTz7nXeJwrzzUvMAFFLbEjk0bYsZz63A972oZxF6BqIAYnQu4fuGKqr/z6w6qMpusY5nADatASCUfJ4VQPR1z2k7k/w2q51c38dWbJn0exNj6HXMQYpaUzTLJD7kQJWQfPWOdO4miONkLKgNRH5k3/p5YNIuaSacZ9ZnQc8CnFF+BL8fnbGh5dn2hCrQtNEum+umf7+yhijKTOXse3K0IW0qTxE5pEgJr1B9Xg5kvr2gJAWfexMhFkex9D7yKPkEaihhQ6mOh9rbjNfScTAET9rBcU8dp+QdPxMeSmSnKDeJrjtA3A7UdTI4ueqFm633YQKj0XWDIN344EiEKlVWQ4mFSZoWWnijo9WKhwHN9L8n+0Mwgw5MJUdEJFutOpllV42Umaqm5aktA8cg19VtwjyoWea6knSfh9LEsA2DKKFuJPkYQhzXexkmIZrImaLrpvfoeG8v3Q4A2HLBlfjvijnymYBt2tFou3AH3tCJN/Fz8dqUDLHVwIfbEj6oGgo0YapADtLCBLpAv+yT++C/j30Tzjtq14byHA6YfbAUIPma2L8Ug0PJN9EjMvFORwBvPzNRa8pZx/2FlchM8bM5Lzmq2Hz1mLoAbFqwFxdKBrIiR+TQuoXcyaw5Y8DhFwJv+yIAYFnvMvQxDvXdJws717Mo8C4+O2USPjJ1Uu7rtKLjdjRSKH9sKiz15arqJIMi0ubofL/Ss1FRog6IFj9Xdo2UzM3NSj57kCahoBHjwHiyoZHpViO3uW7xPsbnWlXR2NCYEkMm+VSCuGD+ynO77NnLsq8Z7IPRBNZCpsnlGS4jxxIFW0ZeT7RVExItyjYloITSJj1pmzO6KGeaLP/3/b7r3ocP3fgh7bgxJ0oGCJ+tcZXevv2E+MKMOigbiamSj5ibIgDnHOPRAx3ZbdrHXFfFHsF86ffBM6Zh9pYzpDoWwXAF3gipku99P0M4UZB3PI28y1jkj29wI9BBlehRnR5esDrTfHjpxiiQxuPtqVr/xWUGn5M5VaQmJR8l+Wiw+FUbybrI1MZHTQNGmqwRAH3lJH6m40gYt1nrPGJTDguVoJLUcV/2PC6r/tCaNK/gpI4AP52+Dp+dMgmm74BeE4LrqniTTz6ikhNru8Qnn+mdKnnOYMuBp/+kJbNtaiZ1JEo+CrGGraCGUkvJ18JmhBbJNwxoppLPx0E1A09MRIsp1Vxwd+pmc910p85u8JcvklmaNxCZ66bHXmz/BP5YPQ8h15vzHmw+PtB3FX5c+VV8fZiIWDj5v6i3EX2rgPl3Jj+PKd3v3N0OtTDGgpRTr8n3AETqwJif6T3FbYI8LLs5SvaUctgCb7AgvcDXRNRyplxiOGbP6Thu9ha44/QDcPeXD8xRkeZiQ7+sHjBP5nxIPlW1YCHAlONd9bWRn6I8k8UYms8jqaDQXAfOgeqI7MzHb284mGwjxP8336P3ctK68PRZOllQdAlyAAAgAElEQVTKVpV8eVZhd3/f7TPKBkpWfOTPwCHn5Nphtir5MiHf3KFXH4ozJuoqcbHYbraSDwBWlP0jLVqLjut3wxfiqNmvPGhMpy1MQuVe9z9Du2Y7YqpnLBupue5zo1fg++PH4opRo5LzJsWPqzcQDu99FOsqxj5uCWBCy89BWocAZm01E5d2dzWs5GsWVS5VP0eUTjf8xp0Eg33A96YA35uqn8sZ4Ti5zOgHz72Z+/Gpk/HZ2z9rzN9XreatDPdQiWL63sAex0uHlmyMvh86brlKpNOqkHNUSgyXnrBXfNLRx5zTYx2X6POuI0DIgcfaZf9/QhWa5VNZzMcOLD1pLSMLG0i0h6wgBy4MK8nHYpKv3J74sxPngHi+2R8Tpe36mHfp/QsLrVOu/ser+sGcY4uk5NsiGhOSe+Ch3Sejcb5jL9v67TAG0SLqyfPKKMcHZDNwRrDCmTTLBkdFDSWsq9gts1SSL4tvZ9wcuZYjdjmU8U4ZOD65t5lcrWf1sULRq2ye1om5btJGWj75WtgM0CL5hgH5I9nZQSfDPgPjcEXxpb71VDPRQYNPPl3J12A9FSWfSpjsE7yImqE5iw5bTEwCQkCGSRZuQjIv9MAbSMpuLN8gzieU/o3KsGsl6bOS3AaRv/OIkiSffIzlutb4BBw7ZlnOlwGZfKFkw3aTurDVeA/SqUHY3uqfHpEnpcY78YkCK3Z/9/6Ms0R9Yhm3j7WLnNmrASWADGXWBVsCF+1BKxhXIPRTqxx4ln7M01yXe/oksk4UM83M7efV55TLXPfeH2SnMYGaHU6eBex3eq7LC5N8hue0xCAqFASE87G+TtGggVTpI5CqS+IDc85VrojHAvUx0XY8fW+jiff/VChxZlLlpQ9lMIgWH+vlFZ2UPmAhXKNQWSH5pP7bc2z5fOlanFX+o/FcZgAEAvF0fj5G9xlZZByVhveC7UUqt6gvShXK9/Pimhfd6V2+4Tz6RnHv9BH830Mv58pLXPr8mufT5OI+uH8gj6QuRZR87zxf/v3Rq4Ajf55ZZkhfoqOenAOd1TLKghQr2M9JsVpQNz+bqXvox2Ag+eKxaRpb7UzXCIqSfM3a/OecqOGCcuIegSNVwZUYA/pj9x8Gko8xnlkf06YzDYQxbXSsZvd87yI3Scm35ye0dKa5kLUck8peJBfuirSFT3oPPEsFn0dJFpSNFj7GbJ2CEx1ZVljSJ0sUnbZcI3NdA8nH47WKkaiT89x5yihDGqDGM/rYkLZdfd1XQT01120p+VrYDNAi+YYBXyhfpx0rSiBlOeqO0qTRdZ1rrQLlm30wyGaiqpIPoD43mjS5oHfGzR59TA5id4t9U6RBK2h9hme30xZ4Q/fJl688cddiMllS3oN1rqDsgBeFTZ2ZJ0ujKqHBBdnzS9cnf5dfDz9pUBeiFnWkaqlhqtrAesNB4NH2NtwwojO98Jwe4B3fjH7vcnS+ymbgwtte0I5lElhrF6Z/U3NdH5LPMTlGxsLSX8ln29G139fjyx/HQTOnoM8SvIgGOgDIQiMPanbXCEZQRVIB8+BmKfkE6sp7EWoEs5JPtIvXj+RTIbqbZNFoeYYh5/jAntPx0vlxcATqw6s6Uk4c5zkAt9P1dAQkmyyOV5i18BUbVXmUfL2M4a8jRyQ5f7nyF5xczo5wnGn+a/nbF/L1fs48slIF8u5TgVqZCpW/n2dWP0PqY4BrLLP0SZtq7o2eG59emlmvBDu+11gv2STRlyhqYENFVVJKUVhdCihHP0KQ+O9KDniMO4YcaZtZiy59fhQHh2Lc55swn983eN54PLuOprm377uz57UVM7QnT3AQgo7JtQmpMk1sXlJXLEzMX91+4eTyUtC5XbJ520jgDUWtzeGYH9Pj7zwfeNtpwF6fzixPIysN0YitKnhCamZ+iyTwhlgXzOnswJKyEjgls76mzSr3eXqMIzSa60bpSHot2nVEDkZKPrcaj4FbrSjs5roiQfzOg7L0TMXarIJaEuCqpeRrYXNAi+QbBny8fId2zLl74lCn+JB81MzHVU4xkk/Ol3N5ohEgNEbXTZV8ouwGJ90On3wCpsXPNytXAEBiymvyyZcUkaM67sAb8kmv6L1esJOSxuhv8QGbks+U1l26f/1zreuDEmytc17vy7hupFuNd+GtqbrCdwHTKHwWQOrkzWgO17taPwbgU1Mm4eyJ4+WDHaOBs14BDv62uU55TE0Ibv6nPun3cQadIk67dlFuh/BaHjGsSj7f7AqQW7948hfoKQV4NdhoPD9YS/M8eMeJmDSqQATP6z6XnYZCIkTTZ9Rf65d8btlg88mXCQtZsaIs5+f0yadGXU4i5bnRzC842a3PaM71kKMcMFSEMoiqCdoUki/Oqw/t6iG5bBM5IWWjKIEQOsdJ1Vx3TM+zODK431o+AJw7fiy+PWEcnmozRaJ2KTrcb0GcrSkP9qvv3rHAONecNy6r+/UnUmj8zehHtD7X1dAspmT3vnpvml/yb5Z6znK+VDWTfES1YiYyGDBhJ/kadw1SmPr8Ld+mZJ/2J4P1QXtWluM9Az04ZfJIvFiONsUiko/OCfP1c+l8GQmJz2F6rNz6TtWjNgXV/1Z/lKtuLvi2YXXDm87Zx0LfWJzOVuItwbPZ+YakDiwgClHFJ58gXS1m85n3YXjkZRJsJWnDru9z1rHA4T/Cst5luGn9wwDSjZKoDIXE4VwyZW/DINC3Ri9n9Ezg0O8AFfv4L6on+0iP6t8T9uPejnbZvNmEPEoyFkC0akHynTZpAj4wbYqpdvZsMtuXm+SLzHVNSr70HgNeV2yUo3NOJZ+2pjKjRsj+PtN3K86XKmkfSwj8IIqxHVe0WS4fWmjh34cWyfcfALdPPo9X5KEqKgofJZ9Krsk++ZpUn0qkagoBqD75BFzPSqjgGDGATgNviIVgc+qqmmsnu8bwG6iy8zcp+bjVpxI9PG+5wXkxDDuOxjR6fqbfLpjVDwzY7cPR353jpFNfn/czfHPCOPSF/V75v04cn0KemAu9yaTAUDHUm6/g9m6jT7abFtyU+DjKC5OvwEJBJW7/em5FhQ6xmWB+pv7qkvxsdn89amNVy7cwRJR8nW2OCeBgL3DZe4EVBgXHvNvs15lgUfJ9/JaPY78/7Zd5ue05MnBso/qU2/komiAB9XOjZpeSaC4lX/zcfryTIY2Cj10D3sTJte7qU66naE8h5/Jii6o8quZozH3cHclbHpW5dkwnCbgz6qxQoYQIsAVbhsMe+DB+Vv1lnJf5Pa+MIxQPGN7P1g5FT9bXTxWJtOwRbX4LU3mDytZ+8sGWxYJ1C/Dx7mdwy8gC7cplMgrgrhcU/1eW9GePH4fbp5p8kRZU29quKVWMLWH9YErsGMfId54HnPKQXIRPXzvYa67L5FnAt2nE9vTlDIYyyecTXffB1x7EU+0VPNIWR8LlSpspOu4wJO/MSIAm92akTqVfzfaH7adpNEOjWcjc1rSRcHP1LPyxer52XEXIqZJPzkfa8CFmkVoeYWglRE0Qr4Qq+ZLNXNe3s+X+wD6fxRfv/iIuX3sHlpRLsk8+Q+R6ukn8+BY/By7cSi/HJ1BcXNUKJfni53Xuur/h85MnYiiMVP3yBhl5c7ksXAiJRu5xozZXdM/0s9uXu43T6Lpppqo4wWyu61byKd+aVZGf3jsVByTjTZ0o+ci3Tr/dpI00y+VDCy38G9Fqxa8bzJ3j18t/wA8rv7Ze5RN4A5wq+fLWwA3dRFPNhWv1YuDJoGiLrssl/aEHym1p6RYln2uBlJi6StJyGU1TqlmUfHkmNpaM43zEjqk8SNkWOXRAPO+mlHh458DtWNT+UXShL5eSjz57xszPIldbY6XIT9vXlynR2IC+emT2saHuR4ZtO3FkdqImYKDuNrvcNFjHYy97BLxoUhTYs+47y/49Zbxcp/89AZNao3c10L8eoOZmtsXW9L3d+StVyNU3mFDgufbX3EQynbwazc4FFt4HLLoPuONb+rmc0QDlRVL6kKi/LRdcSr6zK4pvtplvMZY1JJk1yfedLOxMuy6qks9HbcOYsx/PC570lUyuk4KQK98BbceDZmXnUp5G7DSbOTHpXxXbs8XSbzo+mFqXiGpZR4Avl6+SztnGLpGjaaLXCb+NExNs98Q8TfGGw3ewzVz3lkW3AAB+MsFNyhqR0Y8s7UmfITdLwQAAN3SNwBnPX2o8F0rvPW6vQ31Y1P5RvH/oZku9bEo+M8n35Xu/nPwtvtnBElHImxzhJ64THLjig/bvWnofdiWfPfBG+qs33gwTuXCuzHdy9quCpGZg0jvWunWe+snM6pWaT/K5VVNZV9uuM22Gj2LRGF61uKoQ4JzmxZK2ywm5EwQsbROkXYk+QyIKc4D65PMi+eIGsj52iVIHk811S7qbIUryjlj+GDlFyvEgf0SblsqLr3u5Fllv3Dd/VVQNq5Iv3zgoUpcy1hmuNtTJ+jHOGF06gulKyVyXG6Lrqm3REHgjCh4Zp8xQ8hk3habvA0Ce7xi3mxKFaarkY8o9pEq+lrluC298tEi+Ycb1I0fgNNXsjuCzZcskLoaPuW6UziNNgbWTeok8yAvVnuk6ezSmRhAyWJV8LlMn6pNPpEones1FoEw8EuUdMx/3haqO9A+8YT5+dP9fAQCT2BrjfOKZqmziJb33gg/NuDYJSlGGhh1SEfnsdRLoeYMSQia11IJVZmJAQxP8lRX3uxbBbC6i1MtE3v1wa+Cns4DuGdHvUdPsC77P3Gk+rhUrlFVNNtf9hOhn5Xtd1rss+Xve2nkAgJplkkwXAE6zcNE2ygZSIa/5rMt/oQdcbWPHyYrzajqpJR84Vd7UlKZSjx+DmShWSD4fMIawiTvo4jVlKVProeLnp07ae6wiV3F3uHvyt7kv10k++iT+u3qxlFoamwxTM0Hy1QzLF9vdhY4xrs2xoPc111XTFnGJqo5dRXtE+yaX/ixnjO3ANhNG4Kcf2h23n35AdHDpU/rFGW1306DyPRfoi+VvNLr7Sn9EBHyklvp3/sCe07PLCSrOeV4kXIvKqAdkfDcorrw2Wl6+3++eaX/iMtc1diMsIfnaeBp0RvbJl9ctgdiwTP8GTEo+ec4lVUv73ZwNOxfeseMEr3Su6Lqutzoa7nlLyMmmvvJd8aSvRTpfIIqtV8K16GMMr67ZVCjwhqTk8zHXFXmRtpdFgFk373KQfC+tfQnzNywEoG7s2zZGbDnl6EwjCZyhTBkc7r795PKN+IcaXVoqx008hwj1/WDG8PfyatwRK6kDS+CNxATf8E5XK49cfWZPsiHMXTxXMtelSZI6Jua6ZUndLAQTHKwVeKOFzQotkm+Y8fUJ4zBnRGdhIkkm+cz418oNqVnSMNEhKrFH66Qr+XRCygRR509Mnoi3bDHdmk66BgDCunFgdE0caGTadCFlnwhl18OV1myu61pU5UGi5FPrYJkt2Ba44noOZpxUfWTaZDl7S30atrT6T5DFj9kyV/Is1dfGfk/zoSYo+bIct2fBa2FuU+j1r0uDSWy5f4HFlkD6VQJm4jQX1Ocq/KopjfXQqw/VLh2yTJJp3I2ak+SLn0fZ4LMnr1lZg4E3bEq+N28xGtPGKOSVtHNtXpT/eJysThXvyWBBTuqb712Gxsz8oDYbsWi3meuKaLxhqJrrkvekECBUAeCsi+FY6HiHtD+f21nGeePGSOeF6VeoBXYKtbGrDqCfqCIDQ2XamIvkc8P2Rs0jiTv/yBG7fL5S0nPJGp9tPvlKhsVaOQiw89RuHLXHNGw/KTbHvuQAPdMBN+nR13SSz47jZm+RXU5Qcm52MqQbFCEl+ZTNhE21TfjXhkVxWRnfrxfJl37TLhW8VPet3h79Wx2BHz72QwBAO4/epeaTr6C5biCZInDDuMOTzkO0v5tGdGJ5qaS1x9dDyfdx2gYcUNXQtG4mJd9Q/FxHMbfFREg3+RmT2q4UhEkx131l/Sv4xsCt+MWYKNqud+ANcht0zh36KPkMkJR1almAQYWG6F5ykHxHX380FvdFrhCkt6AFm4ggbZBJIdPT49mEO7OvC7SUwLJSCb3e8wmqyNXzlqyjeIinFq9TUjBc0v4yfjCxM85CV/IB5JtW5pDPrnoWB04ahetj89vINZF87fHl1ThlzilSeyyZ3uXG2C9wUMaZ956Z3JJojyFYSgS3lHwtbAZoeZZ8nVCUfKODtW2H5oWl6zE5yDYpyFuDA4Knkh0Oq0rMsMBg4CTwRrYS6x8dGQ7s27uTPznETk++RUCidGR64A3mOTj6gmkTjyjfKuSJaH4ln/g3yl/zyWe7zuo4Os0nl7muR11z+VTz2jEbRi3fsZcDOxye65LXNr6W/G2qWd+QJ9nVBJJv9pWzzVknf+RZHFvgUqD1xn6pwpp5sSXe72E/BDYu088DwLQ9gWVPAx0RsWGb1BZW8iUT7Ox7tZF8VOXhjFItFrFGJV/O950RTCALtu8wMOVHFyGkXLo73lOS65+YaDVLyQc3EZYJ5X51n3yWMjm3m+taoG526efjrDzfGx1LvzM5WhB9Y3VKqpYt0XXVcQUAzpw4HneM6MRu/QNx3no7aEdK3n6udD1qAzsmOeXxTyu1UM9XlzX2jWwrY22fTEJmk3zmipiUfFyNzmpDv9t0rabJVvKPU3Rhql5NCV2purZvipUyayCIjHqJ9E+BTPJd+fyVxmvfus04/aDXxg5Rszqj6xK876fA/mckYwIAVHiA1RsHcNVji5ULGwi84TTX5dIVfYzhrInjsfXgEKYulJM2m+RTsQ1bAvCZXmldG9iiP+pqK2PDQPTFi7VF1ichKfnAknFQMtdlDLj3gihJTPLNXzcfALCwUolrZ39Wa/rXYEHPgjjfFN+7+YXkbz8ln3w3HG6SDwAWrjKQnLUBuRwH+dM31Cf9ltqEpXO0BzljwEFfj5TkG5+0lqnm7VIrRhv6HIfOnIZtBwdx7RLLfMyRg15L2rY45q9wb4wwg5KPQfjZhPYtv7TuJQDAw+1tOGKjbLavgvoP1kJ7XPbeyI0KoPV3ov1zMJTFu24p+VrYDPAfIKHZfPCv79mJgmaQfI6hoOFyTPhu+TLtGNfK0+sVEU4FYtfO2Nd8fPJuyWI1Crxh9j93cvl6a9Zhsqy1P5/mRY/VSU8gNbdKj6fYli2Oonl5QCX7aBlSOiV0vQoxAWnDkNcyNCGZ6cYe8lmcmgP7/Zu7oV2OAsqm6JN23LP4Huf5flXhYUOTfPKZMLXmVwfTJFN7Tz5KCRvJJ8yw9z0RONjgpw4ADrsQOPGeRFHZqAmyHvTCnziymetSwsxLyVdqA+7/CfDqo95lO9FEJd/BPdcAL96k5K9HvAPc5nWSekOF6pPPB5yDe/YHPmrPNKook+ukoO4KvAGZ6BTIGi+Ez7uNPHVD4HoSWT5bKyTwBr2LqP+W63LHiDhQFRN566DK8q9W/oSv95zjLJ+CS3+nvwJWbPNKmlPwEF1cXyhmtX7bZpaJ5AtV8eCgRcHUr6pSMurUJHNd8QSp+WDAGLByHrDsGZgW2wCANf/KfPriswnpQldR8tFvXrzftnKAWdO7oSGnks/Vt0tnym3A+G3lbADMW24gER76VXYdTNWKWL44b46hmlI3ZUNZ9KbLyyYlX3rtEIA5nR25voSbRsjKarVtzWk7E9b3rkBNVQ8GMWurmbirsyMhAHefORonlm7Ag22fR4mlz8CZLyf0ITERFeeAeCxIyJSIKBHtSairXITokdcdiR888gMAwGMdZj+aiarex1yXPEmnko9zLO8xWGjU+hWFnX18or4vo7LpD/W6+JnbxvXdPgi8/SvAWz9vLU8viWeOIyLl/Gq+OW+Uuz0/ICL5fnjbi0oCXXkuSf/j85zHz0LZNBD9t2h5DHafjk6ffKJNApI6n0O2atp108NxmhbJ18IbHy2S7z8cfj75zIqCezWFXL5FIi0tUeRxuR4BMwXeQGIGZQu8YcR4c/Q5qDvd3KyN2D1YYM2aBt4QIoWQyQq4IjuxJ5Ruw23Vr0jHbFF0XU6N72z7Cn5Q+Y2zLNUXX6C8d1vtbdZvgiRsw6AXf5Dlz8SG9f0ZZsoe5nmNm2/ylHxpgh+8LGhmXDYUNm/NxvaD8UItK/CGj6TFp57PXQf0rgRmvhV4J4nUZzJbVVGuAlP3SH42HHjj5b/LvzMCL1AMMXM7p+a6zsAbiU++duDOc4BLD8ks0w/5ST51QX1Y554AgFl9DxuyD9J3QBb9Th9aw6Lk85uW+BDB4i2lTVxRdyTmuso9KO39a/d9Tcs7a3Oti0WqjgGkz1Knc1IEGQqXEiH5KFxuIJLAG4ZsK44Fb9ZiP7T88jPWVVWQclkHLf0t5vJPYgzWW68xwdaNmcx1ORRzzx4l0rRYBO72QWt5nAGySR0KkXx1qa0pcweiqGQMwC/2Bi5+m70cx7cKRG1WKK6kx6WYpIeGsV4zkU2q7NFXk+vod6u6S6ClXvvStfjN09GcaOvurU1ZpXjqj4aDMsxEPZPqf+qfFMUUsUjRyGgtryifxXw8fjO6G6dNmoB7Ozwiscb4nzGjsxN5zl3U+fbGakRiX9o9Kt3wZgxnV/6IKWxNki7zu6fzfxYofmqjf6U2EhMlInJ9ctgxl1w34CbWo7JEMKfswBsCWUo+DlPQCOhKPofC68mVcvtR6S35V8a7nL6X+3yMAQb8ZeP8WIDBM5V8jYBn+OQzzdvURxop+czq6ohzl9+RGFOov05b+6EkH+P2MYYGXWGcknzAB1f9IjoxjHPzFlp4vdAi+ZqIxrpPM2Q6yzwoRARPusshsKgi787mXTb7DwiajQPSwBv2Ur3rQ80pWPTbuvtlLSsl+WzlF3Gc/J3K5dghUKMkWpR8zG2uu0/wAlxQzWVVc10Vib8sm7lufD7yzZT9PFOTDjntDpO7nNe9/cK7tTrJFTF3Q0s3Ls2skzce/S1w3sRoMXfnt5uXrwWvp7muNevkj6zFsf7uZ4xV/LXlmfBMeZO8+2wIqJIFq7l30U42B7FrIxjpgmbiKEekTlfgjUaQs897csWTWNizUM7CNeSzANjl6OjvKW9KDg+FDhIpfiRGgiVR8sWJsiIsx9f4mreaiAi1VxckJMsgeeuco0QfTZ3cMwNuXXSrVkIAjn9VyvinRRExoayrQrjlb5GfCyWyEKFXVw1KvrSMdF7QzMAA1u+QKKNc0AgS8l52WXcPAGAcy0vy5VDyqR4/VDcCo2cC5/QAOx/pLFNX8uXfnKhxnXwyDpPSVNDmk6/sfEoMtB8jKRUlHyUeE1UhtwTY8Yqanb4D2r/SwEeA3K6+9cC3cNETF6F3qDcx3wQ83UsY8M6r32moFwiRB6zaqPgLjORFpO4p9tlK9plJ52OvlSMiaG3Jf4mlt2/TmyxG8iU+x5hp44Omy8hXiozLpHdZT+abNMOY5CM+jCNuvLHxOAl8lTEvWb1pNV7d8Gryu8Rc6bm0kZeg1o88Pvms8LrO8ly4Pc0vRo/Gd9c8gjs6O8Dg45OvsQ3UDvTjsIBuEqb5hYa81WA6JnNdIN5ECAA1cocYG2TRSTbJR0vQ/M8SFTO35eeY97TQwhsFLZKviXDNPYpMS+7o7MA9I9LOyC8PO4klEM4+xTMnM8FoUu7pF8udpk6oZc1iCY76VTLQczAgrOV+nnLgjXjCA6DK6hjDNsR1ag5MPgoBs+8kCl9S1eRD0HTlGe/cAQCweqN5d78eD7RtGPLiD8oSqZhizy3G6IkJVN9KGiw7o0dcd0RmnbwVfs9cE/370u3APy4jZefvAusehFcuc93t3pW7DsnlTVAlmpR8l56g7CLncWyumjkUIPlEpFsbjL5zNDjUZQ5Y+834WX/7fTvj3CN3tWcgJqn/ZnOP4285Hv9c9U/5oJMBCIC3nQp8czVQHZEcptF1VYjFllMNKtqoZzvgvot4j6bvWtBSFPXJd9T0qfjotMnG8WwE74vT+cGkTqBHSpK5LiH5WM26cBPXl6ArsRvpOeQFV/qrCAGjOlKn4+BG3k7SZeQjkXbLSZ3Mfby0WVVXxqg3fSSjtHhxqFaqwKYN7cM1moewXtKzLeiTjyHtnmQln7ohbFis2/wYkrrUACyomFx9m5V8gEzWm+ZAX7jrC1K95i3fYMg/Gys3rST5RAiySGlR11jBJggLDmCmErhIzI/qAO7tzD/m2ZSBcn08ST6FhRdfAEc6FzYRtlkEUF1S8rGk7XIg8XMpbcDHClE1UFmj/gsTIb3zeTAccvUhUltO5rAfvcqSL8lP9Bv1Qfl7c1meqKo1ekDvLLDNhHScxdpFwIb8G9uCSN4YBHFfan8mI9sreiVjbPTqu0OcW7kMv6r+DLNYRLwz6ayh39DWQ6bAGxwhF4FwzEq+hFpmQGBRZNp88rVDIe+Jcjlksr1cmlmL5GvhjY8Wyfc6ocjuyZcmTcB3J6WDgC0Pup/m6qaTq2fZTVDk9Hpu3HDUGHgjPrZVsBzOSZRypRFjt0rLL3cAtf7c7qmEOkSua3Rs31hBl+XLwhc2ErRd8bmnplOjJgJANzbil5WfYhQ2JvmkUbQI6ca4NuF588zI/GPJOnP01Vocd8fXJ9/YWFnhFafBZ0NSwLIIoyYeNnVV3WU2acKNpynO1PMvSlVlk6lu3ua6vF58ZxhZTszd6iUB02R/3EhFhZbHsbk6gTvmt/7XArjz5Ttx9v1nW85Gz/o1S5vW0u57snRdI6GghSrsqN2nYUSbK2aVKKvJw6tH3f/0wp9w7UvX2rMQE2Yj/xnXtyTf25BjshvG/lHNymqxIg4jZsFzgeoy1318+eO4ZeEtUbYe40qYvPa4LpN3U1JEETUTf0DJhfZ7Tp+du/xKbEYrOSa3qIIAOfK7AKUaBUln2jCztYy66AKgk3w/rfzCWne390CRo/iTknz5NyNtzZoBWIJuXDSmGwMsew7FGAOm7B792PE9yXGTuSy+IBIAACAASURBVG495ChTxoq2zaACHHBm5h2ol0UHCpjr0r7V8Y34KvnC4/8WpTe9CZ4SGdLzVJV81OzN9o0IkIX1T8aOxpHTp0rqKbXy6sYU/W26q0eXyT5Nv3HdM3KCBja6Mtuq4pNPogOUi6uxWuhv3WWsK+Xf4PH0RFooFSNkXHKsAMnHaeANxlKClkeBNzSSX5B8Ocx1fVD3jK5LTbTXlYLUXHf0FlpaGjxEPhF6K/nUMUkiMwd0ZfLOU4mPy6VPWfNVr5PLiBB69JHR9Tpu6+zAW7acgWcz/PRxANOwCgAwkm3S6sPBMbW7HYfPmpwce6ZXdoWgKvlWYxCvVEpxUBc4fPKRY7ZNLUt03Xbq0qJ7huSDmyP1YyiNwAUjdrfQwn8SWiRfE+EyIR2dEZreK38HyZeVBiCdpI+PLJiDfhjNSETOx1yapiWS653YK9oyhKGYCimsdgID+XdyJZ98yTEZeXYXXVVXJzBiZ+0T5dsz6qh/jp8u34zDS4/ghFJ6rVnJl/9ZSqZcHgTCuZXL4vQpGgmEmcBL8WS+vy9d5TcxsqLADbjMFwX6hjwnCKHZP0mCA21kV3x5E8x9/TZwc0x4VKUjMf30wYtrX8xM44xuS3HYBZHZXZK+eIOtx1lkqpWSd9LAx7HPScaFiAu/evJXOP/h8/GtByzBTbLqZGmHNiVf70ANv7rnX478CMn3f0fJjq8dcEXXPeHWE/CVuZEPVDWSYVSWmUBIsnxYd9A/VFfSAF7m6Vn9YIXrzy3LJ596tEYyTv1JqWNp9pjPoZN8JafJXMZi3/LLP7qufD0znGTguL47wG9Gd+OKUV1KqXr9Agaga3JE5O56DDmut+tayFEqWZRxlU7njWw7etukBhop0LCSj0v/WlU5tv4vKIFP291ZXt1E8in3Gyomc9zV95F7fqIt2hxa0x/7eZu2p5a/OmZRQrEQXdfAGCiewb8qZSyouu8NSANv6JRZaqmxolys31dftTGXgj75SuS4uGeTJXFWzYfq5ui6gCF4EZD65IuVfEMZweB8ccEx8WZNjnd//NTJaR8a6Jt0jFs2jTmHr0++XL6EGUC7INe9UMsGzR1Q/POWESO8nqtpvHgwVp4+11bRztEytEYKVckXuVJqr6TP6OP/ukJKH/C6pIY8ve0xfGTG2HQTweaTLznCMWKled5PXR/Q5t1BlXxbHSBdEzI58EaClpKvhc0ALZLvDQQbCcUgEzbNgk0foE48k1QsjWDLQCdu5h1DjVjyWCHwsAbMn2PepXZgHR8JQCb5zIusxlE0nyxfVGrgjaxAGL5+C1n833t2m+qVvhFMHW0wY/FQPNme6PVPveZXsG1y7Jiw2SBIviNnHiYy19J4m+tuXOZu9xk+ocx+yfKhXPJoJ3l88jWgrDjhlhNw8VMX27MW1cldRA4C33Zc+BxyNVfO00idjSj5Dr8QOO1p5aD7Pf3yqV9mZuvMwdIObUq+C299AbWQO143IfkW3ptZNwFfc11qxmfNy4OYfWxRREgsWUvUoR4T/KyxNg20REkcGP+O6qgH3hgi1TaqDeJ6cNTxo7GjsV5ZZIu0HG6n8yqy3oDtzgPGCs1B6FhF5wtCidjH5IjCpjIiUy+ufXdmki9UlHx0EW+u47o40u7sKbNJPmk9OgfXAL8/ynyxA+YI2HolpO/MOp7J9zrn5Tk476HzpFyToEE0j5pMSMsmtDwhc7LMdTUc99coajqBer+d5dTsNcvnqvGuCzjI50o02aOmT8V/Te/AhC7VjyqlyblcP+UdVBPlbjGo141jPYZUft+WqY+IjpN5pLFPdOc/WEvd3ejRdQ1WATHJN1CPSJZNQdQ/GPui91/iLJviLduMiwvNN/8pJSSfORiPcfMwj5JPuV48q37GcPKkCVjUs0g672OCv6JvBdYOrE2vUUm++PcjcaDFKVgDF8xm4OIfy5rPcGxv9qI1P9d4a/PJFwVDgvY9Jz75SJ5THz7XmDfdQJBIPkb6tydl0jEEBw28AQAvVCs4fcktmk/jFlp4o6FF8r2BYOs2ZSWfHUkqz0WUrOSzD/7JgJ3s0nEwovwZQsmg5OMGJV92vcKBHmD9Ykx+7ILMtBQbEE0kqUmU6hA2HxEgpx1PzBttg/A6PiKKPDpmS0uefiQfS37LgTfyLq1EeTsEr2DLn0/FPpXmDWim2uwwqQuf3X9rPXFBh8SvrjEoefKiABEjSI+2kt20wdtct2+1uw4Td7Seuvala/H0SpUIAtpKUVtMnlgG6dbdYd69lTDMkcY457hxwY14fMXjXumd0W1d8Or6zHnXnZFkYzz+e+DBn0d/e0SNTnDwt4GT74/84X1tiTmNodxVmyLTGV9VtOrfRj5pJrypcrVELtzQn6HupEq+HPCNrqtGMjTmxeU+UwPnWB/fx5G7TyMX+vnkIxlp5yuxssd30W8yYaPeZ8tMmOvKW1wBOBZ3Lcfl3aPwUyVCZ0KKW/K3wdmatj1EOl+I1PNQoUeRIqNzdZY9z4lIvlD7TjbVdNP+ep2jTL9PyWTWXOdP3vZJAKn5LwdLo3wC2HfFn4GeV8wXOyCNlVz7I4FEQFh98gVSX3DaPafhzy/+WcknTkoPKj7TVLVdbzyeaUotpS7JBqqoQ8doKWo6oM8NKKGY3UJNBEXxsYkp/eahO09S8o7aU2r9QefEMqqGKNd5vgw1v8uqP9QTFVTyURIjcfdSwFx3sF6XngD1txaZ66oZRt+K+AYH4jI/8KYJeuZv+rCzbCNc795wfxWHkg+wKflCSG/SYXlim4U/3N6Gv3d24MJHL5RSBzaXAQAueeoSzLp8FjYMypZLNnNdIBpzPlS+x/EW88ojZND7+1LlagBAb3lIOp+Y3dpqwE0++WKjFh+ffI67o+bZ1Fy37PCHzhnwtm3GxGVEZa0qlXBn7yKsH1xvva6FFt4IaJF8byC4zXKylXzizIZar+fkI2M3OUbiay5R8sW7NTFqKKHG6jhv3Bj0NijX/1m8iJnwpMufkI6UILPfeZYyzpk/uSFb5KeN6ADGbAG858fGujQWeEO/L1tuKiFwcPBE9Mdz12lp53bopt1FTXQP3GGC2UF/QXObft8Iti4UIfli0qPCInLM1KK8o+sWrEPvUC++9cC38KnbPqVnl/PrGjfCIwrsMPsneXTZo/jafV/zTu9trivQhOi6vQPRM3AGmXjhplzVSrD/l4DJsyJ/eG0jLYn0ckXESH+K31F3QzTguYvn4skVKZl2QF/aH9w/f5VfWXm+b87BaWTmBvHc0miSbiNmOYDrnohI1WljiMrYaZKfQRw6oG8spYhU5vJ7pOa6aeANw4ZZfGhQuc9aKrPPOb6525M6ciWlOJV85vFKKMnTVNGvz+6/JcpxshoYsojBdr4JmH8H8NoT0nF5YR3nF3JZwezRRuevmw8AKJEFqjEapwLXk3xhzQuY8/Ic60U0gIKs5HP5YnUt8SmRQUlDebyiJB8HxzG/eiC6Pq+SzwAt8EZIy/L/qnac3BVn4Bhrxxo2FQlY3wrpt32zJKqZK3RNNSH1i21A+VmAFCP5BOHBCVluImyznv7AUJjWkwXkXcY++TRzXdknX3/cgEyuDAoht5JPBMUykXyKkk/8nUPJpyK1GorXPcoHJD0u5V5+/uTPLXmqIoIUM9iKuDxbhfIrremmEjeY674wUt5s57xRJZ/8HEw++WwIDRsOgBw00FSueCbiPYnUQYsiaeENjlYLfkPBvuNt+ttk+rC0VMJ+d5yA34/qUvLQO0E55pDoBPWFSAeL/R0Iko/JSr4aL2H+6MX486guXDp6lFbPtJDsCd7fC0QtA2iwCuqTz7zb6QN1Lkhz6lb8L2rBPqwLTr9Jl6hniZkHNIHKS7fg0bbPaTvMibVOskNmX4yfMnkiFirR8mQSyX9SnkygJu6snMgmxExvplZUzUXRCMln2Q0GoslwV1sZc854Ow6fNRkf2WemPcMCUViX9y3PTOOr3K3Eznm+9/5Z9kTDTPJtHNrolU7ck1iovmnGaHztsB2x2/Ru+0USomdx/5L7cevCW3PV8aK7okW+83E2xUmlf96iLfr6ZnQSwAZl6ilzTsHlz12uHV/fP4QVGwa042bk+07DWccaj6umTj5l/e3JyJw/sixLz63ZIjK1n/P8ctz67DIAQAfxIyT3SfIzMykKXAsn2TG5raZKIKW4rjVSdBp4Qx+30vFZhiAJOfJtYjkXglymkRgZh1z8t//ikqgXkT4HedTR85pS93TdgMhct5TTXFeABvK46rFXHSnjrB3njr3hWDyzmgaSUBbwNp98j13qKI9L/6qQAm9s9y7gbacBOx2hpJHbyvwVUf9sCtIEHupjugMun3zJGTK22vqsb70vLtOl5jpprrMu1WtOkOuSMaeQZ3IKyRcH3nC6umwUBV1hBEnfkTZv07dq+q7okcF6mLZJ1Y8j5/oGWPweB2rRWNHPogiwVRqErmsK8MH/s9bdecs5Sb4yi+cxcaAZuoHFOVCL/bN+eO8Z6UXXfBpYmfrEy+fiJX7u8WNR27L0vDytJVwkny04U9489fMCevTeUUPyHFoE47LD7Ic65PHzyfTJp2DH9yZ/0ui69C1JSr4j/kcul3GNCBF3aIvM3kILbxS0WvAwQu02P10qqPKIYes3maEsW8IlMWFz14gR0inTToctT3VA+H7l0uRMkkYdfJm6CDFNQS3kVwM+vgTE4EcXUmqujfgzpIPaJKxT8iX5k4RadF3H6kImXEPtelPdO+/6OiawHkxka6XjeZ/ngkoFf+yS1UXqjuRxs2USy1REMn8eoZhqKIP6A0sewENLH5LLM9yft5JvsA949SHzuZykzKbaJlwz7xoAQDlZiMh1e23dJtz5/HJsGKhhmwkj8cuP7YnvH+0g0ApMJOqOCaGvL0aBkHN0tZfx0X0dRGQuk6j831HeyZRoS99//yyc9PZtcP3n99N2gI11ip/N5+78HM6c6xdBU4VxoZuAEgeFsvfLm+CY649JHJv75mCsmsP8XIWXuXRSWI6FGGPJol9dEH3nwe/45wO5jowBqKcLy9XbHg0AWNubEpUd1RKwaS1w/0+cPvkSks9zNS+TfGS5pK6HSTrRs1BzXUryqX2/qIpKANJAAY0o1TUw459gDhqZlq8Sdjb3YImSj6n3rGPFWj0g14trzEF8aiFHxUby5di42jhANz8spJp3bunYbJoLSGPqI7/OzMOGVK3EI3PaQ7+jRde1BZcyKnTCOjByItA53lmurX6yajDGt1abrwVPXKPsMWNMWr4Jo6YBbV3mczFYr6xGvuqxxeZ0MZFelxqu/CyqOc3zVQRe3Ylfn6MGcStL18Ub3p7muhLJR33ygSXvTphp6tF1Y3PdemSuK5R8VU7a1/bvBnaWSWZvOMcW/f7qguSL67W2P50bc3DUOce00R34wTEkCvvq+cDiR0i25je8YN0CzTWAmpL2jppfcsu9qKS4Kkag7cYvuq7vyovAkW07p24PeETWZSn5DBvbQ/UQlTID1subNYxsVBnr+uHUx54UeEMy1yV9xLS9pMtDcARMfsYi+FfeuXQLLfynoUXyDSMGlf7hm5UrzAk9kdcMxpRKDAjhuG0y8/aNrpsmSgNvqMqfVDUm5+cDs2NqN345uhuXE7VikJB86YJIHVLzkHyulGOY2YdGSvL5K/lM+SRKPtUnn7VSZiWKVp5lknHapAn43vixWB0IpaaeZstxI/SDCpJdch4Ck3ZNTyiT9JPuPAmfvf2zxjpTbPL1e7fsn/ZzHoRnLawlk7djrj8mUTZVY1JEzeLRRW7HxxqKkHwe30SWE3OKgDHg/p/glJJusg1g2H3y5SX5jEEwFtztuiD+o/FJm9MnH72Px/634bLkvM3lzls7Dy+vf9kvC9f9G8x1Kag/vizFS1RY/Cxyqi3SiLhyXfNOuKnSN2AMkKLxxpN48jw6qyXgpi8Dd54TmX1mwNcXLoW81FYVeelzEs+akgqiv9+OLbZqqVVz4HritBwoMf9vmPYdV9INHlaCukFHf7mUfKL+x5XuwFnlP6ZZSvoi+RmV47ZQR/YzXrteJ/k+c/tnkr+7Y9vaMA4WU5J88lElX8Y4rCw6s1CE6zfOx7z9sbnngOLbZeT/KhZvsJBdpuTrX4vaxX89RDYR7HVQg0VREsNlzi6w3cSR2HvLMREpD9jHcIfSnhv+MqarjMAVbAOEgxtps1Upt61Bn3xeI6B3G5AhImmHZIPATPLpoL3GYE02W6XvuR4a8ozJY6HkG4g7iDI118059n/niF3SH67nYbi/kuKTT51HhaFBjajla67vqXefajiqihuU+kh1NKdSvyXVt2pANufUABIm+A2jUQ5PtlWxNmDSd82ViSXXviaDb0ZaPq8nz5B++wNDIfYbmAs8f72cXqwdPSpuM9et0FaszHPGDbYl33Ia3CpCS8nXwhsdrRY8jBgyDJl/rn4XnyzdUig/l4NqMfnIMjkV530ILrob6LWgYTQ96WwZNBOGrB1DqR7WRaL9Xn81phs/GjdGK4+W+892WbkyohrX/5DvAPudbs3bBHWHTj5HyTmdOPUvQ0aQ+U4skwYrV+xuO/Uci2xTTskOPudAezewzTvi38UIpGsetwQo0Ap25Z89aT797tOxzxX7AABe3ZCaZ9nMdTPn4UOK6qrJSr6kHp4VinbgAdx5Ds6sXGVJNLzmuj5+BEeSFWA9ifZIrnNGRC2y1DbD6Hw+DIGF98kz6A3+5oNOnHQfsP8ZQKUzO20mXCyMOwBLhadP0YvkE2XR9jd6i8yrRH+vtgk//zhpWbVQmfDXUtXehvom3DqiU3oclVIADPg72vY3103hojvbMETGKZGeKvmib34EG8AhwT9I/iFsAVXEV8tR3Fz3++PHpidKFUTmuubxzuWTTxw/r/I7zAhWkuN2CAVSnTGogaZUtDH9+zct+obidiH55HOqgGVY+6ocCvllvcuMajk1B8knn3fudnDQwBscthX/uoF10jUCxg2O9YujMWzkBK85jTqfe6037Sez7pEDeHDBaqzuJSSRbXyPx+eQh1qZ0/pj36IZGxDPjZmKH2ANLhwZmSuL1HUwTclbNTj3z7PR5se9cMztaMfSkttkVNtASBR3UTkz2XK8fbU+1mfNywfqYUoyMSapMiOCDPJ3EERjimjrovxKAyTfUTRAEn1/e35STmjYtEosluK2QZVfDBx1TkxobfPdHO5VUnJMOYCo35amEpY5nR6xV8EuaURvapZtrhBznU3OiLXF8VMn47+mjUjvw/BI1ABUkdmtuYxubER5cH3yDC95Oo2oPFCrY4fB57VrVJ98rrGWzotpKmmDKyZF3zEjWn+MG2xDwGQBRVpWS8nXwhsbLZJvGFF/88e1Y/sGL+DbFbv/CRd8fMa5IuipUfkosgb3LYLlOCaYa00bnSBKPmnixbVJR5RGvtxmrlrndaMJyunla2AasFYZolqKZ1dGPdlN/M74cVKaSV2E9Mtp0uoU9yT/xpNqEoVYrmM+xUugKfmUOgtny+phpdyUDHKXHwLoRD/G9S3En+pnYBQ2pgE0PR5XygvwqK0I3yZehJVewDWPm9UG+sWNmands/ge4/FqoJMinHOs2pjhq+zmL8u/G/ALaELeiYnRzEZL9O9X8o0M0xjSoi1l1ltg3LbRvx7kfaHF9COXAJe/F5iXz8+fF6bsBhz8Ld8teCec0XVLbiVfRPJFV3r5wzRF1+0ca05LIJQ+dV7H7r/fneSXXSTFUD2tY8CY1IZ/tPwvOHPiePQHPWqlvfMv8jaoUkl9gp0Y0MYEaaFC+vuy4o/V5pMvJKqzPCSfVYkVVLSzMslnz9NVvnxdOj4KJV8tPrqkXMKTbVXjHMQU3dS0QSgIaqtPvgzYlHy2W1fTbRjcgEOvPhQ/ePgH1tSpmpW2Yb/6ucx1Q0YUgVzedHRkmF0HhVBx1UE9N3dx6jcvy5phoBZ9wwtWEr/HtrEp3rQ48rojk006gXGDYux2l1c5PAra8mK5BgaefL+DhgeRtr9i/bSXJUlYwymTJ+LYaZOdybTAG6R/YOC4sno+jliuB3Yw1aHOGBaFUdThyFyXphbpI3PdaVgJvEJcoyTWDnL/1AjJJyWn7552Ioeco/mZBIBAUfJdN1+2XAhphGBbG7bU1zSHUYlgdW7mY66r9sWq/3QmiSvcYLBvwlDQtcWKSpDku7zq2kiNwDlPlKOoym5+nmo/EZ3r5iVz/1sWpoKXwXoYBR5T6xw/I58emiozr+kaiXs6Ih/uJiUfVRSL+0031+LjLSVfC29wtFrwMKLeMSY7UQ4wcLRjAL+rXIAt2DJyPB3Ugww/QYm5rkVtRkHdkf6xej7+u3pxRgVJdF2L8ifdIdFRtww+IQ+xdmCtdvyjpTuN9f741EnaMbHAqLIarHQFT3coseV+tlRRUsfumk3Jx4D4GZmHYhOJ61KLBAgxwKKFj5G6i83T1AlfMsfX6uFuOxzAbys/wuzFl2J7vIwDg6eSc/tvn+2Pp54UHEbPeK9455Wa7jrKduG0Q7bDJcfvabnYcXUD/h7LgYium+ZxydwFOO8mfTcywTPXAE8oJH8B8sZlrqstQjPyz/KhEiXKoeRr9w2CAazsW4lnVz+b349gvFAPGIBXHwG+Ox7oc0R7be8GzumRdr3tKNAmVv8rrtjwKh4B4NX1urP/pkTXzfDJVybXGpV8z98IXLiNpJiLKkfS1rOfD+1bRTsfCofw8NKHM6+lqKs++cg3s6IWkXtcWzoY7ksojkUKIfTwVvKZz2kkH0sVvibFvY0kC4jmXk1R1Cef9W5iBYZ0ntHnYL9fW/n2KPM8UfINxQrBd8+YhuOnmgmOimFkr5HvUVRTENTloiSfN50XIVT6tt6hiKAybR6l37F8zZcO3R6zpvn1qzYlbHI+MdflxrFhsD6IeWvnaccBi4oZAGLfdjRAW1b9jOesZxDnb4BVyReNz4vWL8JA3bbx5ug3t3kHwlFT7amVS6tM98mnbiS74DMC8lit3pNTyZf482TROxqFPtNlRjwz9uDkvQ5Qn3wskEzI65zjr4MnA797d3px3F+EREkIACX6znISKYGNGNvl6PTv/U43Ku4SJV9MMv19yd/TrMCxvn8oe+PQEnijZDzuJvmkjYZ//M6Yb6ZPPlLursFCYx6kAsZ2pioO1TIG4lf0hynrDL5oOfkrUvIl1+93mrkehnczMBQm36wJ93d24OAZU519C50X3zpyBL4wOfL/LfnkMxDPaWDG2FyXifV0iyJp4Y2NVgseRqi+RxoFY8Dbg6dxUOkpnF2+Mj0Ojplx6HRbJD/xO10M2Ekq2/VAtF6zK/nSHXgmmcmkO8ZJKHnjjqE525pl4cws9X61og8UYveriqEk4qAGztHHGNbXB4HtDgXOegXYcn9zWgfU3FOSLwTAcO2yBzFrq5nginmR25+ibh4VgOPs8ePw7hnTwBV/S58s3YogdiqctTvuiq5LETLgraXnAMM73HHyKDlPA3mWHHvlQaB3NbDjeyLipXualtZwsfP0aYdsj3ftYtnddt6XP6Gj3lM10EmRu15Y4c7k6k/px3JOJPqG+nDjghut57XFXcaz41z2ofKVd+9gSOSr5GO5TN2P/NuR+PCNH86cTO05KSVwewdqOP2qKCpeKWDAAxcB4RDw8oPe5eZFZrCaAhGS8bkHcl+yfnA9Dr/2cO24y5k9hbMncJjrVoKKdK1RyXfTGRHR2iec5sdX9KammXAoUJMkSt0555i3xkw6aCDPoVanLiMY8D+pw22e9MkeS+tdjzYe9vXhShdFruBKnUiJiCSQBkluJ8kAFidUCaVaSWxEMHzwzVMy6yr8itnmAzd2tmOAq9txVOllvz+XGa+ZvOGJb0I98Iael8lcUjbHi1CvZ5F87jaR+ohyJkuztlxvaz4LV/Xiy1c/LR079eDtPDdCuJtgA9lsi2uj4q8v/dV6vfX92gJbGeD0GciiOv76aXNgEeOVGUo+Vz5agDgKFpAgQNF/dfK8RvbJmy0mJWke0AAKtidU435l2Px9RjNQewnmzf40r8FaPSVwGEs2STgswZgSFZaDqCJjZ8+Aqqw21ZFWLowCvpzTA2yVPVcvowbOAsWZb4RNQyHue2kVXoojSdsrYP4GzCRfBHH38jes+K5b+hRM0AUFKsmX3st5ld/FOdtRNDiHPbG85uSck/dr6S8Mz2qwHoIZlHwUK8plTclI8ZW5XzEel0i+auQ/nBKCLPk3Jf4Aj83vFlr4D0eL5BtG1HP4edGuNR61D8wHl54A4Dbppd1X1sARpdQ7ON3wlmZCzoT65FotTy3RZgGmLfpIHl4mrkE5eS4V1DDkIPkOmjkNb1v4++h3ezfwsavNSZXfPoNBpORj+M2rtwEAamU5EpfrXkR0N4oSQszt7Ih/ydceHDxuzTdV8inIIoOUf/MOf2GI1G/aimdzXi3XbYgs4G8+NWNy16C5rgBdMALUJ19at0rJtgh6FDjHosSgE559TsqsxwWPXoCr55nbZRGEisPs/8fel8fZUVT7f6v7brNmZrJN9kV2EvZNdkUEBTeIKCoioP4QUBYREEVQUXwiiqJPBOWBQhAEWcKTiCwB2UIihJAQIJCQhex7Mpnl3tv1+6O7uk+t3XcmvCfvc79/JHO7q6uqu6urTp3zPeecc/ROeqEVc7JVdsg5qQkcKLb2hcHyXfHWXvzCi7j1+FsjxjIwdebSeKh6tkQ2Y98PfO4vmftBYfoKeispY8gh4FuRIT6dClsW3QUbZPboL1/8pbFc8okbnpny3mh8m7xiYTeubSKenXgWYkw9eB650OBSCeB1YZjh3DDfc5LJ2g06hQlF5H+cHGW2JmweYeTSn4J+JIvC1DYXvp7PQw9MbkYj6yGbDdHPBG4mX1jzP5oasYqwfKpk1T9yp3TPgsvyfzb2dIvH8ExD6oEAngAAIABJREFUCd8eVMAv+QZJwSWFjbAwRUz972Vhts1LcnfhtCBxm6Mt56MX2s2Y8rz1Jzmmo6QdM8UuFTH5fN+SeCMNNS58NlnBpOziAP703BLDVVkbS1vDeRx+MGTy6WVU1lt2lrAZPZUeqc40Jt8zDSXc8NIN2du31TflD+mdc2eSQxA9rMQDJsGkRb+TSpuUzLWAvoqt3LyGlq2MRBnqXdFYbVlYx0d7yXofkON6dl1iwHA8Srrf0LJpE2XsyQ+ebK8kgsbkowbCj/8aOOn31mt9L2GL9VZ70ZBriM+t2tRtu0yBRclnMPRpSiPl2gJ6Q7nwFbs8p8qd2h4vxUC6MJ/Hr9oHRa6rtbvrqlATb6ijjfMk0Yu1bxYmH7OEwBkoZCZfXqk3UUqKZyNKZ4sDXEcd/76oj+B3EQNh8vVmTG8fHk+QpvQSL3zB9hVYlksm2qxKPoEFIz8Ffuqf499bPIa/rp6JO1qbIyYftZJkm6QrlnKqa6JY8jxTDxWXs+V8CNDcGU/iRZcgxgNsVy18eX3jkAa7uy5HmJEsOq503qSglRl89nbUa9cjUSip9HohmGXNrqv1hTI2rWV1BJwDL97mbCMrqNJl9xEtjpIAHvy6/VwNAsTNc2+WfquKDwDIGSzEAKyuGABkQSiKAeTCO9vcCUeSuGvZdqNPv7lOYXcoWPIs8PgPM9XV382pS0nuPmc5MfFoYJcP96svJvRVU+Zy23t3XlO7YtDGePzh8/L7uWXeLbX3R5k76ebCV9w0aby7GCJ7bWzgMbwcyuRrGwu0jMAtg1oxZfQIzCtEbjRQ56sgzmSdBi4x+QRjS3lmJ/0+vpMsTD6T0jG81j3W5xUKmDJ6BF5pT0Jr0Cs4gJU8iVEoMfniMkn/3Nlxk3JTSVb52E2OQZrfD2aOkAIA9vSWSL8PGzcGv2pvAwCs5WXpPvobk+/IsaNx4PgxOM6fja9XkxAG0rwVrV3bPE9h8un46pETtWPccI1w485LTD5LbK8Ik2+bTOpxcaGyY6DKM2u9rnh4oFl6Q5WPClXeUsesC/G4JX048I4DcdRdR8W/05R8m3z7XGq8N1N9beOADn08SH1lALatchagTD5AZm+qHiGCyZcldraxOfL3Eq6HnAFCV2oA8FKVuQqTjwXx8SxJgjrZhvhY1UuO9ymJN4SXDYfKEJWhuuvKTL7EgLN6+2rDvSh9lMhwipJvv9OAvT5t7YePatzez2f/HN2VRLFXHaAxOMfSDVGSLMOAQX0Ry/3xq63XqIYK3V1X/17oOD1p9Ajc3DYodl01ky6i6yxtZEcYmzFuw6bkE4YOMmZ6K1WwnC5TX/ev65S+kr595NpMvfINtJkkrEGyb44Tb0Q3UHfXreO9jvoIfhdBF42apsxdT0CPQdBsyptflxScOaUlWsPtrYmLpWnit7kWxe63JKjq2cOH4cqFU/GTwR0AuMGFgjl+Re2Rg3v2JhsedZErS0om5X4nTZF+cs4A5sFnIiafw92hhkVek2cUSyWFLbuuirSYfHr55A2FiTeSc1t5AymnWNqsNbrHjkpun+wtBqv22Yrr13MeuvQBwLE/kFvmHJ+4/xP4/nPfz9QzakFOZVFu1mOY9Qe3zZcVlDG7iPQllzU6OgXzgLOfAc59IVv5rDutDPjD04vxzqZurN3qYAisfT17hf20urqEKeoGwxmwvS8Zib7HzJqFAQhnps1337vB5OvHNQO1ajuvVhRpZcK629q3VcqQ/sfnZCWQBKHkM72XaiV5N197Fvjma3h19+MAACtyPjBiH6tSrVaI7Lo5LwBuPiY54SWxpEQXz/fvDeM6mm4nE5NP7+PyfDg/rCsm8a9ciTea0KMdpb9yFgOeamzJGfrCAXhEeXNX0a20Nz3xBcVofHAl8QZNAlKDu65mUHP0YpsnZzM1Pe+Wgvw90Q18eE2I3nLY33w/mXw2xXDW2IsuJRd40L/8Ou3jSXsOZQujsXG58RvV2Y9EvrT17XNRplbCjLrhpRtwzcxrACRxCMNmXd8zwzbH3C2u/MIhY5ODJnfdDAaU9NhribuuFynH6JtTW41j8vVzipYdOc2VlCN5K8dpWR2q7E4V7C7jrBjDFSTP78rGxbiyM/z2+yoBbin8LCrMiCGIm9114/4o7rr0IWVgadO6nUy+FPgI4rGxfJucuE23W6W/yNc2vBYnjzAm3lCYfOq5LEokPf6yquTTx7r9Dbvbo0o+U599rpMGaH94tDaI/ZZdySfHagz/BphhLCzevNje4d1OsJ8jMMVrDZRvgv4vvp+6kq+O9zrqI/hdBI17V9O6X+1Fj0GamvqVg3HVx/dwXuqO6yZj175EQWOKc2B01+XhGQ4m0exfKSXuBQXWC0bcykyuuaYn0kMn/IYkvlpfICuSxHRtVPIpghsHAMICyKcw+Zw45Bz3+bhf5t9qoGtNGeiQDtMtsG5loFSXImRnjcknYiaK8mfmpmPMs98F/vQpYN2bzmvDduUeUzy+7HEs2rwosxuqEPq+e8Lu+sm+7cDDlwG9KbFVwl45z27oSSza2ytysGrThjZnc9d1gXlA5yRgqCEWngFZGcLxnRn6yTnH1JlL8cOHXk2vyOIimtJqTXAxqlis1A/xi0eT+Gyyu67ML7KhrwbFtECqu25Gd1L5mtqVfK6EK7XA+JZUJp8lFmpPuYo7X1hqr/zXBwI/3wNWJl+8KYnOlyJj0yl/BJoG6/OTwYXXBjoPCnfdxvIW4J3ZpFAi9vhRHy7M32uNF2hvO2nrYE9nxiXrVIJ5xWSd5Ep/G1gvCQAetU0uHsY2GXvhgUuKhbxBicIBiV2fBle8OaZE9JUY5WGGE+N12RN/0O89rCtk8pnbjNtWjtHMjUl9wObu8D23NRLWCH3HR37L3TtLdl2b7kpL7RJnve/fem7ExA8k9adcG2fuzcjko/BtirG2SOk2cl8AYbKLm+behKmvTdWKuurnALocyl9xb8fsRphupvosm/PLD74cLL5/Ww8ivPVE/O178dnkKpXJV8wQk2+zxzCbyMqzS0VMnjAWGzxPGr/S90cUp2Ltoop8fsTFhrtQjeqR8oIBQ9lmtDCza6q4qsKTtWm114dZTeFv2djFUCEK1sAxnmmSA7piA8i0dv7p+cSoJG2NalbyJUw+VYHD+xFe6dPTPh3HgXO66+50jHYu7EPcuvU5qOuwOs9VjHs3O7LG5DM9jVyGQKQBjfGc4q6rzgU7b5iR3jdiVMoqd+UECz6KF805x8KNC8O/WfJMEyVfVH1dyVfHexz1EfwuwukC57ywz8jk62jIY8SgBu247LaZkjyB/J1tSywjFkwZs2ZCuqhwJ3IvJu5iI9gGtLMw9tZ/tbViXqFgbO/D6x9P+plL3GTVTXnFxeQz1cy8+Lm446ZY3teZjwAnXAcM291a1pXVWLYSuZh87nenMwSTXqi1UiHPxuTTkqCkjNeKqJOMzcEL7wbeehz4+7fVDmjYtbPFWmB993q4IZePXa5Mrj2zbgZm/ha4ZpSezVN1+0u55y8+/MWUfsnIWV2NXLvm2qbhmtlchvILVm7F5fe9kvF697hclM8lGxda9rzZwGfusF63uitxzemv8irU8Zm+ef0Q5xyHTD0EP52V7hKtIpXJV6vC7sRf9IttmFXZZYNz5Ph5XDPzGlz17FUA9FhAYr5xbeYAhLHvtrxj3nxXy8mz0gwykYBtYPJlHR9UwSHidvo55Tkzn8Tky7JpscWETY5fk9fjfwkFned4XHROD5l84nhUh7VPsrGI1pM3tMcBNK3P+L0jZZxwZR2L1r5PeE8jV7YbVnyWbezGbRPGYDWlV5/ad5R5vBGIHm8yKvlI3QefHf+5fOtyLQmEOmaO2NmdWV7dF4ux7FLG1Wwqmnh00l6Ku64UecDE5FOeI3XNt2bXjRT3QgFqY+QD6UYqU6gaAXFrkjHNxOQ78CvG62kCIfOyQZ5dtVdJvMGl5HCqSi9JWmN//ucNH4YzRgxHb1TPbZFr/TWD5XiZ0hMiHRWyMFXkm8bRm0WOH3e0E2ZW4i57Y+F6a/8+6T+NT/szcH3hP43npXVQYvIp40pB4q4rGFKkzw5FzZBK+PyXbkgMrBqTr4ZQGTkEcXtqogyxrv0/g9s/AOCYK8P/S63G06bEG4f788I/hoZ7hzXbaWI2nvDLOLfKour3qMrz5ZpkJ9OeSYeZmGFby1SjHNnL2L7l6Fmpc9WIrdnXqLBD2ZR8MZMvImrcu/De2IBPY7vH7rqi+rqSr473OOoj+F1E0F933Uofuo2To1xLYLCqSNZA9WomH6s6WGWAEqyUIBaSLNnL1L7fWfgR3k+YDmeOGAYPQaowKtBTlZlEFbF5MjL51HpYpOQLyzkzoJH3Ne2tacnxsQcDB37ZuSn3HK5Eakw+ax0ZR4nEDIwhsysamcykpLAG3w8quKsgu9FKp+PihrHZvdF4zfptoRvoxKFN+OoRRHhS3r2NNWQrL7w3jJsOWva6XeRzow9yt6NgyRa7W2JrPhTQac/y/XXXrQFpip4sMaPKaTHmasAnRo/EGSMMMYSG7AzsfqJ2mHOOG166AXPWJsG9y4aEDCpMT9a3Jd4wHFvRtQJd5S7c9fpdxvqHNISbdZN42xttNr506Hj9wk3LgCf/w9JrCw440y4AO2BKJFALCjArI3sZsHDTm5j62lTcu/BeR1tci8c3pNkSL49m1RUIKgmTzzD/L9myBNfOluPsBDzol3JTGAJ8dRPIqDOSun4YWK/WL8k9X4tMnDZFImfyKG0gMflAmC9p8BgH5bnZmHwjHIoXHY6xqbrrIsBktgi/LPwndp71PYcrYFYln+kdqNl1TRfK9avPXSzRm7aHipJBDWTc0rFOxsCUaVOkJBDhabneQyYOjtozvy1NDkth8gmFUmYMkdc4l/KwMWDxd2Fl8inffTcNj2Cbs2owcqTJfa47F/cmxdlUlRxXbQYOORsmiNienHHjd6keiRNviN/knMrkG8I2S2Up/rupEZMnjMWcyBh2ZudwXDJ0cHx+enMTXizR+M/JU9jOWKwU7IvYxrIiX7+Pq0dVceegFvQJWTliMqV9gafmnsC1eXNmY0CJTcs8VAj72WX8UedvW0w+ikaei7+D7jIJFvMuMflE/0/ca6T54iMuCsdWXidamOqTEHX65bVyBt1H5gulH7caU9Xvcbwnx5HsbNBlryzfkNQ95ben7Cd26g7vbd9tRT3xBtW5Ily6kq1CbUy+mpFx3hkuYkxG3/+r6xMPFo6EURlnoRZ8htrNLXXU8W+FupJvB6ET64FVodVmXjAeAFAZsVd8viYlX7UXfSZhyioUEkur+NuUpQiKu4Gljt+dtj+u+/TexmClnAeJcGiZYE0sRIoAQBNzZwmjd6oy+XqihaPgM1yUU9073Uy+PCpoiAS34RXl/sgiO3ftXEOvkrrVV5HmShSXYTQ3mVwuTcmn3pmk3lBOnuw/ba+XS/8lY6JrHQ72XrO2LzatRuVEt9mV7NJ7Q8tczmN2FgAyKPkUCIHMWCUVGrfXxhDMit07dseggkhuktRhZ/I5UKOyJ0uMMsbdpfxalJG1KJYysAy7yl24ae5NuPjJxM2oTDYLLQV7IpWqEvfHY8xCydCP2RSJBww/AIBbeSrGsZG1M+tm/di7hIEIxTd+6EYUIZJbyPj+4ME46cGTpGO2IAsVstH77ef3wxMXH529E9UysO/nw78NRqKvP/51vLTmJenYos2LsjP5yDsUykgtlK3n27+NN6brXba0naaISeZ5+3mJycdMTL7071Rl8tli8tUCV3nGVXfdREFZ3L7Sel12d129JwFzr7EMyDxPiZh8DTSGn+Xbp7HkbEifvuUCYjxt7DUZxvqzHmmWTStGVliiZLPE5Lt9we1y7cSAaV03alC0uObZNG9A0fU4i/3CR4G5d2duWyTLYvE/suJL9aaImXyKzAQkxmaBoWyz1Yh8Q5S0RmBuqYiHm5vsGzDyzA8eNRgnjRoRthmFzqAzGDcl6VPGQP8TKcgQ307UShx3nINrazP2+AQeW/oYlm1dFq8loWFDMc5byAIMSfRAqmiOlezbNwBz7wI2vp25/x6qcdw3nckXdacfhtqV21bi+ZXPW8/bnv6XDptACiSlVn/57/HfKqP+zsKP5EpqkB/N3k+kn1FVPrgUKmJINNibqyxO4kKukn4FnCfjs4aYfO9jekK5yRPGasek/mdk8u3EVgDtE4CGtqgOxQCEJAkHUI/JV8f/HdRH8A7C86WvAzceBgCYGYTU7GBMwhqqTclX1qyEplpMm4hYSDnmCmPVNibfHiMSCvr+49px8v6jjcFKE4aBjUEDYzxBKnRUMy1KyQW9VVkh+NExoaXNY8BXcn9TGpLrDpl8fpJdl1Xw4a6Q+n9UlxKbiyw4xgDijgnfdUcy804yQyrldOHXGHuIJZb4ZAzYBWctJl8/mSmqUCuhT94MiZoeXRC6ZNJkCca6uVvJp95dzNIxvSfXwq+W76f744IN5uyU1sQbzgGijKuP/9rZdhZ33bQvLDXouNRgbe4gaVAD4gOykq+t2KadD6tWo4GpTE7b3yFsm8umfJPzPAC8vCxUYpuVuP9z1t6BuOuOaBoBW19fJHGiBB55+xHpt7iyQjZzw1qLaCmZN2lGVPuA438CfHs5kJPb5ODGeImffeizmLVyVvY2IgjXsoK642UeiUuW/u70GKb6XyYI1z7PorlYnM9LdcjB8aMWMgwtdV0xu+vWakhwndTyH8ebIhYbAXXUquQ7bOFP415v8H1UGF0jDFY25dswzTNAwkaKFUVvPhq6l2eEmr08mUvN963e9couuyI0a+ZmuUPyuxXusDYlfRJGxszksz03wOGuGxl8076nrnIXLvvnZdbz6Uy+ELEi5o6Tgefc6yVFwUuMHKKKh5sa4/OqMkzMt4WgBwd7CySle8Vwrw3oNb47m+HbJlGqY2ZpPpxjeTl8N9SLxsx8lc99wJsT1TuwtWp72cx4BQxMvglH4oInLsDJD56sZdeVrrSQBTxSvlc1yAPA20/rx1IwlG0CwLGlbwumvy0bdARrsz9Kvkv/eanzvG3mmzCkKSogy7+nz7wq/juNvV9LYqrNhTXG8akukcf6s6WxIht10nmCflTh8vJWczHPQ8ADrO1O2P5T/KdSem/oTUYln4cAyCffufrM4jidUb9507Dod53JV8d7G3Ul37sABo4tvEHakK33a4jZxC1RS1I291s8lggpprTqkAN506Xjl59JWIeiiEko59QSZJkATTFV7KotGWujTTTdzFsD5Rs3vKqSD1DddQOINhRUEmViT6VHd9V0UMNda4HsrpsUvD7/W7l6l4WN/KseFX/blD9aTD4zsS9V4RUom5takOYeWiuTTyj5jJsOF4V/gNlJKXZY4g1VUNnvNGdxNRGNDS4Fwf82k08F/cZt1lMjaZNSMug4N7wbm4Ks6Bel81QA7Oqt4MoH5iX9NMXm+x8UBAfq3mJ7OybBXXWbFaBKvlwN8ZDiHng+UEzYmlQxYJsHlm51JPoAEndVchtiY1jMKffG/IRZkkHxpI2b2GrjHuux4sty/rSRndI5ut4mgf7N0DddyRGju26NQ9TJ5EMg1ecxHq+pzLGGZFVcifVl+JZXkGOJ8v/11oSVrd4OA9PWL1VuiJXU0VpUEAr722tTFKnzvphKs7jrHnXXUfjqP75qrTsOeUyOPXjeYZn7FtaRMi45cDBbgFzPBk25maZQsLvrhmuYa1O8tW+rlgxFRdoIEd+tMRZvBuT9JCZfT8RKo4k+pHe472lSTL6fF26U5WfDrfrxl5Cgwj30WtbbR4mCkcJUNwAEQgFLPXVM9ujof64cGKj0I7P1UpR80TreXenWxqRkeLYqaiJ50zaeMyTQUpXOjawP2LwsjjtLIfrfn6G1scccrkb03OTKzBAkjLetK6Rz72xLvsu0Nd8049rec3dua6Z5uB1braUO8l5X2uLS3wFPYty91G0xaDAfM1fOlA+l9sqAzEo+Ho/HVV2r8Jc3/iKfj9by2F13l2Oj43UVSR3vbdRH8LsAYVmnk/M3h6luXhxT81ebK+A8cY1UrhFS4HH+7NhawsDxQHMTDhs3Bm8VPJzROQx/2Piy6Wpp4pbZgskZIag5Y/JZY2GlCxKCyWcq98Gxo7VzKpMv6YepBsMxxV03iCbue6IkJvMKBUyeMBazyWbwvjfvw4n3najVY2tFjhdkVsaJSFDv9Kwz3k8v3IyYOPOTYcfmWiB167TY4NbGaosFzywsUy1DprPqmmPyiZ9GRlotmU4/9ivj4VVdq/DgWw9mqoIKObUrPVBzZtbXNsgu1Z/c6ZO1t5kVlV73uOh4n3Kgf0o+yuSrRbCyuusavgibsNxaDFnM+w3fTzt38z8X4bbnEmU/jQ2ElXOBJc9l6+iBXwEuWRy6jFiwrnsdzn/8fGzrsycvGJCSj1GGhxuTb5tsuhyA7K5b1HxhBwbb/eWV8BOqS68fiTJUaScyIhfVaOFeEpMvC7csS3ZdE+Lsuhl31iYmX5b+qatpzhKTrxY4lXxaTD6i5IPO5LuttQXTmhozM/mkq6kyUWlTv1Cu35Q4BkjcuI2s3C/9d2r/1JinTvba6dPAP5Ik+qHZ2t1tJPfXOajkKAl9bUx520HAcWE+CnOy/i3pnBr/OOpMDHteqXQj9uf/9nlnQg4Bk/tpfC66tazGtNc3yAoJMY/s4S3Bbiw0HFgT1x12QZJdl4fv+0wSe9bkbROOcfn5v807nclETLh+WFh+uqIEFEy+LahisyfmPLs8qM73tSr7nVDWac1dV5KXFSUf7YdF/vFY0l9Nedy1Drj3LGf3pr01TWtX/Hpj4xtaeRFqVovhmgFvb3nbeb5q+SZFrEQX0mRj0/fiNtKkrwZFVs5gsjDXR6MANPsJW39iH7kPKS6uqCfr+kCuyxCTr4zou446ddv827Qy4rtP1t2IpV1XkdTxHkdtu8s6MuPcEYOw19tJXAWVyeeB41D/VfWyGIFpMXbE5HumIRQEF+VzmN1Qwux1M3GBqQryN7UWeiRxQsLkM8Xkc7NldhySdi6acZGlSPqiELrrMim7blWZuE8d1QkA+M+2Qdr1Ehz3W0IfbKq2WDcGjpl9djbCg9VD3e1HSNx1addqcdcV/4uFLDqS8jzFEm28yxQt3h9OP1C9QK47TZBRylddVtes7roHfy2JD6bgxpdvjBMQWKsyPAmru65rI2iJSWODqnRozjfX0hoA4MxbM7g/zr4FeOhCdxnN/TldgNxe2a4do+xEU5Y6G0JGoqAp9I/J11poxV8//le0l9oxY9kMaWhWlCQTe4wg8QJ/d0TmfqJxMNDYAXx1hh4nctIUAOGYe3zZ43ho0UP47G6fNVYzUCafUeE06gAwZHdXpIk3irkaswo3d9rPcTuTaOprU6XfatZrHwxlyPPEk2+ErkBFrYtJoINA3ZgaoI6brIy0OLuu42u0KRhig06GdjxFsWZ6IztWyac6zfPEtcvw/f8syhw6ZkPtiTfo39J6p24QGTTGsTqWxDWauy7FqAMy9VFr24YJR4J37gG8ajYmxcWKI7C4N2S9PP2mbARsLKSJ6bUp+aoB8cZQNsnby/rcTGEN85Bhs7148+LUMhwIM2C+druzXFZj2pRpU6TfEwZNiMfR9GLoNkxtANLdscRQT11HBUzuup7BIZaBo1yjrPxKo4d1vodvKeQAoeTbzqs4fNxo3Ld8JV7etqimuncYGP1Ok5h2ybHkvMgqy+N/qaLGJv8kc4qoaXBTlCwnJQ7fA28+gO8+813tuGjVNM5FPFcrW7UfiJl8lm/Sz2ABSlvzs4RvEQhlVpOiLqor+quAirT2OFsg9yD64gn5nMhyUh2erxnuskIiLWR4V/tNGIsXFq2Klc66IYPHdSZZqCMWft1dt473OOpq6ncBHBxzGvL446t/tJZRFS9qDcaz3RuBu3VXPolTRydco7iRgLIFO247MikVHc6ZYsTxUEDsQxVffeGHWJTXBdCscT9caxNH+mY/8/TrJTH5CiiDW5hCpqPSArrrCQCAt4IRWt8v7U0y8ImnfH37oChwbGIl6lIW7G8OHRxnWVPfVx4VFEh6FPfbtD9MTckXF00YhgCAIMVdN2Zg1r7wTR5tVqByznH363dja58ldocFsbvuQJh8jgVcCKU27DF4j/h6+uTzOVugYcczqyEzoQmCiabCJZi9s0mOu/Sbz+ksNsz8XYbW1ftKFzgpay8+Vk1n8hn5evRg7xZr6XJQNioXgXC+3rl9Z5T8FMYMgJ2G2ZOCuBE9l4Y2YDBhP35vI3Dy7wEkyiQXkzFI+UYz9kLGGQ+DaYxM+/WUsVHMyfwqJ475HnC2HkdJCNKud5SG0Z4ex3HqzJCpozH5wGM2c5YYh3YlXxqTLzKicLvihc7hYbwocTxqgTH8aHC7xujxpWzuACe/jeuYs6fGjtlPNXcqyjcSk09Z/W3JvbJCDvNv7x4DoMYOtcV5FaEj8p6nr3kZWMQyk4/XFt/Ugs68kAECvLZKdqlryNeyPnDry+6oRmFKOFHyKWPb9P3RsTu4WY/dCSB+brVkozywUzX8RbKFJXupKAG4jGl2DGsYhs6mTlJLCIlURkccY/G3bxq7psdsctftz7gHzEpErsRL/NToEbhqw0ytXFxe+X9gq4cKOgdwzWASkO9CXfPl7Lrm8S29l6iuOJ5kzjIOI0x7a5rzvElxFsd57k/IlRSYlHyMceQyKPmueMYcXz2u29BdW60MZuldPVZAxUw0URmZCLCnt0Q7K9anvuj7aeKKTM50JV/Wp64n/pDx0wMv169BkoW5t6J7homYfGJcxga6urtuHe9x1EfwDkAHtihH0gUCZ7Yrzs2KshfMqe3liTupNzj4bK0PUuBgrQnBEIs2XUZOQMgaWMC24rl1L+M/Otr1EgNcIwMck17gAAAgAElEQVSEyp90Ro/hGWquKww0Jl+eVWoKPixt8PIlbB56AFZz/Z4nBUkiBlH7LYOE8kUIioFkGeIAHmluwsPNTdJ1As8Uv6EnFiGQXYTtSlMtJl+8PQlbHN4cLbZp7rpxY0aaqeNXhLef0Q7NWTsHP3z+h/jz6392tq3Wl8RPGYCSz4FcSh3XHH6NcUPTLytwjRZN9bvoKHZIvxmrZasFNBdzOGGvEfqJrfYA8TFOUhSBGazKJstzf911fequ+9pDyQnlPXztH1/Dl6Z/STp27Lhjpf6YrLZWVkyX2eW+Znhe3Nd4Q8kY5qyZY4xFOlAmn5hlJLct5mWyWIsSNL6mpORLq2PYHkDzUOvpy5++3BqawYXjxh+HfXKjAJjfV0EdTmT8ZWHy2Vy+0jbvQVzOxeQD5o/+DJaM/Cg6GHHT5sl/f25t0Rg9Yf3URXbHwnlnYw/BuilUFuHgcUw+eXx2kzGhZYWssR+ukBiMQVu/dCZfiEqVwxfZ3tXvKYPB5aARB8WbQtH2SKzD+9fcZSyfRZGcc8g6meKnkueshnMAgH2G7oO2ahiLUooPpszFxqQbpPv7j9NlHwD9MlT94NAfYNf2XeX2PYb/XpTuMp3FXXf+uvnS772H7Q3AoiCOQGWl5dtX4/KnL4+OG2QQcuGKXBhdrZNt0MLc9FfJZ0JQzRa7WHRNvLpsZoka+6L8ripjyTQrxUogOidbZC1Gnpz4BOL5OkVGsGXFjpWdhm9SdN9nDOi1h8yoBWnP3RtAIq2kDX33Z3fLZlK2bBsKKGeq9xPeM0aqgfiOeqNvoY3n5WdQ7dWUfG7iCy0n958aiAHAN4wnGpPPHJJAflPJuKurSOp4b6M+gncAdvGWK0cMSj5lclQX/qsHt2Nqi3C74+bAu6/bFT49PKSx031WuVFnN9gSb0h9ispMDw5CFcBmImTGsdCioWPKlGtaQmrZhBwybjTeLq9NVbS4gnzLBZOYfEUSkw+wbyIEtBQokQJFVVSY7k+IvQFxr2WGhROWI8MIs8PUP7lN+8Kt3Zfyc2hTznxCQSK0mUyHHA3owfuYw+1vqR6/zJXJDwBGl4R7nyJACsubkcmXdVqzj0rb2HvgEw/g+g9cj4ltE8lRojToT2KPGpSSs1bN0hQ9bSVzNtqsPTG6rQHZEm6M2r+mVi+acRHOfvRs7bh6T7/7kJlFqBlLrIol+fjMVTrbQbxjVQkjCa2223nzUcsJC7IoP6OWl29djtMePg0/nvljrUytSr5BxUEY0zKGtGEAY6kMnN9+KEkSRBNvFCQl3/+OOLFT207O/uvejjxmVpSDAFYdyq4nAB+/wfrMXSESAJpd115GxO9V2eXil5Pvz0RGU9emLirrPm0o72Yef+X5K5Of4EkIDMoSg8zmaWVmlqZ4umPZaq3t37QnDPDUvakyX1WCSpxUh15frgbJnKclVXGP4UdOfiRkcUN4G4SVnpKbYb0mizudMNwkiTdqfGPRhpn7RWOmz+5Kd6L44aR+lclncGO0uRpKiMdidkmvlCvh9Y1yzLyfdLRjRdcKY/lCkKjbtcQbDbrycdnWZdLvHEsmAhuTjz7377/8n3EYERGTj0L8fq2Qx3FjRuHOlmb8tXiV5H0R1tk/RY7RlFrj/C/mhYTJt+PMARUydjh0d13THiac7YAcPWcJV0IlZS82hInKdW8ACisjPKrUNKeLce57DKgQRVDjYOAbcgzYclDGlr4tmDhoIrLAvCficeidgYB+n2m1HVt5GmOY3UtFXF9EWWFRc+l/gSbWI5Xb1hu+F09h8hVUVcP8+w0M+WzwlJ2r+q5NBhOJyWcwJCbuuuH/q6KswLWZy+uo498PdSXfuwBumLjVI+pkeVdrC64ZEjFyuMVdd+LR2qEeFtq7hGBMaxUWjrMmnWXsg6qgExOcOByA4SeD23H4uDHoYYm0wJDEGaogzOpLkVXJt3SDeSHujpQ0vmIdvuygy5Q6My6QRMlHs+uqfTUlOzn30XOxpU92AzRZwkzxlEQAdM6q8XEmbSLk9tLux30+PSbfRyerririnUaLpkEJJ7UQd9e88P0u/ws8VvwWbsn/FE1BYgltLhqUWBmVYR0FocCSyx9//T8BuJQ8GeBgH9iUfC2FFhwz9hgA5qegWrMzoYaYfKYQAEMadJYPQ/aNvTEAfVyLAUN2NR8HUt/rP5b8w3hcZd0cOkqPT2nqjUdj8kmF05e2eNPHxH+ingxPrub3nF5eCL2bezcDAO5deC8eefuR+PyvXvyVxkZMwxWHXIGDOg8CEMYe7K8J5vBRh8fKJCnxBo3J108l30AF6bQYZJq7LufxNT19VTSZ5icAOHUqsN8Xtc1I0l42o0jq3bFwRZUOZWiBR/OXpwT77zbMiUGN86SrXe19MeKuqya/qKGtp4oXam2vziXvRpUyJgxpStqpcklhVQkqWLxlsTSPi+v7qkGiJFINGSnPKTEMiOLhX00wsEPinqZ/+7mBiOKcA7scBxxxMXo+/ENjEapMkxIkUEUN59jYa84SGuORK4D59+uGKbGW1jDMqNJNYLUjxicj9Wvsxn0+D3zmduDzSRxd9bkLeZJxWZEskcqIHFUg67KJySd+vxON0ecbzOEesjCnTDB5+/AamV9VsChe945XVpQVRZnmrusY05LHQzSWaEZZQJFhxJ5ErLspMZx7KvbvMazHwOSLXPd9j8nzQuMQoENW5n336e/isDsPw7jWcc52kvbMYyBLTD5nvSwnGRHSVqbhWIsbCjdox9W9hc+qcky+ePsnlzsn94B0ZOGabVJ9vdEYKXBlLHi+pmiV+jDSED5GXKrsdVSiQM7gGVP2ADAPlaAiKfkKIuyP4q57+8qnwt91Jl8d73HUR/AORntj3jjBagyUFMHPxJBTU8bPKhVx4PgxWNuwQaPnA6G1qTHXiJ3ad4qP0fOqLSu2ZjCxOeH476ZQkE6yg3G8UvSxmoWLaMCAB5vlwP8Dt02FKHgF6XdjrtFS0g4OBvAAh/vzMQLrkUdFYk1Iz8vwzJ9b+ZwS3yNUqq7bJr8LyX0n+l+Iq23eZviowgPX3HWlOrLfVlReb9ME8V7HdoTvUnQ1vjoLYwvA/c1NmNbcaGaNdK3BYd48AMAH/Tn4eCVUTrQ35vGpfUdBaTH+O22D7wpaD1hcmbLGLTvyYusp0wYEkF06mUHwset+3MyYrDC5cA4uDc58vQm1BG4GEG4obWgebj/nABX4sip94lhVprkyg3CmGhEEUp/GfV8D7tfZiAOF6T1888lvxn/f/MrNNdfJwPCdg7+D+z9xP4Y0DNG/fSBVsUFrA2Qmn8QCNYxN0/X/U9h3bBv2Hj0IeXUjRZV85SpabEq+uHgWFqYOwWRxu+uGVwZMVfJFx8m16xSGcsDyUVl5DurJyK43YTtj+O6QDmyqIcMkk/opJwG5fGj63KT2ravXvIGnd3XtlEn49d5v41T/saQOsnG/bvZ1mLFshtFtr1LliZKPbjQVBohJEUCVhpwBYvg3QWGkt45CLfBjJpxYF2uE5wPHXIGufMFaJGHyERUXucc/vvpHXPCEnq5N2tg/+yvgL6frSpYMyZLUNdU0/2a977w6PnkA7P4xYOcPxYfU95coaOURZ5OjikT+NCrcEMqPv20L1zxTtt2kZO0w6X+yMvlET/5rUCvO7hyGpxpDBeRAQukMg6wALksxL7mWXbdqMF6KTLBSXDXPR2+1F8ffe7xSOiEuiDk0bmPt63DBpljnAHDAmcZvO0nmxnSGr4K/LQ69qtb3rHeWi5Vulufu93NsfG34UJwyshPlfLOFaWtu0ONuk4M4RxXqLoxkG5T6xB4yfH690d4ir35BzHMntDr9QeDALxvbjLO0HxsaNDQmn2Fe6WEMYB6Ou+c4zFwpe3VwJIrDBsgsv7qSr473OuojeAcjVAKkT9wHeQvMJ778OEJ3IgOUeBzPRZbDjQ2b4vmYKqrKQRl5stBypWeqUNIGYYWB9H/4d2IjOntUO27y3wy7BIZGRakSAFjj+5g8YSzut8Sby4JSTraMNuYTJV9NS+PyMJPoNfnfwyOsR7WePksnJatgVObcqS+iuy8RuExCoB8JDVPyT+D7uVvhMQ6Xu246k0/5zZIFe3dvCfZa8l/G6wR1XujD3l6vbHwyCo5PNDXi8qFDYJs2qFWyxMNnxkm7/YIhuYXUpjHKvENAo2O+ZE4Gsnb7WkxbZA7cLC36BgGAWrPjTHBpcChZVQWDUcnXYN5IZ02QUrHFJbNtWmzHP3UTcJTuKpYFWd1QqbD8xMVHiw7pBTMornybi5nlUsGExctTzQVcqMFdd0dgfOt4AOF6lPfzeF9bmFjDqHBiLDW7tQAHR3eZKGTFc17+rwH1d0eBB5SZBLQ25PX5jQfxd9VTqdiZfBF0Jh+LXPB0BoV0XfSs3e660fXRWLyjtRlHjR2VKGTI6/pFh+yWHxDFUNrakXVk/bWlGQ+0NOO37fZM8+o4pZ4ETJEFnm10JVEwo7dq7i0duUfsPBh7PnM+rsn/IewT58DKlwEArxfyuH2Bnp01lpGouy6dexWlhCnTs8bki/5qYYqS76x/AJcsDpvIwL4a5MuGUu19Pn8jcJX9nQjYYpEJcIRrlCl5zEOLHjJek2nwZDBUqYlQasmiDsjvX4vJZ2JmaewkS3ukGJXj8n6yfpsUbgEYZpaKeL0Ytl22Ljk7bl5PCwfClT8WFsIxvTF6PwPpyQulc+O/F+bzOOIRkuGc632r+vZ5VXLX9fJGN/HxaI2U9zzxLhJtPHBOjb0X3WTAsD2M37YQg3w1VqdBzhOEg1XbVmVq1/zeuB6H7qjLgC8+kFrf040NWFAsoJJrlo3Myv8qsu1OgZFsvWRgiq8xfQiGcS8Ucb2xB5WejobKfC1V5TkwDzjhOmPf4mdmSaRh+s67IyXfmm7ZVTmZx8P7KjHZDbyeXbeO9zrqSr4dAE52nhPwjtFdV8WthWvNJ0bvD3CusewAaEwJYbH3g4TvRAWNclBG3svHgihnMjNAbWNW6RzsyxZK+2OxyRC3pK5VFQY0qskuGMPiKOvug81N6C9oPB0AaC8mcVeyOi1QRcfR/ss4yp8LGiOF9tzE5APUGA5JOOCeMn2CxBocPSwh4vQxhuP9WdE5u5LPDb1v9MgpuRk4+K1fWa4kFkoAn73p+aj9qIaNb9fUkywoRhaxIIjYiwumAY9fXXM9aUw+4yJcg0vL3a/fjc889Bnp2C3zbnH0xz1lVjlHSymHaecdjukXHOksG8OQ7UuACnCrulYZN3HN+WbtWC1eIKoFvt/Y+zNAroDHljwWZycOeKC54ZiwYH1i9LBnIk1w0r6j0DmoBLz1hEXpli6cCYWtmB/NiTcSXHr8bql1DgRZlAFZsffQMMj8iCY5oYrtVbuUrKfscgqA5Ilu7ArXoUTJCiDjZseEHRH3xlRHwHnkeiU/1ze2r4BwRuqtVNGYUcknsi9zBhzlvYw0JZ9IbOWaMR5tKuABthwiccVPBndgg+/HoSNovWo9TxT6MKtUjBQTScmBJr5S203Drt5SPFb8lmgdAMe9zU1wR82qva0jvVfIRariFXHinSmjDEmEYHHXpfUQY97sVbOx3+26y5gWLJ6Fq6jmrltqBRo7or657/D3H/49xhblUBqaku/F25x1CGwrpycMkHSoRHazKfozxZntR3ZdG5M6CzQln0FpoxrIVAWtgBxAhYQi8AtSGfUpcBbKdgJlsY6o9afE7rTCJNqkrBESEwuJB47wL9pR6sZ/NsoGeA6OvoqS+MbCggpj0ZGb83Kay+VPjvgJ2lCK+yvce/901sGZ+ucchyRrMoU45jMmjyeDXCC+ld7AnSgqUbqZn7wWk++oS41hmQRyypgu55uN7roDxSHeAmmP5UqKYYrzmIsSqfVGJjGfq0w++R00BTID3OWJkRN9ib7nvkDeF5tC7VRogjYFDPJ3T5Em79dRx7876iN4B+Pe6vkwTbW1Tb48prVLUNLQCyVfjvvxBFkmE265Kiv5AFkoMcXpmeQtTja9ZGlK3DsUa52ByUe3HC6RbxTW4qrcrdbzqpKvuZAoM+xLjok8LiNHth8BY9ipL1wkdu81u5v1VHpwzcxrsHzrcnDmGVkTdAlTY/KVweJFksbkS0vGokI/n6gwXVfGbTtrrwEZrFsN6MEfn3sbW3qijcO08/vXlOi15QaN2WydrDC5/A+f/yFeXf+qdKwhZ2efeMRNyCRIch66u08ePQhDW6Lx273RvUlzxI8RAlx3pRvH3nMsFmzQGcADtTbuMCUfQkH5ghkXxLHjbpl3C46/93gs2rTIed2Ty5+M/87E8BC3/KdPWs7XwOTTyhKhmTyaTFkurbA/43XdYaZeml2XYtpb0/DWprdqau2cfc7BXSfehUlDJmUqb2I2CBw26jDp94ZIyddBmar/m24t5NEKhcQ+P3gEc5dvDucHZT44ee4vpN/5lPcq3gtl8d5auFbSpAcAHitejKuOTgxRlWiQ+tyzTr5XDm/Fnf6SOIREPur/Nk9X8qlfxSXDhuDMEcPDUtQwZ5qXHPdngqu8qjyZ4iffLuMB1rSswFVDB+PGtnTmGaDLIVkYyMEaec7OpIeK/q9UOcax1cCyWVYW9Vce+YrxuKooEsqKJqbM4UR2SXP3PnhEoriwKgQdcVvnewFOmXYKtpe3440Nbzjb0vpDNto2Rb+190d/O/k7DvFSg5KvRiYf7U1Oddfd51RDSbnn9Pu1Kc+pHFfwFCWfSh6EPFbL8TOQMYRt1vqWBcbvOIXxrr4roeQr9CdWsAMbfP3d9SpKvsAyr3Jw+JK7bs7I5KNjKeAcIwaVcMjEAYYmAQDm6wn1EMaaZSzKGJ/ynIViKTsD3nxUc9d1hEnYmen3HviN5viCljrSmHzc8rdImmSan+RrxN4n3GP1IUApV4oUabQjnjTfcKZ4QznkCZ9FzzxiiqreLdY5yFInZ/YVp+6uW8d7HfUR/G4gQ+INCm05sTL55AVFLOA+95LMcaq7rie76369c2j827Y8iSrogmCzFVUZUFQECDkrk/hffwI/L/wWX8o9oh0X0K3myXC1xz9JBxUQAwC79oULkk3kfGnNS5j62lRc+tSlAJjRNVfN+ETrK7PkGre7rhsMHPc1N8XxkugzdWVXTBKqKJupfj7Doc3F1DIl3ovvPTA/bJ8xbeza4m2oiFmoli/IqHhRhZ4PfEf+fcE84OsvGutb1bUKG3o2WPtjsuzRnsXsIYonLaxdgQxMPpsb1rVH6nWLXKlZ7bvb+2wCrWN8nHY/cIQe01AIboK9N3vVbOl3Fojv/vaP3o67T7yb9IYZ/07DCytfMB5Xs+smY82MASn5LJusOWvm4AN3fwDTF0+PN9+q0Hr505fjkw9YlJkWFP1inAWUwnYHqiudC6+u3IKWUg6tJWIx/18WhtWkKZu2R1n+1CDqGnhq8p5PPfipsA2l3HH5JFERB/A+tgIfqTwRHxOsej/D/lq43jZFBrNt0RwvsSksYygMQSErHLX607sgwb0RVM7Sx8IDVL1wLK1xJFFwtWXbctGjwTtytsssKgxxfbka4I7tZwN/+JB1M2/7HqSYfAiZOIXyZkxgCpNVCpOS3rvEmCXkBAW+PfTDT0sVLNiwAPPXz8f3nv2escyx446N/65Sd12q5Iu+EzWRkzXRTLEV+ORvgc694kPPrXQn76LwmY9d2nfJXJ4+k3guZh5w5LcMWd515Woyz9vrpbIdlTepS3pcP2S5S/V6oeX6A9N1fIt7HVWv6fZkJZ+tLxUkCXvmFwr4+jA9mRfFdsOc2VuRvyVTIjsBaS31dSZfnG06qqPKHfN0x/ucfaUIlXx6PDgA6KkEaCrkojXDtIvRYZLLpNjh0aWm+ZfB4K7rgKkX3PMxEHfd7RbFNGBmhBv7YDjvRUq+Hh6g4BeiE7IST7yDEbwxujabki8fK/nCObGsZFqmYZ2kPhrGT9xfy3uou+vW8V5HXcn3LqBWgbrPMJGIOhLBnmnuug+2hMy2HHHXXUEE69mrZodMPstEVWUMkyeMxS9J/B1VkFHvRXUXqBimfcrkc0FM6qPKZoFadeeQLHsZ6g/7Yuif4jIr+mrbCgp33QqvAIwhjcknIGLy9TGzok1X8rmfWl9hE743dDAuI8HMtUX9oYvwWvF0uR8QTBRFSHVpBh0YPkheROXoVCFyRIXMGDQWqi0enoq0RdYo+K2cK/9uShTbYAxoGwMMlgXDWatm4a1Nb+HYe47FvQvvhQ2S4G9ouxpwPQZhmhumg8nXV+3Dr1/6Nbb2bTWeP36CGqw66pvUvrv5Y/foR7KM930AOOYK7bC2sYpjKmbf6og4onsP3Ru7D95drj+uN6USUuCsR84yFnExNk0wskYHiPnrQ0X4i2tejNkFD7714IDrLViUAra34HIVVjcRyzZsx54jW5Xxn+HZ/A8IzHocLp3JR8GQvVtCwS9a+O7QjvhczEYjz1Gs636GBVEw+ZoiVq2oj7Lc7Dmw5corhvvZkTwe9RlL22EewIvWFVv4C72+bJB0iYoSzsWW+977ZcVXmcZ/MiiAbXOtiHNJ4TGGz874AIaxTUpnk94+v/J5a98EkuyV0eXRXyMHRXPU8tmpdbhYJ/sP3z9+flVL4g3BSlLDP4giX/L/rjSYA/b5HHB2mOl+4caFqX2kYIzhpmNvqukaSWQJqmHnyFz32JLHcPXzV2Nz72ZtTvMsXhQbPVmZl/QvOW6KqxlAHvvi738qcSh3pJIvTe5Nxk94g7G7LhfXm7/JS4YNwUHjxwAIk+XMaHInudMV8xw9ZT02twmhcovU4OW05AnHjDtGqj0IDMZTAPj8PcBXZ+htuOYei5t4b7mKZhG6gbvddV0ws93sBpqsMIWuqTJPCi+TRaSnLR48fgwmTxiLvxvet4nVN1Sd5ywIKuFetQ8Bil4R2q6JuOuK+8qs5BN7i8gYLIdTMs+DQrlrAwPHWZ3DcGeLHvqmjjrey6gr+QYMjjYmx0EpMZ2Z43JB0RM+8Di7bvyC8g1WZYDPExfSe1pb4uNrutcg78vuuhRiGfu94laTrGk83mSoQmhcBzMt+O77VTGkat6EqbEV6ORdrW3dlUD7+1xDEvfD5L4MIHaRDDcStqdJhMOYtReiwhixFFEln1xTavB0Fj6ntb4IuG5offYf9OCx0dkBeRvKPYn/mlMsYO8JY3GJI4siA4Bq1ghNMoQAYHsyxnualS0LKbUen/n3MzOxpczCI2HRcIPiMY3lVGy1nrr/zfvxu7m/wyVPXZLaN61Xlve9dH0iUP/6c/vihlP3rbFm+0BSXb7o16IKYzbYMhuH6I9KwAw1HEDybqlSizA73gV3XbGxznm5mmLyfefg7zjP25R8tt643HUFYrY4zU4anxyAKLFD5iVzJTnGgfn3JweOu0Yr42Ly/eWNvyTlFFeqN/MySz5E8g4lJVfKPXKLKPYoURjYqlBj8pkYNCbWdgBgQcHsBupieauBzulYKnStwO4I2UY7OiafJ82z8rfS1rfaet2nd/k09q8kSUv6aFA6gwLYpuT7yl6yCy9HKCulsXF+8NwPnOcBE7tMrNkMWDUPrqcUdEyIrrG/M8pS3txdTmQ8GpMvUpw25Ztw07E34cbDrw+LRB/+6W2K8UxJrLCpN5sCgKKj1IEvT/4yLt2rH4kUxDgkrMkLZlyAu16/C5f+81Lcs/CeTNX8eEiirKcx0uharzKgYPwdlp9TKjrLDQRpdRFKAAA907YtXuc/IiVPl8WIrbWjGowDYJuSFduRTgy91LDv5VAmRuCcl9O8eKrcwqRvHx/Gv6wBZeX+9mkK2aR9lSqaipECUFoPa1ug6NzEXfIr41KyujSY3ozKqHS2h2gcG25nTU6XuUwJCjmAeYUClhEyCZ39RN1BOVTy9fJqLIvINkEPd78eemmEu1emuevaDDf7jY4UcdF3r8bkM4bRieq0wUOAFxpK4VxgiOlXRx3vVdSVfAPEZ/wZ+F1BjvFzXJRkgcI1lWsWb851lVCupCn5GgJhCUnKifhyAnQjq/bB5vJqiskXCw+GmHxqvQFjcYdihZfjAdhOqTFb6G+XK4BcN9MmdyqgfIu4JQQwZHkiWLBhgcbke7qhhKsHtxtj8tH78gzHahH+pAWXPNtez9POq0jiAdbQoAtEYXfayDBg+HQlwQodJqH7g3nLZ1NsxO5FBsULhSb4WbKYmXDQHQcZj7uQFkcoCAxuJbYHP/ZQ4MTrgcO+Ya1PuG6+tuG1mvopQ77/b/5lTvx3W0MBpbxyT2vfCJOx1DheKkEl7m/AA2zr25Yw+TjPtNkFIGUEtyFV35ZhsOtZQg2WDPJ3ViZfLwO+MWwI3jYIziqEUtSt2NRx9JijcVCnffyqmyQB25zjdNeNLoqNFkFgUPK9+yy9/uCILQ/JSv/JU7QyjjBIuHZW4g6fuJCFkEhF4piByZcFnJmzX/6dzKs2t191Be7OqIy+vbUFp4wagdlEKSHqcqmtHl36qPRb7daRXhgvL+v925QlroKBEgvrAxv/AhdiJRfneOqNtaQiXRVRtqxVcrgA+X+54EDEaoMqyZBRHQCw04fCEsXQsOtiL9FzKzb1oCASV0Tjdfri6XGypIJfwPtHvh9DS0OkLrWrYTp2wGaYMYbz9zsfe2Zw29XuTjwXg0Fj+dblmLtWVkpmSXpGA/DT921MvAE5XL8tu25/E+Fkl2TIeWUJEzJ+mlFc7CVW53z0wm0gAvT54Y21umI8cGTXHdZC47n6khwoDF8MDJyFIl3AIw+Jvu3AZuKyXOMY5ADuWJdkgh/fOh4fGxImSKtynmRap8r/ITtJdZx434nONoxJPQzlGNINBCrU99+r7m0s5WibrjFEz20mC2NChABOHdWJj44ZlZwzjC0efZu9QTXeg4o6XikUcHfnxDgOs6+tYJCYfipaxTTkmTn2L9YAACAASURBVGPymcAZtHlZns+poFfEyKaR+Mj4j6TWW0cd/+6oq6wHiCNoxrcItbLMeonwMfm2yXiaJXXE7rq5EtAnx39ghr/VALtDGoYg2bvKHbNG4oqVSDI/TXD0KAJmsOpZ+qiWSWP7qUw+KqSl805IX5iHLBlXA6TT3RkPpEDKX+scBgD49no9uLK4vwCAH/V4bdBNzhPkSmAV+/LLoCfZkGLyOfpsi8nXb0RCWFbxxNWqLdj3sMZh0bUGqxwZ4xq7yhHfbkdAtu4b2DHc4K5rg+cDB5zhLCKSMpjwxT2+6LzWNiZmvb3R3a/fHBj+b3OpHmyOgXPifSdKsffOf+L82CV21qpZeGnNS8brVNgUVHQ+So3J52BHCnDLd0GfGw0mnpXJN6tUwhNNjehlDL9bHSkULFZpsaHxPb8mJl/ey2tKSgAY0zIGy7Yus7vuGbrxytpXMgnK4vJyhSNPs1su/xcwz+7i/m6DgydrlvKcB1XXy4WZGgLCHZOPvhM1ZiO9yjQTS6qilN25cNd1rT82hk3IzkjOqcx8W/MLIxbfslwOB0TZ0Gth4Cd1y9cIGSTN5c/WtyxhJNSwId1ek7HcB8Z8IG6DA3hzjZJ91vDNqTGeBNRvigPwTbvmfij51PlM+pWTM5nGOPXPQKUH/NGzAbiTAMRjlwGbtvch57FI4Anv/5437tHKCgTGEQ9NwZKWYMQF4RrrfHJcedTiPRmUfK6+uEaXLfi/b6iPM7k/3bbA/o72XDAqT1K+T65I7YHyv0v5E5ZjRtdkFVlWqupQczb69sY8SjnyrBgzjt34TjlPjKe3nwQsJXEf87Jr9FPLn8Keg/d09mujEv+PjvecWOOF8n//M4APXx2fn71qNpZsWeKsPy05BYUtzqoJVUMtvdE6nAdD2diyCdnm+FNHdcZ/Z1UMxgSHyF13U6ULg4qDwJDsjz43qhPoeTP+nbghk1WUc6ztJsYYgpwwSPpmd92sTD6f+ZGankvGOfh5BAhSvSHqqOO9gDqTb4CoGB6haQF0ya2qxfsdL4mdEZ/J60w+apdMmGJyXYNLg42THgOLXYIpLszdA49kAlUtQ+qaZIrHpqsCdcwuFVPdAlR2CxWyTa61S3I5vFqRLYrhc0kTjKI6kS68bF77PDrZOrSiS+q9L1mA9fqLUbDYK9Y+o7ULQBNWTBD7uOSdZFPySa4/EdZv6+3Xhg4AUA6FpM0uCgxt37GJtrkJCiHdFNONuoXkVMWLzSW0VVgeB6bolBNv6MJJlXNdGWTb9GXYDN726m3G43sM3gPfOvBb1uskUZ98tD1l+Xk7xULTqTOmA/ueZiyuJteYs2ZO/P5ue/U2q/JOhUvJlxnv+yDe2fYOHlr0UOZLTIk3+gizN2viDfOWmIyRoIrrZl+HVV2rJHddwaTJAhvb8RdH/wIzTpmR2jfBQpne1IjP/e1zVmV7WFYeCOUgQI4y+X7/QWDOHfbODp8c/WF+frUkUbEhdlNSFyiVeeupihrunJ+okk+sP04lH2k/Yejz1MF7DX8JVbjXHxuHmCFIXUuvb2/Tjglm4EBj+Kll8xk2rq8TN+GsTCe65gfKurEhN1QtDoB6JYT91BioNTD5pHisCrtVbrQ/WWNDGHwn7AZKPw8UW+JrehyxXRn5yqS4sYa6xfcQh5m03asyT9cSd7U/YGofYiafPhem9cWqdKFmbYfxR/ymY3KoJexMf+UsYwKdlKrUuUl4vIj/bfMLlYEzueumFfjIT61Gq1JeNbrbWVuinaqIybdUSeySK+KNjW+Ac46eSg/OfexcHH330dZEX5wBbXmzQYCDrPGiP7scBxSTOG3/fOefxmulesj8xw3HKLwa3HUDom4XENLuzr4SS84yTpiqKK8RpmtN43tUazgHru/bEifysX9z4Yij397U16bi2HuONZYfvy0yFvsFXDTjInz/ue9L55ltX6Iy+eK9hdI7v4AgCLSY8HXU8V5EXck3QFQNorctuK0NqpKvFBD3H3EwV9IUGCZlj7pUNpEFjU6yOeYbs+sOZlvh//cFUdskJp+hrbA9XSTgSJh2NmH4YuIma5v839cmM4aogsUkTp04ZiQ+s01mC/1+eDd6tI2dWXgLWLp4c8KYkThv+FCU0Kct3EOxETux5VDdda3jQdUF1bj8Bix5228U8rhLCRq7yfMwvakxbobqKL5+50uoguOGtkHYXGussXmh1X9rZiWf/dw3njC7qopxZgo2PH/FlvjvQk7pQ8XCSPr4r9ydNODiAy7GN/f/pnQsLfFGV28FTQWinObcvkkbgFtXGmvCNMJ/8Y83sNsV05V6HJX06uxUjHt/ZtfMvqBPUuBktYymWeIB4PwP7Qx0rTef/NpzQGMHvvi3L+Lb//y2tQ4xn8Qu2Mpt9ZSrmDpzafw7q7tu2lc8Z+0c3Dr/Vlz57JUJk4/5eGWdzgq3Ie/ljWOgMdeIwQ32+Jj0Jn8/qFUKV5AFHByVKkc+y5zRMRE442GgxZ3YZSAMIAFbb7gyR9284HbtQu1Wdj4u/lNi8invX1LyxUxCQ0y+DI/qZazDWt93G2s4MHnCWPy2TWapmshkKlbkdaeNXHSVUABsyekJnbJANWBmcQ95uUhdP7N9V3TN593yt18RrX5DXv/j+Tq6IT28g1vJN6qZuKUp45SDwTd1/WPXW6+xQx1b5DoHw/eCJy6I542eqkPJR8ZuhWbXjf6fuWomadtkpjTIJ8omOIuS708f+VNqGRekNoRS07CuBDyQ3h1ADIf0YJOsHJYSb6SMy+zJ3/oH9boNXnqaBnFelBRKQfG/LdZmPBpY+tfY47eAt4zQjne2Esap51sNR4UcA5Y8K7X99DtPG0oSpb4lu+4rmxbh5AdPxq3zb3UquWlbKiOOkf81JZ+isDdl01Vx84dvxl4dk6Rjunou6g15Dmkw1dEb9bekGCht44SlfKVpJGojgcVw7JCxYQiBjX1b0F5qh2t346uEFgDPr5CTFXU2JazCXTY9Gf7h5fCPJf/Q6jPdAgc0edtjZHfByb4hV0SVV1ND89RRx3sBdSXfAFHhBiWfYZZxCQR9BgFP1OED+E3bIBzaaF9cGBJlmtp2OJHpDJUyr+CZRjd7zOgOqgit63I+tikbqQDJBmcgHI3Opk7c87HEjYQGPs/qEj29uQkzUrKdUStmFoHsmcYGlKFvyGYWz8OjxSRBAqfSgwGqqOG6JU7OCyVllSj5Lho+FFeTANIAcMG4nfCtYUPQ54eJFqiQtKGrD7MaK7ipfRB+0dHuaNkOEwPEBFuxq5+/2nIm2VwzwxS1rSdRT2vMjDRBrwaX5dP3PF3LXmtS7NH3uLWngpYS2eZOvwx47teWviR939SzCZc8dQk2mxRrNcL0vQPALx/Tsx9m2oCMPrDffaFK0YKXTcn3xT3dbsgThzRhZFsDcM8Z5gKRMWRNt50Zd9aks/CFPb6AKbtMMbg9h0/lobkrpaMZddpJeanK5Ekv2rwoOsTjWHhqaII05L28kf1QSz2vFLO9j9itOfpdqSpMPhO+8kSocBl3aGr9Qebtsoyv7f017RhX61IE9V+9ImfzjNlBVFHw2anAVeF3aGLymSDebo6415cNLGQXfPBMBsL/VFh5LHPwCqW9mMknWD7929T05+01BjQ4vQyPmWu8kiR3Cta9KZ3zuJxxUUCdr6uB0pqByUddBweXkjZpgPc41rBpPdnnc0l7DoasEdpQ4UZF5OJ8Dg++9SAeW/pYfCxNyUGZfEn8UX1sxkw+JZu01rl+uOuObhlt6VvUZi0asZs/GP5vcdfdqW0n7Xh8XvzRKBs5bNl1ucUAvIEsCnoCvRD9m93C3lAcNW506kyy3vfxTs5HPpoTxMhJG4VUOZRmbGYIEBiSXeRzpL+W9QkAdsFSYPksTG9qxOv5PGatn497F5rDPQhzedWSXXddOZyn/7X6X04lN8U2EpoiHOuJZO17DNi+IQn7oiz6tqQ8FAd2Hojv7/9t0n8bA44DT/88U58BEU5IfgZijdGVfObBmLa6WJme8Vrmln85OE49aCy8yFDSxyso+aWUdsV8Yx53F+5/oXl3lCF2c9Ivpsn+dD3f6Z1k/PX5OazvWV9T+JQ66vh3RV3JN0BUjEy+2qAy+VZ6PLauMw7c2D4IW53SDzcmdgDCWE87IhZb7AZgEOSu65A3HQHTtx0m23DMeLN0L+/lpU0rZfLVwpbUFgjL88iq5AvrDK2LFIJ6rzP5QqjPRBonGbKaJaJIdH2KRLxctMj0xBuMMfRG1ztC7hsxo6EBkyeMxeJ8tkXWFvPqrtfvSr3WdGk3cTktUGXDwkeB6yfpFwBAR8QK3fWjqW0CwBWHXAHAnWgjsQIm7+HZt9YjR+kdM2+0N0KYEL+e82s8vPhhHP7nwzP1T4Xq4io9toEypT70/X5fSr+9LAk1gHRlSuwOvfhJc6EhYRB327v720l/wwX7X4CmfBOufP+VaC40a30FICtr4c7CSqGFWoh7HkIkIMn7+dhd3eai3GRxLbLdW5qLiRgKXOuf6xpZyddnyq6ronVkxtrtyXdqAVVsS8qGDGxZjzGggRhI6OaevDex/ogNjzz9hsfaX/9zfMQWiN8Gxt3ueLZ1suz3oT98oYTJJ+pP1qD+JgtAxp40kXekvn0/gwTFlRiSvhKnSYWQgSqqki+FyVci8fBMcSvT3O3UWGNHjT7K0r+4Q+FvqY/68zhp1Ah852k5w3a3EmvMBI7oGcTChF63p2wLku9fKVuUPQeyKLNtpkzWH1Z7T5TN1/DOe6o92rxy3Pjjoj4QKIpKypWTlRfJ7xFeGPMyAMM1xLCqJdAz1FMLLNwvJz46ZiSOJ0kRqg7lDBB6e8wpFkDjR6e9CZ9XDOFylDXBz1vd3tuDDQDCpHdTRo9AD0nQMWnwJHx58pcByAzlFxZvwJYevb6mfDgGu8pd2F7ZntLzsL7tXKmH3EuOceCnE4AHzo3OyetpdqW9uvpnHwU2V+OqYV4sR0rIIlnX0+AqY0toGF9jOE3rY4iY8dFc2Vctx3KfbQwG0dOJ9z+5BqkdD5703ON6rOFfLIpIZY6hRkI6S99ZCt+5TfFcRx3vJdSVfAOEKSZf1syvAr3Kgvn/Wj08EGXVSyfoA2B6UgYBn/mxoHnHoJaa+kV7lViI9P50GQQ0NXOvWcnnRs7LWd0js7LIAN3l87GcEpdMMBkYsy5EKhi4dbVU34X4/8WSnJ1OZ/JlEwREqQ0Fu1A/u1TE6ij0u3iCVEnhMaAabVByNUqhf2sOA6pfMNwcB0mFqgytDYJxkAg4VMlXpO66d5xsr6ZjAnDZMuDAL+OGl27ArFV6BmyByw66DKfsegqAMHGNiTEUdk22nq7YFL6P5xdtsPdDup4orQeo6LjzhDsdiTjszz+VfbHbicD4w/rdL5q1NUtMvvs+fp/1nNhPO91mT58GFKK507JxpOwcF9RHk+quu1uYdS923ZxwFHCkPW5ijuXioNE2Bp5QNquwGW7SmHz0lrIu/uqcH2bXTXkW0bN/Z9s7eHZClAV42O7Goq6EARTPnPoM/jFFd89RIelxUumXUUy+jgnJIcuzLebk+du0PlKUrauypSfMXdJ2rur3ZqZAdTGGLZGSXI3JJ2Vwz1Rb7WUFXEy+1xrKOGH0CPQ6hligZINmYhPoGP8chrWoe5NWjion6HdGlXziqEYuahsn/VSVAraxzpSxIskCgb42qPIV4FbyxYwlDry8bFOcPAs80F0lmfxHYuBVKm0aFv+5ums1rnjaPFdR2IwT8d073rn1VEPiiSDmvz0G76GtqfsM28fQIVXJl1xDr+ekbztF8R/Vvtq+gwx5ZMzXGY7VKiXETD5LH74wcjhOG9mZuPMifc/hBWWjnDyCkYReXh6besNv63O7hcxWMcY9xdzdk0/m1WuPuhbn73d+9Ctpo6uviiXrdSWeUCB1lbsyu+t2W7NncxRY1LeNi6P7UBM11fYyVUO/2h7FZs/Dcfcch1+++EtjXSZ33UrUPTU1j+0NMsc5wD5OXPehe48xoFoODQq8gryXdz61ChQSAvMkQwNTMu3G/bdkb7Z62yjy4BmTzkj6QOf5Wl026qjj3xj10TxAmGLymbNiActyPuYXdNcCk8X/7SgwddYXJKpQGW4e87C9HC6Of1VitqXXSRkR2a8LkEyatstogF/bopP38lLyDSnxRg0dysrAqSI7gyFtsQQ5L/5/rFHNNpg0FrB08UEt8fcRSy0lZXZlHBeRWscYw4sNEYvIcScdBUuG1RrQU65dgSWUT56y0QCA7r5ESExlFFGUWgHGcNPcm3Dm38+0FqPKKMYYztnnHGM59X1s2h4Kj189cmK2/kSMMxVXPXtV6qWq4mXXjl1x1uSzlDLpgzl1gy6E3HNmAl95PLU+FTOWzYj/NlnB7zzhTul3qqWcpSTAIJt404ZybMtYNObNWT8TNlhYR29F7ktqdt2T/xBdH9WXbwSKLVq/BPJ+Po7xY1OAmpR241vHS/2cMChRUKXFkZk0Mvme076cL+z+BQC6UrRS5cilCcJRP06870T8vzdvD91f28Zgc+9mnPrQqVi2ZVlcNKuCu7XQKsXmkZoTf3AuuWSytLg6LEpCQNlAlvWirai6yZrZaCt9Hxs9j7jrZkNV50pp5wVeLMoKx0O9+ZnaOHTcaBw2bgyAhMlnctetJVmALcatXi6BS0H6pyFVLM3nsTznUNgRN9sLc/fgpI3/Ff6wKPlEe5Wq0tofPx7/2cMYbpt/G3pJhnYGht07QuW0msWRA1izNTm2rHEP4KxHpDKqUq/C3QrteO4gf7li8lGkJd4AEkVBX1+k6OAcf134V6mszuRTexeBKMavm32dMzxC3A/GcPEBF+NjEz9m7J8a+kUuQ0AzIBMlX0cxZNZ5zJPmFcEOAxRlpYPJV1WVfKJMJIeqb8XOguqfls8ki9pi6qlIwvckDD0TluTV5CkMvmO8eQjgIUCgGdA4BjMSasTzYyXfF3b/Al45/RVMKR0AAMgpSj6qnDatd5WKXSYQTPhyUM7EZAWAHuWblLLrqs9Xzcae8vwfPulhAPKe48cd7bjrnSfi33sWiSGgZSQw8WgAwLMNJazoWoGHFz9srLtg2GuWI5mkOGgsgGTv2R9jEQDMLJVwR6u+T1T3MgCw2jfsfcFjJp94ymnGXT9Oj0jXbUpI8CxKvoLFqGlW8k3vWamX5GF4CEoaKQwgVnYddfy7oT6aB4gzc9O1Y6YliQP46JhR+OwoeZOyoJDHHwfp8S0E6HSlLr3UscDGGeDcnT3QhSwx+dRyomwlPif/n5RJmGscwIiyvhDk/bwUh48q+WqJdEMXcfUZVhjD86VS3I/sdXKrF6TpeQA6k5KW2m9oCT2efQMwhq2OLxDCn+focIF0zpRd12PAnMZIyedgc9UaK0yA1qgqS2w4eszR5PqI0WDYKEvuumriDWNnauOaZM0Cq0Jk/T1y54jhWLFk+hU45nvxn1Rpp7oJNOTSMy+rkL43x/2/f2IKq00IucN2A0btX3M/KEzsSXVuKvpFrQwpDQ6gmHe98+ReTUw+lyuw2pfF69KDbMuV+1IP5Hg/huLw8MbGN8JrLO9I/f72Hro3pn1qmnTNdw/+bnw+ba5vbcjFffRSvotvHvBN3Pzhm3FA5wHxMY5wnKfm3YjmbVXJ8eiSRzFv/TxcPfPqWClRc9wyBRw8XgE5uMTWmrtiW+r1HmOZ5ohBisHDpqj68NhROHLcaC0MRxoC5jZe0ZhfdCN2yIT2zAwf6mYnGNwVoQCqgcku1yn/zqLko/1VjaLiC3V5RAREyXN+7q9o5NG36ggJYGTyEdzY1oqfzf4ZLphxQXzM93wcPioMoSDH5BP/J/W93no40CLLd6rLIs0mf+4+5+LHh/9YqTHEpFFirHFgTTYFrismmTrvxYoWHmiGATGHqBl4pR4WB8WMaSC7nOAzH6fveTp+fMSPpeNZ3XXjp/0IYQ02DcWqrlV4dMmjcXzPalCNFQMn73wyYYeJeoQQpYa5oIp7M9vU1lfbyDp/eG3JjVz19Yc1C6R7F4k7TWPy5SPJ3qSALFKB1M9jY0/I7GsrycaRghIghiqnJc+dqM89lbB3lx6/m9zgJ34jrR2qEt4Mhu6gL44PTPcGnAF5T5lJCZNvVdeqVLZgEnMyqffOQS3YWN4WJtc6/RV8sHmfpITnA62he3WPY7248v1XYi8vTHZC345Q8pWG7q6dM4HJXdPwr4YSfjK4QztuUvItzQtZQq6QRUw+YeQKE67Z/ZTGoDlaA+ieRVbyGWUEL2+U1Y0yEAO+Fawwtu+jKrna5+oJN+r4P4S6ku9dgDk4qXlmPWXUCLzQoJKt6XUJbFuhUDQPS6qLb4Cgf66Ar9yDSd7i+GeZMWz0PONmVB1ElMnncmGgXe2o5HDSzidJZfJeXhJAJSZfDXsROumbnuHWiA1WZW4mhVSnspmUITMU9fglkM6LtjcUzG4EAPC53BNoZdul6yZvti/GRSkulaGf6mbvM7cD+52uFcv3c8GjS3pvJdtTlQQuoeQzfDc9ZQuTb8whqW1kdQ3MgmRchX3d1hu+v2YRy+3qYYarCMiG1OU2+7OjfqYdM5XXsz+mo5RPeb8N/UvKkhWUNXLWpLMwftB4a1lxfy8t1V3sTDCx2lxKvrgdAG+s3orrHw0Tlew9OiObVambgQHt48EBnLZ5Fh5b8ph0/uG3H46VfDaGjyrEUpfBEU2h0N9abMVX9/oqAKDBT0umRIRnZ8lw437IiOSbYmBxTLN/Ld1ouywubTwafTPPrngWlz51KfqqfdjQk9G9nUDE1zK1Rj+DrX3pX4HnIRNbSsRuNLZpKB8z9Fm2iEyUSd5gcNGkSkPqXnTY+KZ+uQSK1FHCRYv3UxzUDIuWcoH0N53r5c770Qt03ZNVptGUNrQdQ+INAlO2+O+///ux4aFclddntX8Vw4ZTXW/oZvXsvc/Gx94nM9o4OAo5D7sMJ2PtoQutfaZIY/LJ4zX6dfyPtTkxLqnIcNI2/ZznpGsGHG81ZfzedeJdch+Wv5CcbOnEWX8/CxfOuDCeHyu8ggAB9h++P6469CqtvvhOFHfMT+9P4tlpY4xF/1qYfJZ7mFd0Ga5qg230qkmt1K5QedlUR8z4YzCklCPtRAo6o6RKx7qXw6beTcixHJrzcsxbj9mZfOqazQH0REbipqKynu/ykVhp/uamN/Gj539k7TetrzsoS7E2pbjBqqwc9YdzjmPvORZPLHsCWWCSW9XvCuAhIzUag2rYJoEL978QU3aZYqxTMNBKsSHY/SEx9F9RDOVaodNV6/MYA565Plac5byctVe/OeY3yXUWJh8DQ2BYDzdXe4zszVqWQgagGd2xocvjHIW6WqSO/0Ooj+Z3AVkzv2aBqgiyQZxRp8KA91PJd+9ZONB7I/75raFDcOS40TAtEVqvWMLkc8V4SJh85kKakg+UyadfM9jCGKMZWp2BzZF9AWTgOPQnbvdFsZFxKTopKpasggI/yN8a13e09xJyXP98RQ1F6rImFMBk90uZODlwgPko8wAHjpOz3/WX1Sb1Sd1YtYxIvSYO9q0o0gCgXE2eU95nwN+/A/zrNpiCqKuwBYOmyLphUUf+1ijrb3MxZ8zaaEM1qDqD/AohOQ2U/ci4zPRdaohnkwmWOGo7CnTTd9SYo5xlM8V2jMpMvm0ytpb1THgup3h67vRbkk3kf51xEGZ/90PpbTM1GyWAPT6OyukPYU73Slz05EXWS1Vhddf2XQFAClcAyGydKw+9Ej898qfYrWM3nLfPeZj7xbk1jF23MDymZYzz2jMPm+Cu3LLmrOxKXGaeWPYEznvsPLy6/lV3XQrmfnEurj3yWnOzivHlwIluFg1DEM4xKWvkiRNPdLtCGx5mrRnmAwfbAQCayDxKZQHesyUTA2+bIj+ImHw3t4VK7P7G5FORxuQrA+jx7G2JGWGjw3WT2+ZxLycpPNpLoZFCfNta4g0C1iAbzXZu3xnDm4ZHTBRz4g36RKvRt8o5x4L1C3DeY+dh3rp5UvmqZV2gryYIuJNNZUPWxBtAyF7BpCnAgV/WysTZdTVDLenToFHSNTY5ob0oG4lsSr60YCV7DN5D7oH4FqN3s7Z7LQBgS98WAOFzrgZVY3uSh4rCQBzalNwHnUeobChCAKhvqJYQMllgVMZZyqoJl9SZhF43zxAyiLZpu4unGkoxk69i6N3SNcRdd+iu2NizEW2lNpKtOUROkdPomkbnWKGQ6ovCvRRVrw0/LxnHlm61h6+R2iNKPjruGDhyqntM1FcxrgYC9TkAALo3xIYJk5Lv5J1PxpmT5NAy0p4wio1byjdI52yzR/hM+z9O6bVd0RwuJxCM9hZ922LmuZgbTEYbKTSOpdcqk0+UeqvLzMyzuevaMIRtiZWlHoBc3V23jv9DqI/mdwF3k9h3gp1GJ7g/tWZPgEEnJxcHKVGFyBNcwAMj1XloURZo00TKuVHSiNdWmjbOMgKwmGnQbZkwTcKKKgDnvJwkvEiMPMOCMbhqE6Apk8++wFWB7DFPlCcmuXOIY0w/Z7sGAKopSr6kPoZbC9caaxY1FCQBNVKmkuKe9EwAeD428wp6lI1VzpCts1YRQVLO5BqAs9KD58f9FBktJSWfYvF77tfAtG/EGb1cyKLka8ln+z7VjYlQ8rWWckCv/p1I6Nwr/jNtc2baQNWSrY3zAEdeq1ugzz7qfcmPjUvMF7enKHMGCDX2irPsDm7PXohjO4n72N6Yx5DmDGwMZVMcx24aGyaecBlbRNxUAXGt6gZHyzXlm/CRCR+JmmYZwzIkfXQ9bTVmloCYgyYMMWf9xQnXAXt+ClBctARufDnJNs3B8dzK54zlXBD3KmIRhjEKk3un882+Y9OTrHgWJR91Lx/SoCsL6fMzfY19zLYqm1FlhP1tON9A+khlgaB3WZjyXgAAIABJREFUs3M22KU9jP35k8GywiWnMX9Jdt0M/Y3bV37b7jZUIAT45OgROJ8kbdKUfFG/vtZpZ0Lzah+mNTfK8aOYB3iexNK5aH+iWGcphoLdTpB+/vaY3wIIZbhDRx6K0/eUme4cwMn7jiK/wxFxx4I7cMpDp+DJ5U/ikqcuka5Ji8kHHiqpE4J69lmvFkVEgVViBZnKAFfXtYRVb392NiWfOifZ5/j0+2TRP52tJcKaDq9T1+wqD911be3F35kyv1JfDikmHzPM68q9DSQjNcWe23OY2GdObmF7AzobUwbta5qh2/aGzu0cRpR8OkqMyF8dE7G5d7MW4gAAfKbEqSQMQFN2+J5yeF7zOvALVqW5DRzA9mofGnMkLi95NhqTr2sdAGD19tU1tUPbi5tR3sph/rxQZl35MgDZ8CGgKj1VCAdlwTZOlQp5hjLmy8L/SSfOtczPInaxMHLlvbyVlk3z6tKs6qq7bqB8iwDAfLOy2mQwcCk9gSRkRY5z5KNvaWTTSMtVddTx3kFdyfcu4G/NyQbINOH8dHB2F7hsTD4eC8aq8svG5PvePl+XfpviQcwq6RvbPz63GO0WZVrcJhIaeU9svTJNs+omQ/6tMvno318YmcS+WZbzsdnzrMILfW6uzUtNLsDKb3kxtysAXccrkYCxN3sTH/L+lVreVScVXVmkPLQx+YZUg9AybhgDuRqYfDMaGnBW57DQ6k06O6RvefJj0klAm5slBACf3OmTAID92iZp58rVAA35/8/ed0fZUVzpf9XdL0zOGqWRRhmBskAgRBAiiCxABBMMBhtjHFjDwg/Mrm1sME7rgFkyi7ENNmDABBNsYRNNsEUQAyYIsMjKWZqZl+r3R3d1V+zw3gwGzvvO0dGb7uqq6u7qqlv3fvdeG8u+dYB4Yv2KyHp1bAwZTZlkyUbYuGUx+eqzDtBv2HCd/zZwwTuCojMqJplt2Zg3Qsxwu/vw3aP75b3Oe5fpLZ57T+SyI/9Ck3kQAMbvG9lOuWjNtgrs3KikEeHbTAYqzHfjm8dj/9H7BzWEsaGZko4GcR9P3HVUcA2lwNNXmy73wVpn1/EbEZOib3tBVPKxaxUlX6FMRiarl/stx+Rb0LXA/61TIvPXZk1xEUftDhxzY4ystmZcPO9iIQ7lkJoh+Pn8nyvlDuw+EL875Hc4eMzB/jFKqZBd1+ECy5vmfotAUfItXblUSM6jc4/vLPDsAnVcFbizcedu1gtdiAfeRVfIrvrS7aHrGotVtVk24EjlSty3uDQkhIgM5d4MnxglwK3pi/GOFOz/lUwab6W4BFve/7oMsn5fizlc2NEuxo/y1ir+G2NJdvzNnJx4g8NyiQnEvr2mTBOu2f8adNQG8yWrbySCuZUlI3hx7Yv+MdmoJCdvCeoLWMAl6sqNr6dS6E+wJZfd3j8/JUjEJLPyUij44SLk9SdgHHlMHe+DKjWILH8eJgaxLP+a5t84WUtZiUfOmx8o+bz65KQfhVIBL6x5QXGx5usBVCUfk4uKAO5+6x7/uMDk48qPyAdfeWWRRQPUlMxeJSYjdJSBjO+bHTKkSpyiX4cUYe66bsnf7v9L/9wZu4seGv3FfjEjOTNcIUTJx8sAXozQ/oLbn4yc0d1OhcpOM4fMVI5RAL2lnK8UE8cjhS0r+ca7DP7+kPjKTFblESTxChC8I/f/na3X3D9X9rhtaN6t2D+1zn6vv1lbjCtuCnVAEpmHA4QRFlRTkQuWXZ7NDbprHRK48gaZranQS0KIlsknu9r75TWyR9QejO2bXUOCe/3P91Fljiqq+KShquQbBHx+o7rBD8saFgZ+82CyAVNCcUuzK8zq3HVla9fQ2k7YkvLmjZQqpF3UrsZ807mR6BJAMMZcryFCO+XqYi4C8gY4LCYfj4O7RuDwkcOM7hJFPmFHyMYhKjgxQ6ZU4hYkF7KSbysJRCXTBkxh8nkP8u7Mt3B9+ifG9qnmF4Mp/h8gxsYjIBjX75bN0pInLKv1pTTWMlML/9HZjr/XZFGEmGzjv986KcbVImYMmYGeU3owNNuhnMsXS6hN22iqlcZs/yalrAxTTL4bFt7g/57WMU1bRob8qLf05WFbBDUpG+gzKPlqmoFsE5AKNtFR7ELHcnD1foFyaWT9SCWQuNI37vfNTweb113HBN/03HEey+mpK7VMJreiAaInaPCnxX8SvukoJV+snrSMEd7x+r71GFIbWJzjbCYBIOd9K8IXsfqfwIPnR14rb8R4AdU0/m5+5Wbhb6b81DEbBgryk2CujUC0WirjGPoVstnUbbh1GNs0Fn8/8e84Y9oZAICjJx6NfUerymZCCKa0T3GZfdxxXiHncOtP8bDLNf3VM/k+MLoCBdi5L3Az0309YXOxDkUQ/xrd089zd8kn4Yhi3fmxsKTjtqS4VDNmxoPcV9PIKYFgDtvYcjirswOLRjLWBA2NCcZwszz3A0alFd+vMCbfc6ufE/6Ocn2nAI55K0h6kyq5jGzTN37E+CNw+tTTI+p0+9dPt2PxyGG4yECY1eHZVaJhcPHExZg1ZBYAdwzwo9FV8umZMIHhxTNSeM+s1Ooxv7+5TrkmLpPPNMcn+VSyKTvY4EssH4aeta7iRH6nDL7S4nUxkynx7rUnIz4bXg5niTfyACbkclyZgYHl5UDWersYrolaO/m6wlzB5VjZMlhMvgKl2H347ujk1tZaS1Xeie6YLrJUjB3Jf6+C5473P5MlR7/7B7Ezlh3KjJWzwjP0FvsFI5IFptQGUrw3TeMIX07jk+7IuHjexepBIrLZ3EMyQ9aDN8/oiBZRyts+SgWPpwIh+GFrM1Zo9nMAsMpx8GqIu7bSN+lv/Ton/t3a9y4A4LFa9xln7Izveg0A0+rdTMBnTj/TT+pFARCmcJ9+vCATExB9dt0EslHYt+l+a8EaydZBOQZvFVV8ElFV8g0CyomnEgcmBZVoixKhY/L1FfqUzeMJUtZfwJS9Vd3+6eJ/MCbfdkNcIl3VipLPTonZtkIkwfW2bdzs8M8tzNpqih/YmhWVnVP6c4pQyt/PK1mKud1d2GDrY7cE14iVzLRfQQZmYUKx0mkSauifgRrT5O8r1vut/7G+Dret1LvNbUnAHGL1UYQIisTdhB91z1GaUxpLpsZ6WShSOLJVNxTB1X9f+XdtCWbZndQySStY/fqgX+Mbc74R2srWvgLqM447TresVAucfLf2uqhkIHJcts66zljZDH3mATdOmmpc4W9ONzemnzGw0z57V2QblcCxnLIzf2ObutHE+W8DrWMEAXGPEXugs7bT/ztO4g0jdCzQyYcbi/tMGG5ei+MuDkDIKs7w1RlfxS8X/lJTugwQdfHnMxtPbZ+quwRsVNWko5V8vFKPUpqYhcieW5J3RmlRSK7Au1/lxy9Qy4O6c7n0DcqMX53SM0q5xtaaadZbWuWWUh/h5njN+VWcYpXfPFLEY6HLT/H3jeImptxYTbJSOy5zXQc3a3305u2WRs0GzApi4ulqBsITb8gIi0ersk9c9SRgns8vnnexkKlaV+EDPe66UbLc8fd8ecntAbhZ2UfUu+7E8jybQtFX8rFYZr856DcAgJN3Olkoy2LeO6TkXmOrz9mUOMqSRp3JyBLL+CJYUn0hKLHLpiAvc38QSmF584UuCQtrnimFLhzSjkfqArfPMANyEvj7B011eUMbUe66fN/Cvi4KrVjpI4UCVtk2Xtz+Pp7+8GnhXFbKTJsv5bVyysK+B4W/l7wdeDXI90EJ0O+xJRu3/kupK+zd62QLSoC+Uk5Q8vErGy9W9lk27n3zXlBKY3mAiI27//HsvCDxiFTWe0Y6Jp9Oecu/nk00j+ZMs1/3OykHNzU14gyDK+03O9rwtaGq4VwGP4MRSkPXBpnJd+CKHwAIPNbqU/XCgCTEwtxhc/HlGV+GRSyfO2pZFlA/FDj4x0JYEjnkVKDkiz85mggg7Kgfk48G30q0Z0kVVXz8UVXyDQLEANaVKfx4IcS8nPHcLnEy08XkWzhqv1hKAp1FfWNvXqPkE7HScfBs1rWAbTMwGEskcAvwNzaSoJiyUkI/oybdFWm9UP6KY/mLVthmyMTkS0sW79ZSCerSFuD5OulcjOy6gLvIn+ncoy3rlveUBt7ftmZEaEebJ7UdOVMMls169Vw2i4vfvBU6qXLF5reN/TH3MwwEa3vXYvmG5f6RKW2qS656lVtrX76IV1ZugZOEGdvuJjF4dtWz+K8n/ktbhI0tk0Jh5pCZOGHyCWKfpLJ9+VKgSN2oeW6j5vo/v/PUd3Di/ScCiFb8yAr5sEy8ft8Mx0e11uLaz87GtSfPDg6a5oKOSZHtVAKLWGLG7IjkBwLXdFWPWiDrulnzyqUd23YUlHxRm0lXoA3rgYTjfqMc8jeD3r3x829cNhvbHPPP5IzpZ2BSa2XvRAw0LoJ3rdpr5F7a69m9+fGR5E0WN1av7bnW/12kxViJAfg+zhnmxjLcZegu0dfwIRm4F2hxO9aiQS1nEQIUvRXCc+9kLq4Mc4fPlS8T1to/1quUK7Zpi5v59rqmRs5hScVfOIUCfyc01MEuAJHmjeVpma0UPacOrR0quL+71yH076jjPAiALaiNLKeFN4+FMfmMSr45X1QORSWdoggUewDwTPtRoe3Hwbpt7rgzEWUBALufFauuGqdGdHvjzqU5d91X178KAJjeMR09p/Qo31uvp2SxCcASXhx595GY9utp+OHffwgAeGHNC8I1Nx54IwCzK7CCmIp8/244d924hhNdPfLmnyltebl1Vnao965ZmXjyXLmwQYUYgDxM5kDFLVo6z88XYUq8UAMtXCXfq2meMRuUzpDgPWzPb8fSVUuxifOu4A22fBfe3/q+/1uU8d15rc9LvGE76vcY9q2ZYrP1FnMY1eiyyQ4de6hwwylOUfmLWoILn7gQT33wlDLGrtn/GuHve4+411eS8+AVd4Gs4ymR2FuxHOC0P6FvR9VgGGXgWi8p+QZqDPKhEsJiz17Q0absOUdvFhnFDekG4doiLWk9FAgtANlGwLKxNb/VP25UsBK93Eo061hYMkxKgr2fDQqmXqwq+ar4NKCq5BsECAJ4yIb84doa4zn/eu53wThRBaVkJZabbVDc3Jw8+QTYMZR8usERJ6bD1S1NeMaL6bPdsrSCiSJMUKAkbcIcywll8sUNDn5DDXBZixsLJ8wl1/R8ZWG/BJW9yP/VUCxpz7VKsQzlzV+OEHRig7F/MsvD0Sj5SgByUn8szwXBtgjSNvdWy5QITE+QHQ91U5MyZQHqexfLM+HF7ezXb3kBy97dGC/TKsOYPQEA7255N7JoOcwy1rdCiWIs+RD4/anAdg3TjBvLt79+O15c48ZuWnzP4vA+SU/cJNgyoaQ12ypcwY/VtGPhgJ2GormW2+Cb5gI7RrKJCkAkJ7LIoPQe/vuQyfrsxcxFhhPIs3YWnXWdalljnwDBaCIMs3hjzmdQ+uyhoK9hbj88/ODu5WRGj9eCMi+zuD5xMmrXMiXftjXiCe77WbktYLNSSpXkIjrsPXJv7NC2AwBgt2G7YelJSzGrc1bkdX47ENfcNEdH12WEJGyLUioAOy4C/svN/itvLPYYsYcf+2/H/n6/LYbLWtVYa2Gutzr8ub7OvybK1VfOhBuW2TPunBann7lSTmXbhNTDxyyLM5IJKIi0MJ5cO1YpJ7sauwfVmHxBvRFKvoPVbM1hmzyfte49i/doO4red5OUWSbW6EJYp2UcoHEP1CBjZxAkzSBcG9SNrSYZL+VxIq87Nin5a9gbG98AANz0yk244PELhCQ1ALBDq/sNs285CrHyIXl9d/+w/P+TKlUFJt/ME/3fFgDijZ3XeYMx8bh1hF2v7+xAxeSzvLf2XEZdf01Mvqh1gu9bmNHhnCHtxjYAVzmc0Xx7lMBPygEAN758I4BgnLjgDDGG+gV53/t/+Wo3iZntqG6mSb+1HHF3MJ21nVh60lIlcy3P/F7nUe7W9a1TjHMzOsQYxt1N3ZgxJDjmr/0Cg02vUIadAkbthn7N/YnPQ11PNpU8JR+L/6vUUB4EJh/3W35v92mMWzLSdlr4Zoq0pMytFASlYtE3ImzLb/PPqTKT9xwkReFxk47D1ftdrV3vTPs+dpS9JwvAD0qu3DKYoVKqqOKjQlXJNwiIK9Sf1RlNm+YtinFixslt65h8hFiwDVYQHroAve4SqbcamlzJtlqW1nooM/kiY/JJwzXJgrYs6y6gYaKQ6fnKrMcSVJdsXnCi0s2WuP+PG3+kf/zUYaLygULMMCVDfoQpoipFVjs2Zo8ZhQeExZdiLPkAzT/qwFX2j7jD5Wn5wizBQMRmjhCFTRcmoMoC9Z//6S7A2/oNCqFDfhr8PvJa4NjAumoKnnzO7HMC18AKpsQSpfhG4Urg5TuBFU+oBQybxiiGk5Lt0PDeWrItuGjuRbhqPzcrZOy3a1LyaYROHt9+8tu4vuf6uK0IuHzB5SBEjLUST6FFMb6FAL1mZTiv5KtxavwNJ+Ay+6JbMCDOhmLRlaCjxaQo/Py7NbdVvsIHz54bNCWfN5TosOkoTD5UOMWykpraZKPw2J1H+hn0ICvuDMzMZ1Y+o7DjeLRkWrB4wmL8777/KygZM7EVzezGgsQbFxy0A7I2v7nQXzkk9y6w9jU/ccOGvg249sVrlXK7j9gdo2uGYLSnuIqbqGmg2BU8+DcU1Y+4cSjjYH3feqW+sBGa54re0NwYWT9RpAsgrZk3tbO0FR6TD3ANMTrolAXxMnF7DHCU/OU0KvyCthrp71SicBR6OJajrBVMxrBAjTH5ZDB5xqIl6Bh39711n3KsLlWHe4+4F5fMuyRWG4nHKNePcp43A50rsiLZK19nc7HhiGjSNfU1aQxOExib/PteTGyHG/8mQ7RsIFOYfDGTz22ybSzTJNxjSKGILJ/AzdufEFDY7D2c9QK25LYY66ARfZBx3WNvAQBsh5NTOl3vj7iGQQaWRTVlpdxYcSQwNBKI8UAzXlzlXDGnKJqi1iX2uPk9RdCO9HaaRwPQy6dRc1A/SkI28YEagzyTjwA+aUPnlSSyylUMqRFdh4ugUtZgt87RLVnfE4A3ss3unC1cT+Emhby25zrh+H/v9t9Kgjq/zTAmH4K9JB+iqsrkq+LTgKqSbxAQ1z3HBJMrbVi2OQZd4g1500ZAYmVN1SbZ0CzPq7zF12T50MWacNlwLtznRbTuuvwiJzMIysmGGzbZm87J78NlIcrCM8esIKpgzf63QpSrJeJZy/ljcBmfvIsD+7+dqIkm5MyFbnmK36VdYXtfKwhELd+ubgGP49atthcGosTFY+MzzqaK7dE29xmEu3pOcTr9OIBzgegr9mkuAE6dcqrPJkwS/0vub6FEkSfexqlfI+R6AnGcTcm35n6La0g8F7aJXTxxMYbWqfE1GbpaNa5wJoulE55h887ld+Ky5y4LLWPC/K75AMR7ieOuW0/6MP/2GcAdnzeW459v1smiLlWHnlN6cNPBN+HCXS+MbMPIEg3JsOdj5omgO58aVAbg96//3j/98rqXjZd+b973/N8Xz7sYR44/UmAHDAwYHYUgLyUWYBuXsPdAEcR1BAAUJCs7pyzh3+13n/puqFvdF6d9ERftflF412PAZa6776+lNgU8cJ5/7rVNb2qv+dqbnqum9+1/68lvCe5CPPjck4PFsYwDfu0zbZpTVkrMmDwAG0A+juboBn2WdP7r4Te51zdHZy0/yX4olrpHz+QLlFoLuxf6LqM8lG977Hxg59Nw2+u3xWg1AFviGZOPZzElVTyIcCv2lXxyf2eciCRgTD5+XfP7aqdw8VNmViCRgocR0NCs2fUpMU5id1O3cszYluGtLzl6Ce4/6n6vjMZdl9KKmHwlae3jk8H55TONoruuiU1naK9J8uCQ3eZ1/RPa5+OcGtqWldSyIVYwClSg9M+QHFJC/4O6HHjze8MwrRzFy/s62f2Kfa8IbVtg8nlyaZgspesDMzoIexX/mVI4nJYn7d1nf7FfYXZHsryoF6qDu0+/P96xHMtv/hk36dY/1/1TvYeILXoOJaSttD8nm2LxJQXf76jZTNzvBs9vNs1g1pBZaM42C2Pa5K6btkq+LMq/1+kd08X2AJw2rBN/++Bv2v7o9hEFbpw2pBpw08E3CedZHD5+z1tR/OYqqviYoDqKBwFsmnj0uEdjOLeqMLlL/StlULjwcYdkl1YTky8GFVnL5CMloxLTNCkqrrn+saABAj2TT2xbrCVuNty41xiZfJJiblk2o70fvx5ZyecVLoHAClGaUQC7Wq8Ix+5oqMNZnR24q74uIKt4Pxqgur9pt9GE+lnRwrBgw+PKsSTMNvZMNHlJffRpROFwJh9CatMgbY7nFMYkYpvDZO66zH3CvbZYKiFPvDGrU/J5+NmzP/N/6wS78c3jfddJHeKxu4Kvi//OjttZszE3jckyFLxJwQtzM4fMDC3Lb7RMWLVtlRDvkbdyT++YrsTX1KHEsX12G8slKAkZPzyY6xqbD3lW2IVPmJWMfLKNkQ0j8d15343lOlsWCMGwumHCIRaM3LRmsWktxbsSys+kwGWcLQXjtEiLuPx5TXZbvzuVKaB4JQFT5Mg1fukRXUZqimzJcw3y1kneVUhtiDPmxFx/wsrJCoA46Mrn0cyFhKDQGxYfOuYhXLbgMq0C5SUlHl/8OZaNyblD5/jXinVx7MmEr/Wi1K9hxbCS6pl8ji/rjG8eLzBA2DNQmHyL/w849GfY2L8xWUfh3af3jfPra5QR5+FjH8Zjxz0mHJPfUV/OW8n5uX7/i4EjrkzWR/9bCOr3+2qnkyk3S0Utk4+BKeN4ZJ1sZCZ4QP/979S2E4bWDUUXp0z2S/ksG1oWk89X1PNJ2QhB3fb38VQ2gz80BMpJkqlHb80Qf1wbmXyGtmSlWmRcWIjfEM8oKhiuDZNt5L5VYpzIIif0jb8XmwbKY+1egOu6/Ey6G7u1cWD5r9XhDdjeOsPH/FOb0yh7vPfNG6/5UrwnDVPy6Zh8UWBrqJDVmDDWo9tihuSAxpFAXTve3vw2VveuVuqJSjyYpyXFHXYgYBpn2uy6xjr0jEfZXZdSd/2yaeCuG2YQjGIL6p4Fr1Q+b5fzFMUhW4X5K8shOFRRxccNVSXfIKAEIEXVrKxxISuVGM4xuvdySj65LwYmX5yYfDo1YFhMPpNCSKvkk7IDAvqYfDJO2fEU5bo4YAa6sE1H3Jh8G2wb1BY3glT4Lfbswo529zgBSIglnIJgJFkrHFvpZbJb5QSOBIGAqtahs/QSUKynDZrj0dC97XsbwuNw7N7dhc22fvNaqm1TjoUqMlhcK0/g2mGodB8lScQIUeKECQ7sG0lC0ZefX6FIUQhj8nl4dlUQmPi4Px6nLTPH20Dv2LYjRtaPdMtOcsvGMRzIbAAGS0ntBrMyb4BcP8LANuU7tO4QaTmNI8jud/t++PojX/f/FrPoxcMWzxX8c7t3Y9EMLlmNzFpjGLfA/efhzuV3xu6vDuX0OSkosdGUcZlVtU6t8H8UHH4MFfrxajqFJWPd8cozaXnj0urtq5VsjDwG0mrOiCa6xESa0sHPEkswED4HyPNwnBbCyjYintskQ4oC9SVJyacpJ7/PJVzijuNHiGzfIoA3U6VIw5kFy5+v8yXmtixew+vo4ngfyIhzhXZjaaVC53EKIFdwzxcaRwHTPgPUuWuzbFQZUhuPEZMuuvN8ihtrUUqn9pp2tGRbDGdVpZyPsfNj9UkH1y3RBWPyPd2nyQDPX8P14bLPzHAVjoZvI2tnjff0halfSNzfJ49/Er866FfKcX+ce0mWkNtakbsur7RsLJawy4vfUpLoEBBksjUck8+QTM7oxiv+HTXTyV4ilsDk018zsmGkVIeIoqDMjOhACLLI+fezsHuhcM4u5d3xYdmh87luvtLJ+kR6Eg7P5PPWmbW9orzMY2i96tGQh6rk48HH5HOY10huM3rz8ZJGMbA1KIw1mUXON0qb3JuPmXiMWjf3O0dLrsfTACv5nqoJ5kPdXo2HoBDkfuZBhefMlMNFUDWZHLzY4ZaNNza8kdjocueiO7m/1GfBSwJ826ykn1GXu8Eqk6+KTwOqo3gQQGNMuXuNGmE8Z8XIHNrLx0zgGpOF9K/M+Io2w5kVQ5mhc9clpGRW8hn6/XImjaUhcT4YFHddW1X+HDXxKP93kQDndqhKIx0CllkyJt+zJz2rZz0SUWHE97xgCFpXguh6oTsfBn9zGTK4tO4chKKgUdnGUvIliNvH3/ZjLXo2THHuV5VjY5rGhNXKegIAcDw3pu8d6WXklQX8MKZkyL1Map2E9pp2fHWm2r8o+G4IJYoCY/JtNW+g4ghknXWd6DmlB7ceeqtvwT164tEAksdpY6LyxYt20hfY8K9E9Q0kJrZMxGFjD8MP9vxBrPJJedFJFWb8mzlwirRJ4F12Dr8cOP2v7u/P/sH9p9RFEsUsZIqGr85IPgbjQsxC646j7QWXERzXKGVJSr5jRgzDOXQl8M21vtIESJZldMDi31CKXz+1AgDgJHWb9FzewljugrtuzH2VPGaPGrmPcM68ourxVjqFnMQo1NUQP54hsCyTwdkd/fhlU3jcPEKIz4YteO9XZfIFqDwFhR5DC5qaN/zLN+ToNmgUQJ4xIKnISpMZvn855i+RfaAAGjxFWZrj0JeXXTe467lj27Db29doyiT3CmGKfDa3UARKvmd7P/TL7dO1j3ItD8eylGfGI65S1ASZpdSQblDGr1/ikR8CL97i/p58mKLkS1tpHDXhKIRB9w2zr0gef3LCA3NMPn1bSZlzsoGOH8smpfkR449AjWzwNPQh+SgKkCU5v64jufjSAGDTnG9k1clawXMjiZ4Jey8pnn189C8BQKsMGt04GoDLDnzgqAeEc8yQzxMpKPeLD5fT5z3rX//z1/jxUjUxT3inidDZTAArAAAgAElEQVQeoGbXzSAHpFz5xORhMKw+YNvr3nweLpPv7S1vJ+tfBFYa0nvrxk7OoATMg/oGITHxRlFc75lRjhYByzEavv3imgfB5jm3LRUvcUls+gpi2B4KLiYfdwcmsk0VVXySUFXyDSD+UluDmxtdmr8f28OgWNhgl7epWWXbmDpmFOZ0d6HXU6rxzB6Blrzzeeio7RDcpgBX0WSV664bwiMybdTO7uzAPQ262CxU+CUrL3STLH+sBII/xcjuBHAWm5AybEH+bCZQwKbttNHKyIN/JiWDko8i3P01atMoM0h0razRjqtSbNcyBiaQlONuDgCbnKIYuwsAhs9EUTPu9h21r7Eeudf9+RIOnjoUJ+46GshtA5beIBawHOC0PwGn/FGpK0w5Vpeqw8PHPoxdhu5iLKP2zXPf8b7xQomiYCVj5TDsMWIPXH+AqxAyzRlJkzEE7kXAETOG47Nzu9VCK55QM6QCgMYKziPMgs7wzd2+qT0+oWWC/9uxHFy656UY1zwusr5yEOb2bAIFRV3axm5jJQMCU/KddAcw62RgxGz1Yg5bcltwxQtqnCETYy5tp9FzSg9O3unkxH2ODW/s3AG1b7JCQHs5JCYf7ybmGWVeXf8qXl77ciJldMXuutzl1z3uKq3jMPnmWK8Gf3gZFOV157oDggDfhBCf2ZCEycdj747Z3DmSWMkHAH+rDZTXJaK6BB9ZP95/pnGe7TteKJC1hs0dD7ZxY8qVMCVfWKZOE+TlU/d0VnjZTzO8bJPfji8t+RIAVcnHesGYfFap4I9XIF5GabE+sVcOpxx4df2rcvFEOHXOEGCDJn5kGUl4zpl9Di6Yc4EfAxUIEna1pANWvC4GJf/mipR6TD69/PLOlndC+zG0bigmt04OKZFgnDxyafD76F8qStUSSvjGnG+YW+LJu548xh+7W5ZViTgGo9x1hxYK2uNcdaEgEBUZ/PebN1ydslIYwbUrj09etpRD+iTBN/br9g3lhBAxZnYp7yv5dO6tfKuyrGtiY7K7SDsWLN5wWueuzR9s/UC5Zkr7FK89ojAcde66fIzFDKes7/fWuShX6LB+86QCJvuz8VODHJCqw1MfPIWrl10do05v3eGeHWPyxclcnwQmMoRuLt7KyQP8VQVQgajBri3I7rreP+auG+UaHb1aqiWubgmUgPxcx757NoPw62iVyVfFpwHVUTyA+HpnB37Q1ooSCTb+5ShJwpg+PRlNmnWujYJgVfFcHNt2EMoTYmkZZVPHjMJr6WBS1g0OXeIN/1zCSXGbRfG7hnrXnZeq7ro6RgXfhm4LN6FlAr447YvKcWZt4oWLekn5ycoMIeIz1ikbiWROiq3kIxY+t9PnjOfDEEfJ94vW5ohaAoSJekzRlGSjnrP4sQdMGykFWi/mlaDkC7oWYGLLRPf3qAUwgd1rf6GEDNuIvnwX8OD5YsHmUcCo3YAxe8bud9mQNvvFEgWNwUjSbbrnDZ+HtmybV59ByYf4Sr5IMZ5S4O/XAS/dYTgfriD5ydKfRPbhkLGH+L/Pnn02Xjz5Rdx5+J3agPixQJOzELIRyUNkEO9fS51GWcsSb7SNN17Pbzoeee8R7eZlZqcaezCJcrkieN/1rVRlQDRlmnDR3Itw62G3mq8nHpOvVARuOhp4469KkWPuPQafue8zieaOSrJa8+DHx/rNIbH1PIyxAjaTyV13aK1e4R337mTmAb+GUZKcyafUr+nLLplkzKotMbwHAFc2CZR8eiYfj4Fh8plbkJlNL6x5AYBBFiEBk48U+wHHZXdQSvHkB08m7qeuV2FZRcPA93fC69fpCyVg1X9/z+8DAGpTtThx8onBmkOCmHwNTqDM0mf9Dp5toVhylXwxjMM6/Hnxn3HroeZ5JQ67nYhdcmGnsHzjcuHQ5NbJkfN+4BURGMIAgre6P6Ptm8CsMzL5iNYVVXZnj+Ouy9chxOSr0SevISBCZlAZW/k5B8CVzU2YOmZURE+Ah45+CMfWjPb/bkkVfBlanCe97LqeS+3SlUsBAP93wP8JvWTty0okHUGAHxNLzt4LeFyUOd7c+CZ61vYo1/nKNG0CBnN7lAAHvP5t/+9egzKUJe/65m7fxE/2DpeDhAQWkuybIgUglcUXl3wRS95eEloPoJfpcrSItJ2O9LhpSxj71VSbLnvvu1ysxCwJFHS8u66g4JXcdakn19nIg2q8txJDkymdhyyTUQSKb3bfM4fMjBW3vooqPu6oKvkGARSJ7JKJ65axj/V86DWHjDkEddxqQ2B21z2zswOXeVYPXQY7QpIz+Uy4qoPg0vZWvJlOA1Cz6/KYNWSW0oaOtj2qYRRO2OEE5fiwQgHrLQv3ccw/WRfHsgA70tvLOKrbk5Jdl7eUavr/juO4gg0xM9d0bDvxHsWFKL7Ir39nfNWLhu+NSXZg2WdX/GT+T3BQ90FwEmww2PVCgH4AKOaELHCX7nEpvr37t9Hd1I1/nPgPHD7ucMhggl6uUMJdz7+P/kIRGcerd8MKteF68+ZW9xSELLYJoWbXNTMdomBbtl+fUclnJWPysRdMQFWB98NlwP3nqkxIhuNdd6gnP3gSb216Szj10NsP4Y9vqUxJGby1/LQpp4EQggktE9CYDncJDMNgu+uyNuozGlcNZtHXuEGu7V2Lcx45B5c8fUlk/T1r1I3JDQsN72GAQaSxwysXmjJNWDxxMcY2jdVf68czIkDvBuCNJcDT5oyISbKMttXEC7tghje+uXlqwxZOyTfzJO1VwnhiSj5JuBdj+Ohdk8Kw3rbxZy4eniXV0YF4bHQTdIoFDJ2SqI4ft+njqR3QdaDwt+iuKzL5cpq+JMnoy5QOSeJLFYneVVmVRTxWjsfkQyFQ8t3z5j34yzvR7rlqbSIo1CynSUEIRarUp28hwRq8x/A91Lq9/x1PQllbDNg/O7VrQjlw765QokC+zxgOIx3BYJdZX7rz5eK8R88T/v7RXj8CAJw65VT8dP5P1ba43//bc613jHieFuL7G9c0zjsXrMphxuwSwpmtcvs6EOka3iU1P+kg/TVE/GpkWbKHC5eTIwRXtURnugbcexVqKvQJz4F/b4QGTL7XNrwGAJgzbI5Sp84o0V7TrpQD3PmDAMim1L3Fu1veDe27bh7Ja5h8gYoXaO4NGKn9hjHJXNuPnXQsDug+QN9v7yHx+w2WoVcY62Uk1uLfbZ4p+SJWowY54VAETF5FutntDU7Jt58VxJsucO66QLCXKZb07rpOKY937eh5IOpOSMQz7S0E8RVZa2x7zPoYFb6giio+Kagq+QYBJVSm5AsTcp/NqhZKU4w2PmNoR47LJkUso5JvjePg+mZXANC72rjLyalNO+GuQ24RzsRR8g3jsiBt4bPYgwrKi0l1gcvsU8c/5bsy8hsuXQw9i1ja+BYpSvEfnR34VUjMoaJv1XWfG1PGfWu3b+H4HY4X25GWmoIVvBddYOPThg0BJa6V0STQxo7J5/8fc5QRfc381Q2pWmQ0wutObTvhR3v/SJuEJbRJiG4QAFwlH8cQO2zcYX4cMGZ5H143HM0ZlY341Jtr8fVbX8Cqzf3hSr4Q8AkvGI4aHx67JwmKJRpLyaf7vm1i+99PfUrn2s7HBooW2OSNgtJi0ZyEBAAwcmcAwBlLzsCiuxYJp85+5OzI9oGBj2miHe0HhCvVksQlAwLFv25Tgac9lxrN/HLInYdgydtL8Pj7aoZqGbyQ+ZFDVvJxIkAc12aXjUyM4/yqZVf5v598P5oddeLkE/HT+T/F3iP3jiwbB/y30Vbk3NAP/19teWUDS6kS54kfx4RTKsVVYN3RUI93+MyQUgyjnawOtEH/zccBhSsDWJwSyGoNFLVJFCiyMSdrq0pytr7mOXfdlbaN2WNG4ddjdq2Ql6ga36KkKbai8DOa7p4pGJOPghR6AW/NWb1dzWoZBzq5y5Tc6dHjHg2vjKuLGBSXYQas7sbu8Po9UAQx+X763p8AANfuf22oeyvgrW2rXwE6JmnP6wyhSRBXsSuvfXnNOsbCDpwz+xzsP3p/Qz1uXLg73ryba5+ie8XvhXK3HhbExGXvO6yvOiWfrBiJutMwPYcpJp9F4nOhLzUo9HVQvqN8L5eJVJzH7FIesFNG91YhA7p0rjGjyuVCRmjZYAwopIC7j7gbz5zwjLnvCJ5fHJZWX/sE7fFYrFNvsGzlGNLKcyEIjSGta1lGAdRXQochqaO/yVTBh4lgWGcI8dCfruOU/5zBQOOuCwA2zWNdGRqJy/a5TPibZNUkgzzqUqJRjSLYS/p9GagYwVVU8W9GVck3SAijzpvwpelfwuULLg8tc3OTOoGZmjJN/ISQ0CyvDC0airflx3cjaMmIgdprnBp058PZGxMazbG3+EX7C11B5q76dL0f24G3opoWroa0+ozyhOADbjEaWT9SYeMxWE4tHj72Yd8i3FHb4VP0GYikOHuBc6PWuetutizfXTdJ4Ga+Klm5F3eI2ShisqWxenIVpKwUajQLGx/QNgkooWLsLgAo5iOZDg8ufhCPHfeY/7dOUMswBczKF2P1hbkiPb86YLymrTSu2f+aiij5wXt0H2ShpGHMAcDxt7px3ELgWA5GN47G2bPP1rIPgOQx+YJ+Ulxw8A7SQU0/j7w2Ub1RGAx3B2VjPeNE/+f6vvXCqZ/v8/PE7roM2ZRmflzrshPgqEo+lrwiDpKyC68/4Hr8fJ+fJ7rGCHnsSMqF8GtdnpZjETXhjYcrX7gyKB4xQz109EM4f5fzsf/o/SuPyefdSMlbQ0a11uKkl07j+hIDvRtwy2u34B8r/yEcFsexeaMaF/zIctlDBKOt8pmMTLHA18uvMQd2HyhfEhs6Y+ChYw9FR00HFk9Y7LZPCD5w3M3qA6lixUo+kzukCcyo9v6UgAkuu2SxGv/66mqcbt/n/rFRzwQ6d+dzI9s09dCk5EsS849Ytjq2jroOaO4yXhNnTmGKhzTEZ7Nj245aw2jwSVLs1lULrHkVGDotsp1yEE9xohpXr+1R16yoDToBBQhwA2fwdVV8GlnDzrhKFC5URCiTj6j1yMaAsDsdUT8CWcc2fkOm8SU/v7D4zoLBIQIKky/f67vauvNiwIIjXky+qHi9zCjBI+r7yOSjs62ObRqL2lQtDh/vzgMzh6hhMdjI59szjb2+hk7t8ThhiWoyapng3XHyu53cECqPjQktEyLddZPOyUljeCvtzP0qtpdyikINAIq0pHfXpXksKUWHO5D7Nr1juvA3/z7PHH0iZJyy4ylqn7z/2bxbddWt4tOCqpJvgPA6t3Dy8SaSZCf9wtQvYH7X/MQbHtOEzG+yxIXMghWDZTO6oNvIUVDiLepSu4QQJc6dDNO9EVDRvcvwCHgB7p+a+ISsT0cP3084npPa/d4e38NmQ/ITssPBaK9pVwRfMaaM+F4vaA8ERp0aK0+CjGJaJhfVb4l5YUgWjMKy7PLY11YZbF5HfDiWg6yGr2dKEhAHjmySLuYiXfhUtx71JhuzDrDyJXfTseMi4ERXgXZnfR2m/moqtuW3+bHR7nnzHsz93Vy8uVEMZP7sZ5/F7sN3T35TYm8BBCPh+Xc2Iie//PqhwKQDgfHueNzQt0GrELKJ66572pTTMLROHwOMWUXjuruyfrXXpTGkQVJ2vfGQesH08KxmSXD27Hhsv4rBCWO8EhcIXPzLQQ1TJK//F/DjCcCGt4OTCdmBMmpTyb6pXYftGpqcJgmItB4VSgWcNuU0zO4MTyLiXQ0AsA1KvjgMxfkj5/u/O+s6K1buych7hqnPzAkUIhftfATm3By4jZ065VR8dQSLvcU9j9x2RcEHiJu6lGX7DPeylXyabIRhSg75ncmgIAqrne+zKQasDjJTSFbyERAMrRuKvx77Vz+LJeAy5QE3DlPFTL6E5dm9Z/a+wD+mU4ZQAC++twnzrWXuAS+ruNzjk3eMl/hGd5+mBAKm7Jm6OgmxVCXN1GNCr2OM5YZ0A+pSdahLm13AU5KEYlKw8HGbx7z2fwAtAi2jtWUrRdx5oAHi2qlLWBClMPIVvhwrKVSxTICe/tW4w0vIIX+r3f0EZ009A4Ar+0e6E4ace3Dxg6GbMsYGk2UXl8nHMXkr/Qg92MSW2M69AqORfxak2A/YaWzLu2ES5Hh1QTgSoE9SWIe9M4ISsg98PXafdxu2G3pO6UFXg6oU95l8WkWw+ND6in2aMvG+5VRIYkVhqCdg8pnGzbC6YYYzARIz+cpcltkTLBIL/cV+1KRqlHMllLReHlapgN8UVsVqo54zaoR5a4yqGa4cS2ni/uU7dxL6WM2sW8WnBVUl3wDgQ9vG4pHBROta1ZOvsjLhvlwWFQPP+OEZYZZlxbJG6e6AEcNdNqBsoSSCy1A0eCWky67z+2gYmryF5eudHWr/fJdb8XrZ9TjM8m0Z2D87tu3I9VdcNvm/dG5cBUJQIkSJYwIAMzPtsBBubSPceSZkxX3SQ8l67XH++pSVhsP1i2VA5QWzJDDF5DNtguLUx1CXcYCr57l/jJ4HTHAVaDd6lvnbXrsNC+9YiL+8/Rc88K8HAETHbykH/Ft8b4O7+Vi1RcoMdu5rwp973boX3tj4hlJXHMvhsPphuHDXCxX3BH3fgs2G9lN/5PuRdSRlDPI4bcpp0YUGApyQLLNEkyrTgOCd+u66z/8G2LYaePG2oBAnJC7fsDxxNs1/qwCpeadnzz47djIUCsC2LK2S74wlZ4Ree/mCy3H5vpfjhoU34Mp9rwwtmxzumyt6sYd496471j0nbNjOmX0OWlPN3FUeTrhFyzzg56yMlUFfmXNi0FMxCD5A4ISIYqb91sGNruskJcAvmxtFBR33s5IsgWHzElMCUQBp707+lV+PTSEb3DhIqqBgI7rEzQWycZVfP4vsWXuunkrZGAonucQB/T/E5uI7OOQPh2jLRyueOGWJzsuC69O63nX+73FNrmcE2+gu7F6Ip094Wtseu8vJneKmOKpvzWRLkNHWoODg+z+ifoS2TChiyo28Ad0UcCLOWiqzyawQiym7t+u8MDZyGJI0CFLecylJ9YbVZz5vVpT3e8mfJrWIbtMy4y6VMIaysS+EiArQFU/4KmIhgRAoSNFz1/X6GMai/8FQUQHnaJRdQubebWuU8+WgQNT2+Oe2vsFNAocjr1Uy1rZmW3HFvldUvC9joEBoTD7mSaS9joNN7EjTSlxCAIMpHqEJstK015uVGUmAr61QKuqZfBFZdRlKALYVgvVcDhXAjxsa48YpCEoea1NkqVZRxScfVSXfAOBFiVFWIsGklsSuLQuXe43YK9Z1pq04C/SqtAMCRLg03N5Qh+eyqoXE8rclRFGkuQJBfIh8LYrzdjmP+1s/OceNlSCvUTnp73IC8ov1i++13yI4cRhbKMzQuevulumERakxbT0DNfwfBTeDI9AfUr1ji+lGbjroJjx87MMxW9DDtggmE44BVcwLMfniQLfhWrWZi23CFC57X4B0sxu4fdkal6nx8rqXfavyo+89mqjdJKAAGIE16h2aEFfxc/wOx6OzTu9GYq67vD7xMWTO+utZ2NS/yZCFUcT4ZnP22QEHJ6zLSsmk8fgYKKGBko/VKShQgt9H3XMUjrk3nGXDsMcINxg+WxN2atMEux9sVKC4ZXftWEQb01FmUsoY0zQGgJtJeM+Rg5P9OudlT20u6DeFarxLbhYdOlWbQIBlvQaArM0r+cr7rvihRL2VVGHa8uUNxx1OySaDN5JVouSTdQVCoH1O2Rkw+Yq4uF0M48GjPBNPOEoEWGXbOPCPR/vHeIOcDMI2hYniYZmxunkGXqddeDf3mLFM3HdAQEEsy19HdLyw05ecDgBYNG6Rv8FlCpUwIxp7czuPFL8BnYLFLe9e4fDMP8vBfW/dp5b1xsIjxz6COw+/09gHI2Ia//insd0Qcibus56YC+YwArMiRPFYkf+mgOUpBYokWi4jMeIX6uNhB2uyrEglRMyu61TMp3VhyRzH9W8JccuEZ1HMAXbGN6io628wX7yR2yCcmdIekSgo4pk9/9nwtYchz5Q4/D7Cf9YUuXQLMGouMP04rOtbJ1zbUdOBvUbG25OFQdglhcxBB40Rk6zkCu6TlyVo27ITeYzFgez5FAXHe57sG9rujZLA0BrMZEVaVL5RCgKLijLF9Qdcj7sW3aW0td0S1eBhsnNcVi2bb1n/TXNiFVV80lBV8lWIJ7NZnKthlJXzYOWJL66wYprIxCxCvHBu+Zk6TfhOexue0yT5IKTkMvkAhSLkqv3Cp1WT8o7AZS4O9ffU+nJRz4T4wrFYLicFs65Uyad76i9mM9i3a3g4I48Q7a5Nx+Sj0u9ylXwlAN9pb8XO3aOk+rmF0kqBNARs1NpUrZDxLOl2lsXkeyATBPT+3/qM1h0uDvjh8Nm5nMsQEyT3+QZSXoZOpty+ruc6X/Fw++u3l9VueJ+CTvUXXNFL95wKpQLuffPecGbcwHotAuBjCJUnAPKK0YfffRh73LIHzn00Ol7VHxb9oaz24kC5E05YT6pA1kFh8rF39teL1b4kEKwPGXsILl9wOfYauRd+sOcP8MwJz+A3B/2mwt4mh5xdtxxYBMCD4YH6deDdOwca7L0Vi+47OfpRfdbDq/a7Sigvjyhdll/evSfrBEo+2aXyv9pbsSUWC4xr3VsOUiHrmqlGxijSvVFBGVfB5FKQ+FKCHOH9fqKmBotGqm5ROpw0PNpAkZTJR0FwdbMYwmDeiHnaso1ZB33DdnH/WHSFd33y+ZF/otTbEBagTziQtMbG9x41Ggqf/vBpLN+wHICr8GFzEFOohCn5mAzhSKrWKOYir8wuWjYuePwC4XzWzvpxDNtq2spjUMeYlwjEtWyrHPM3ASiApiLn6YL48pTs2ksQJC3qNyVNEcpHMPkoxUYDG7av2AeHODhmkmhYkj1fnAHS+ejk7RLzllESq4mJN2QlH+F+pDklykFjDvLje2rLgwrseYa3NwcG5LhKmV5vzPDyP/8+UsVeIFWD/mI/tuTE+HBJjIa6dywbmCgA9PxeKWfCtpz73coKuDhMvqSrfiHhmsGeP+vFFo+Vx+6ZfV9u/FgqKOaYGGVJpJRdh+2Kcc1qDPc+6f7D5q9YIhoJFKe8AruKKj4NqKqrK8SdDap7T7nbqEBB5f6vix2gAwXQXCwqgoEpRhIhBFaZlorAhkKEmC2s3hgV6H76cYfYMVNdUYwn9uzkSVpWWPKxImRszm0ObcNtR796rHYcTMiZM5fKcUzYUUDdNC4eMRTL0wGz5HmPWZncdRa404snYxqbKSsFUt8JrNOfL0dmbMiK7+qa5gbgH64Lwv/s/T+x6pCfVXt9GiOauXfHWXhZrJRcTNr/wIGiL+8+2TmjG4H3xbM3vnwjLnsu3MVWlyWwEvB6vQ836WPLaHHBu75G9bxHz1NO/+2Dvwl/j28eL7gf//qgXyfraAIQqnGFtx1sIwR/aKhHA7fBPXGyGnA5CfzEGxop8Y7X78Dyjcvx9Vnx4wQ5xIFjObhi3yuE41+e8WXs3LlzRX1NhAFQ8mVoP7D8T7HKzhwyE8+vft7oejTQyBfDGUFBvEtvjSVe+WFu8O4o18UaO4t+QvChbeO+enHtv6ehHiMK0YpmUhdkSWWhL8JMV6a51w65it/0VxL3MC/NpaKx0P19r0YGMuHlTPRGOWlv5RGtyxTL+t1fKCHFtIgt3QBEJd/V+6kx3kzwY0x5MkmRJphnQ1CztgfUkBzh9D+f7v/u413WYij5GI8q5cXFdYiDaR3mRBq6YfPT95Yox/5xUnmGO6Gt2ujEMwRu4jeGbRoF1BenfTFWPYDokkiMEp1qPJJlSwIg6ymN7q6vN2bA9ctHnQ85lyvm4FiOonwjRJSUzKk7kkGuFwi+Nzf0DHeimANSTUZ3XSFbLvcMp7RNCX0mKRsi4+30vwIAfvLsT/QXhGCjx7jUudymUUDLppeBoQfhw60fKucrySD9jTnfwMJuL5mgd6sUADVkItbB8YgZOiVfVGiVpB4mzK15Um8Kr9VEy6aBks+9cFXR3XsOqRUzgrO+C+66cOO8WzHldhPLlUEcsWLZI8YfIZV1UeDeCVBl8lXx6UGVyVchdKLYSsfxN9mVuOvGjW3C3DFliEw+ua3yXj1BKYjJp2QNI4ktQOZ2DEq+qMnXu6w2IqtmnWPemERlgHWbMS+qYW/cIrp4iFTL5OMVfATAz1pbhPNRCzezVvLlClwP+fYcKxV7M3hqw+TIMhQUM0e1cH+LiJs8QsbYDsndjosRxDboJjd1ht8e/Nuy2pZBOEGlz9vct9WqFkAWQ0nO/spjaz7aDTZZ54JnnuiLzDYCGTU7tQm/2OcXwt+dtclciQcC/9PajB+2tQjMw3nD9UyeuFCYfAyzTsFFT12Em1+5OdE7M81bZ04/E7sM3aXcbiZHhWxHCsCh8Tcmw+uHo+eUHsX1aMDhzV35Yvh6y94DW19a4L3Dk+9GoVTAL54Xx7M8V2TtDFakUzhglH5tjsUu4pIiMNOWXcZ6bPtMvqDNHVp3MBUvC3l5E8rdnilubqVIyqEoAiDtE/2/TeEMKFwlX4Yx2TRGVBMDMAwlT2HRRzdElDRDlnfixNDqLfb6azZjyU5qnWQsz5h8tucWl7XToW7NupXj1ysfF/7WZaosBySmQdtXxREb2074nX/8Vwf+Co8f9zi+NvNrseqhAPq4bzUsJrKcxMWGquSr8diLl7U2Y2uEl0wcJh+P83c53//dV+jTxguTvWjedQZGSSGHl+nn9hpu2B/COg0UcyhZKfx46Y8BmJlvFEADJ5+bZXq37rRjibHrOqeWcysAgI1erFZByefdw6H2U7BoEdjwNt7b+p5ybdzkOW6V4hgY3TjaZ4nzz/PWBlGefeIzTxjrZMxtWclnWVakF0NSle8DngErmpfqQiZffFBww+Tw8iAF8P863GcgxFkAvx0AACAASURBVHOkbisFrq2L5l4k1Pfd3b+LqbXumhup5OPOl7iJtLuxGxfPkz0yqM8wBPd/lclXxacFVSVfhdAFuH28tsZfcMuJlcAmqWntZiurDF18oKntU/kCQv3lxukhoKCE8dFU9+KoWA6iuw93rb/dYX/r+xdtBXXP14co8RzLCQ2sGifhQNjip2TG46CLyQe4LkpxR0qvZeHgkcMiyzO3Er4/BRIIg/z1KTsVKnzyZ7pTjciWonvLB6iW+QVxFYpB3Ce3vQN3kjLPcoIXY2DyDAcZ7TXtmNpRvpCoAwXQ7zH5HI1rLLuH/1lqZi+y2IGfBPDC+8iGkcK5wbaA6kYdC/T/9IdP+8fKcRkDmFsJDbLryvM3Nzfsc9s+yvUm4fDjYhmOytQaCu/SDI3PlC03LmK5iFLyye+niWxD/5DpQE0Lru+5XikvzxXZiPvZFrHBB8S5zw19QWCHzL3Gehg3i7uUZZQcCNd1QGVFC2vEwNjzFMhTaNSIlRViJrY/q+e9dZtdZlCFmZ39fnmZb/vJB/65/5j1H74LaxyII0I12lJK8eC/HhSO9RZ6/XudMWQGbjv0Npy606nmNrwOM3fdfKkQkdWUeZaY38DnpnzOeC4J4rqU+0y+k+/CtnTA6B9ePxzN2eaYbbng3f4IVe+yNevGlpTZkY6GyVfjxF9vou5UNchzSrZiv3Ytmdo+VbguStEYF3JMvi3EEhQhcky+9y2K97e6rgzMhTkAk+UI5jVO8I+aPJZYzWnbAmzunitIirDJsmATWxObFegiq90fG9/WGvD+c/Z/lt2ufk9B8IoUz13XLxlPSV5JNrFRKoXvWcpd9ePOkPJ+6rX8RtSn6jG83g3jwMSOh+vc74RPasL2yDc3usbllJXC4omi+/aRE47E54e4GaXZHvOHe/4QPaf0hPaLF3dWbF6hnPeZfKy8d+TjIq9VUUWlqI7kCmHKYjUQ8m93U7f/e7dhuwmbWB4URBB0CaV48Og/C6nV+cXYggVaZpwuFhOFQK+oySe68aAPYYJNErDr6kM2+c2ZcGGwQKODQBMSxuQjGJ1L4xfH/h6L7lqknNdmEkN44o0N0iL6bioFSsLdgyxPGOV7micENkoowRLGTBSTjx8tvIJ4Vv1kPLf1FaU8gRe7i2uXR3u2HfFAvDbdv06d1y2edlwh6a2Nb+GV9W4/TG7qADC5NZqFWA768u6m2lHCIsdjvIxqGBVZJgkI93LjWmN5hBknWrItWLltpVs3Ibhr0V0gIFi+cbninjHY+Npfv4ZHPMGRj6FTm2DTxYM9tcBdl/t6huwE7HoG8KdHlesY7jvqPmzPb8dR9xwlHP/YCI0D4K7r8Eq+Pc8F3rvNWPax98zJCAYSbH2LYmH7TD4+Q67HKtFtAmRkI9gcz2ji2IaBxbeVM3bKZXTwlXzcMabELDeLuYxRDV3Gc5XE+gtDUvVECWLoED3TKQAt5sUssWWIQrz4NGLvU/HmnhRzf/cd/9j8kfMxvmU8uhq6Yo0rWeEodGnM3vj7yr/jvMfE8Am9hV7fI8EmNia3xVvbHMop+WIw6KwQWaeSpC48Yhv92JOx09ia3+gfT2rUoRDddQuae2QJRORQGrpxX5Og/WgvGnFA8s94Te8aIREQ4CadcCxHGJM2yg8bJLctKw83de0C5FYI/RpHPgDWvIX+jrH+MTXrafC7xF1rDsFDvHqI6K5bwZh7LptBY7pBG7N0K2oA9AK5bYo3yLKTlyUa6/IY4ZV8AtFBmntC2/Bksu91iImNbGJH7lnKVvJxFzYhg02GuKPsHbLim0o5tGZbpQzMBDalKBKinXfWOO68LTNn/b54I5op+czKYf1c8t3dv6s9TuEmzAGALR7Tsz+BG3UVVXycUWXyVYh0hJKvHDclX3jn6g5jRPB0Y9b28PrhRsFJ52obF44V9FEXky8JlVqEH+kvolw88Jv8Bg3jMAxhG8XTh0W/T9eljWBo7VDlnN5d131nYQzAm5tUF8qohZu5gJUAWCz7IRHj2jCk7HTsTZvNCX4Ht+ytocC7wrjNaflk5e/4lmQZWCmlGNFco44L77v41+Z/+Ye2F7bDhCAmV+UInhdFf8F9pjb/bD/vxi8yjeWGVANuOfQW3Hn4nTigW58ooBIwq2QsJd9Xlwp/hilKmYKPYVzzOIxtHhvEnBlEyHfyyLuPaMtVmuQhZWuUfIuvA4aGs0CH1AzBhJYJyvEjxx9ZUX8GDBUq+SgBUiVOAB4TniV3be/aitpLivc39UIYJTNPEs4HylZuo+UpheK46GQjYva9n4pW5oqJNwgAAqcMUYxNr/xqxe4vCZNvbp1Zkbd3l5hNkp/LBkrBIyNpPDG5tOk9snILJjSL7n9lQmy3JDDIGatswagFOG3KaYlrFTwzDvu5VmnbV+gr6x0MX/80/pHNoIRSKJMv5X0XXc1e5l5NmcFy2daBgJNd7JTPBhrbNDZR+A8dk6+PqN9LQ9qVuWQFim45z0qJ3Ba2zTK2H6bQ5/vnl5ee8YZ+1y2cZQIPQhAEGAgFH6Aq+XotgotzK9z2uPvoIm428z5OwR7mrluyeOWP4Xv3qk87BJCy4YbJJzIu3PVCHNQ43+u/hQ39G7Xl+qn3LYzeXVHyVTrXifNxYJyRExWGG9r152xiR8YHjuP+r0NDJljPdibDjOVkJXueinML28rutd19b3zMZOp1joV6Mo0blpyHKedNcxe/vpa4TZUujAM7W0rwHqqo4pOEqpKvQqTC1yfM75o/IO2ExfZjsVbktnX9cX+TsoUzxwtaTQgBkVwCLES765qgMPlCFE5X7XeV0ZXZT1rCWQedtKggi9rMhbnrjvQYaETD2PKvZz3RPAsLlrb9sJgw5YIJJiUusHsBxBeUy43Jx6uNCNGzvgiAnR/yMsClapHn3mddKn6gdiHzma3pH2NkcF3ozZsFwPN2UZNJlA9uk5AvIot+NL1+O9C1G3DRJqBrjlvK8FyfPOFJ7NS2k1YpNHA9k/CX7wJP/Ew8dsJtQLvYhzD34QVdCyrrXJlIwhwq112X4em3vAw0vHBuED75ZBo6FlHPKT2hsbI+UkhuPckUEC5mPh4E/8fHhaHo4fdL3xU3TtJGQGaNUALAsvHB1g9wz5v3RNYfpeSLA3k+sBCeeMNYj/c9nNPZ4R8rh8nXbJuTUOni7up+DySSPosiEeM56VmzQV8bnKIQj6/S7LoAMOM3M4R6dIH9k9QnGPxaxwrZQBl4JV+cECMAkCU5jFr1F5w2zN3whjGMLU+L3FrrllnlaBiSA7QZju2uy6hFVgpbc65L5W8OTp6lnALY0DoJ0zqm4ayZZ2Fmr+rdwe4tislHoD7H9pD3nzQmn6WRswHg9sNvF9hJfIzB4kC9F6ke3tvEddcV+/aDvrf836qyJlBu8bG7V2xaoW/b+z9lW0os2XvfvNf//eBi0Y1dxvE7HI9x6RBPCZYMgrE5P/Pbiplc8nMTZGTvlMsgq/w9WcTCqMZwT5Bylb5pO3i/YSElWjNu/G12l3kUtTEMSyDYoXUHQSlPKRXYdEx5LYN4YyAfUY5Hygn6rCtvETYeRSwcPfgG6yqq+ChQVfJVCMcgIFoVaGx0k2OY+1wcJZ98tFzLlG3zLraSsEOIoMzR98LQO+n+wgShPUbsgS9M/UJo/fz92RKtO5LJF8KCYMHOU8SccapACAjV3wMhRFEEsDu/tbEBlzc3xd5yRHE12H2+4zh+xjfmrisjRezYil9xrOmvoQCa1i1z/9j9aygcGWQtLCf+HAXHruLhqN+Kzsrbmm1Fzyk92s1SueDls75CEWc6npJg6yqh3EfJdmCgCKy3wih8/CfAQxeJhSeqAs07W94x1v2NXb+B+4+6X8kUO9hg466/eeCVonI7X9jTczviLfqasQaI7v/smzt96unasv9+iN9+d2N34hqyvRyTM2Iu5UNGfBQgkNi00tyvc9clxMa5j6rx02YNUdk4WTIASj55XSDEX1eS1aOC3V+ckBMAcOkelwoGMRlha+VgsB0oAtZ5XJQQsDyAECOe19/0h0vdBENw1wqTe1gUTMwaIDpLswkBN1yETmlLQf17jcPclOVEIFyu9MepV/dF7a3GspUi7ljye5uuwzYvuH/S0AyEAqsdG//sewcvrnkRp087HZmCzKmC8dnK32/KVg234d9N+Jz5oWTMNI3nMU1jcOSEgCH+UozM1ZWiwHXN9QhyfzNZY1kxCJkhKz75u+KVW3uODGeDp20ClApAfSdw+sMAxLjL8RIURs8pDgqgDcOBbGNk8rakKEHvrpsfv2/sOkyfalh88aD98sCP1LAx3ebFr2RdzNGS4E7L5NF+y1KUv5S655lX3K7DdtU34hko2R7T6K7Lja3utiDGoW5fzUryX3jGzlSZfFV8alBV8lUI2zDx8lPE2bPPTlTnVftdhdOnni7EtypFTNNRmVb52AqVuesGbrU6K3+SxBvi8SDWH+tjGEwLG7uOZxnKQlIlTD7W7vdT/4cM9IJAzuu63i2XqO2narDBSx5wbUsTlmYzsRblPsuCHSKks5h899cHQrDrrstcggI4xIo9JkrcYKIgWiaEEPPRcpDv3j1W3TL4PtWkvee28qWggCcwLFu7zD+k2+AOdgKA7f2cmDD7c/5PSqnyXFsyLfjhnj8c1P4QBO/3zPnj3B85jRvzxMD9fH3feuSLeaztXYvPPfg5pShjpzSmG9HV0IW9Ru6llJExqWXgGGzEEwYL9Z7L9aRDBqxuvg0AmMUyQ296PzhpGEO6Df1Zs84aUIXygEFi8iVZB7RhXBuHh15z+YLLY9dfCfj7sHmRXd5osvWBX4dsB2t61yh1Xn+AmoijpqW7so5CfeZu4g13raiLEMm+OfXM0PO+YiJGhngAmNYxDY63Vo621PEqr2HlMPnC1igZ1zc1Ki5sIMCde/3ceA0FEZ5bWEw+xwLI+rdwYUc7pv5qKubcPEebcCUKfAgEPhZo+ZCYP+xH1p1zdYrIa/e/NhGTT/e24rAYmSJQjgsMVM6YZuDHUnuNPl6voPxtG4dtuW2ocWrKine6VIqdWeOoSj7Wp47aDuG47PqYsonCtgtT5EV9NY9kxfVEybw84D4f4eDvhB9lFrEES2eSXrF6lhy9xJhdnt132iHAlpXutzBiFi568iI/g29cUO6l/Xhv8VpmhHVQAvEMeQOt5NPJQZQA+XR8rxYT4oSZCDNIhEH0ADOP6VYp6U2OFrXuun3E0iRkccFqv3TPS/V98Sr5VZNroIljSOHX2lD3ce7vcg00VVTxccTHy9fmUwReCEyaYXdM0xicNess4Vgkk4+bjQuajZssnJer5PMTb3gZdmVEJt7gzmeQhxsiWOOuG2HtNAfrdWEjWPhkITDK8hXm6mR7whwlQA36tWy6HCEgVP98lq1ZpsZpHDELWBfERNtq6aLmqeglBI4XyFYHFvvlzXRgwcoTgv+Y14FL/iZmD3MSCOsxEuuKAp9ll82WYCCgyDJ3oZfuCE54QtkvX/pl6PVxBKHkffLYQJTie/e/gs/bnsDGKfl+9uzP8MuXg77dftjtH4nrJj/2RrdmgVfuBVa9rBbcJWDE7n3r3hjfPB7retdp67x6v6uxYvOKRBu7mw6+KTTbcVmgAM59wxX6b549oFWzp2ZbxJVM3/xLcNLA5EvbaXxvj+/hzuV3CscfPvbhAUuAMFAgFURrUphAZz0PRCi9PupELAAVmXzZGG6TxNKyi3VMgcwAKG612dW9Y3NIAx6mm4Rz/DMfWhMoHHT1JI3J51iOPzfq3LHkNvhg+nFliFoKbIkpbvyitRnz5JBZFLCazO5o1zQ34p5ckNk2LLvu8PR2oNCHe3MrtWUSwbunhbeLTOjjdzg+eVWmxBvexlQ3j0xomeArl+K+b3mzP7szZP70aVru9zQpl8NrXDbQqMyW5eK+I+/THk+Rov9cKKVYuX1lWQmWCFQZeURzFuiTvBS8MpfMuwQL71jIXa++K3nMkRB5I0o53l4sYS3ntaC4zA5A8qQk4PvLPze+HxTAFiv8vvhEQYViP2qd2ogYyW75jEOAd58B4HqB3LE8kP/KUciMaRyjawYO8v731l/sh01sDKsbhve2vpe4DV75ecm8SwR3Wv595hPETq2EyVduTD7+Mitk3LYo7rpAnSUmN6JwYyI2SglZSsx4C4IGp9ZoHGVs7bXePkDHzAPE5yuG+1HHCqHANsvC3RsCkkBVyVfFpwlVJV+F+GutYUIa4HbCFnbqxRXIlkros6LJmW4w3fJ6yGQPAlVYsYgVmXjDDInJF9E/08K227Dd3Ou58wPJ5GNCdZiYlScE6ZA+Kgw/6e8C1Gy0OvQSAr6FMzr3wBf3vwyzb3IFd919/qW2Fl9+9ghstI/G435uRyCVbY7twl0UtgpEq4AWrM2Wo8S1iQ3uOWQZk4/vZ0S2SwYTO6AiMBdojx2VZcxOTwm2tnetoOAD8JHGZmNjtP2P0XHXGPPnjY1vGMtMaZ+CKe1TEvUh62SRdZJlHDWhmWzFNgLUvf84UN8RfUG5YENOju1osASnrTQOH3c4Dh93uHA8SezJjwwyky/BOsCzQwEADeEsPmDwGbQM/H3wSr7HMgaBnXPXTTmp2IroONlIecRZky3eXVd5yCL4NUX35tj5uMplhzhwvHVCG8lOWof4DVjcsZNUItAlCiK2ec2+p6Fe+FvP5HNrbXYK+gwSCcHf05Z8wOQ7fofjccGcC8qut5m4dTHPDH8qMhjJ2OY6FpvNW+7Z7Z865dRwJR+7zFNEZBMaq5OAH2cmIxL/zG957Rb8acWfIo29UfjRXj8CwGJz6SMtD68X5zmtkl5mvIZ881HK8YYSxVpuCMvy6GAlvDEjeO98T3LFHPjoBTqmp4jgvu9a/ffQBGk86q1g7J+x5Azh3P1H3R+rDh6yEoe9T4eUfENef7EfaTuN3x/2+9j9NEF+X9v63C+QAsiXihhSMwSre1eXXX88Jt9AIITJV+O563qv+F+FLZghzUkUrruujslHCUGBRMxj0ncwsWViZI/5b00fw4/iHUdsM+kaX0UVH2dUlXwV4vWMXskw0MtwnJh8QwtFrEhHtzwQTD6AKNK7RawYgWS5rFqEPyoq+aIykJkWtsPGHQYASPNKPkn4qCwmn1tX2KKZIwQZqm+HgkYK5Hwg9TA8WVuDGm7TPjrdLFi3dO/4qpYmfHnjJhxiP4PHuPOO5cTetAlLLTG46/J/5PvKZvLxfcqy1M4xkiHwmNgyET/b52eR5ZKC9a1QdO82S3IAsf04YGf99SzjtYONCF1BgLHzAQC5kt495ar9rkJjuhHr+9YPUM/Kh4MiBjvChD/a1rwG3PefUgcM7rqfJKFQji+VcB0QnNpS2cj4mh/9s6GCu+5XXr/R/827yBeLXMgCy4rPfEuoVNAp+NRnTnwWXRhbAgBsboMaxuSLq+RL2Sn/nhwtw1Dsv6Dkizl24nyxKUp9w1ZKo4VLoowOy7TakKIDouQD9PPrzp07V6SEOcP5I1DgE2+4rZjWzwt3vRCTWiZh7rC5sdtg2SmZgtAE35vC21xvsSw0pBoEpea/C/9Y+Q8A8WNPmsDifxFQUBIvdlnYd8dgh4yBqPEhj6sFXQvwbXw79vUDDf5+efm+Id0gGGE3eAyA7+/5fczomGGsL08cbImlOHPrnpZ73j+ybE3AuIpmAuoRJn+X7DSm/2oqAKCroQv16XrUp+uN5U3gXVvl99VSlwHWAUUQPPbBE/5xFgN2QdcC7N+9v1KnaQsYZzyUy/3kZfuwPVkrF5f4fcfGllIOj7//eFCP765LFKMvO1cgBE6I0Z5Ia7TpPZpCSugMjgQuk49HlclXxacJVSXfIEEbv6gCnD7tdDyz8hntORZTIGzxEvpDyhcUmK1T565rok+bEBZHMNJdl7vXGTSNF4iopMimU9qyQLTlK1TJx9x1QbCr9Sqe1JTpJwSmbQafEW8g0MstUJZk6Qq7T9u2wYebTdnh2XX54VPsmAiy+m/eCaJnyfFV9W4Q4haFbcKM7RMuJh+XHfa/nv0x7lnxQOi1p045dVCYfOwWi57/cg36XRYfIVi+YTl61oruTNfsd82A98EEK65jpu1+G6YYNJNaJikxif5doGq0LgHjm8fjjY1vVBwLj4ICvzsepfVvop8Q1DAp1MBUKCcm1L8LtLYN4LyxkyhPKAVqrIDxtql/Ew77w2FCGYc4KNACJrRMwPINyytm2sQHz+TTj5J5I+b5v5li3r3UFtgy9x95P+oMsZLiuEZFQZedM2xFoHy8QW5O103X7HnHZvJZDhzLzOST16pymHxxVjteuTmcrAPQEJxMoIe2iGVM9kIB1DqDx0Y7btJxOKD7gLKu9T1j5RPehlNW8l26hxu3qiHdgM9N+VyitpiSLy7LlnqGxC0jZmKUk8LL6zRhHypEnLHEYrICrgJmIMA29CzeV3nZTqnK5AuVMcPb4NftLKVoluKdDUbokTDwip4igCYrg+nDd8WQ2iHY3r/ZKwOs99i2Y5rGYGTDSGN9v2ruAgzxrHU4ad1l7o9Dfwa8HMh+YW3I4OM5yus1f3/vp4J9QzkKRB3k99XZ6H53fLbqZScHysvLFlxmqIkzyhMbBW+fEmeNLXvWC5x9Qr/R1myQlOc9R+0PI6L0E6LMOzVpt3wegBOmYIsTJ0juJ/ddavenlGKr5GZeVfJV8WlCNfHGR4A4gXLHRBAJwtzkKAgoIbCbw9Koi9aNct11mdlF564bB7wMxQszSuKNiHr4hTOtebw8A0FeBKOUbOGJNzx3XQJck9azw/Le09Ghr9g3aEKarOT7oiEDMeAqUflx6ZD4TD7awAk/BNh75N5qGf6P3g3YnNvs/3nDwhtiteNWzzP5bMHdcDshvoKPJYXgMaxuGM7f5Xwc1H2Qcm5A4A3movfcdx1ZA6RcK+WTH6jq391HlJd8pKyuAShFbViOvNb/qctIDHy8XE7leFLf+tu3hL8XjFqA3x3yO9x7xL0VtZNGAVj/Jn7S2ow53V3IzToF2O87/vn6VD1OmnxSRW38u5Afu0/Z1wpP//DL8aN//Agb+jf4h1747At+7KErFlyB+4+6/6PPUkcovrCHfh3kGQQ5jsm3UbLIDa8fLmxaeAyEQledZ/XMAx4s86DNGXV0ZVkynN2G7yYcv/uIu7X1pqyUvx7qYvLJh46eeHRwKi6TL8bejF8R2Qw/v9WTeRqGx1qbJrdOxrKTl2n7xY40pAZOySfXVI7xKoAok5QyXl3eesKHu/jtwb/1PRaSgm20AUSGUQiy67pv5IlNr4OA4P6j7sfth91eVvvGthLOEyy8RBJZwm8LOoWPKw31xeqHVIZSRaaL6677+SmfV86XuOprdbLtR8zkKwlKPoIcLfpZ2fnvcpt3Xw2pBujhnr+tIZ6Cj90la/2Ud8U5rD4Vn2FHQ7Jv5wvB/b3LhQUoe48kXasmWnLvbA3XlkWsGAzPoJ8pTmEVj8lX+TocHpPPVURTEKz2lJdfmfGVoIDX9X6iGhd26HTHS4GQcKJK27hE/XX3NAFMyrutMpPvk+SZUUUVEagq+QYJ/INds13N3CcjSvQMW3DY8pUKscwS4bforvv1WV+P7B8D9Vsr1+U3uEZ3z8yiSiIo8jyjQlePxZ2XY+REsTHOmX2O8ZwVy13XzOS0ia20nzQxi7FvUjUmFx4KgMjWqwgmH3+GV4ISqh8HgmK7b6Ov5Hvo6IfKiktHQNFWnwY4ZdQ6Tkja1B8Eq9+xbUdMbp2M3x7yW5y040kDwr4x9QoACp6FcXgdgJTLdFm6cqnpoo8EhMZw0fBYfJtzm7H/7ap7CICPVYbYErc9yxVz+MMbfxDOf2XGVzClfQo66zrLboMAGG2tAgDcVe8qOHt3PBzY4+uglGLZmmXYmt+KjJ3BnYffiZN3PBlt2bay2/uokR8oX8XR84RvDnDn1av3uxoX7nohhtUPGzC2TRzwxqGTV+ozV/MxeXgl3y2FtUK5sE3TQFj5ddl1g9+Ga7z/rQgD0YwhM9BzSg+md0wXjo9q0Cs+Hcvx71fXNm8s6zmlB4eOPVTb7zCE829d8HdFCdBQpJjeva/7d9vYWIbSOJ4EOiZfZ607X5w9++zI6xn8O++Y7B+L6/IdBta70owgecer61/FO1veAQDccugtmNoxtby6mcuclZDJt/FtPFbjKgRfWvcSuhq6Bjy2bGwDo/f/tsI2tGZbjZlZw9sKwJQKTF7bFCOmNYGraA06pQnBEvKd8t/UuGZVccGP0HoNe+mjVvIJXhwE6KMF1eUSBH1d7rsYqBi8Mp7b/Jbwd6KkVtxNyO+qL1/yi6zjEp4MVNIsee5i42+9new98hmCeRk8zngoN/GG4AAW0k6tExiDbxw6BgBwzMRjuHrcmvpBlfGRst3O5aOUfCNm+b91Rv2gn17oC2IJfdatDwQqe7fK5Kvi04Sqkm+QwFsLmXW9EoRN5Exoc2IyxCrJ1kWQ9+uoxNIFiBZLfyH0LLQkIisib43TbQD45zVjiBgfJIxJN79rPtpqzJt2PkOYCf0hguIl8y4ZNCafLb1Hk8tIEYzJFyBFUkr8JR58WTGrmn4MMKFirWVhIfkAdy2/C1k7i5ZseBwgGWysUgCdjVlgxd/8c9sNSslbD70Vtx122+Ak2+D75v3P3HVTtN9PurG+X4xhd/CYgwe1LzIIDEq+Jk7x0u3OS9/+27d1Jd16PmomVhh4vXFRTZQwUJsf1gwTN3PDXYXJMyufwUn3uwy+rJPFhJYJOG+X8z5ezygCcgKcOMGrGQi4V5Bt0rI/h9UPKyu7aOVgayBQ+95j7qFFV4oluPc0qiUI7l+nZMY0v88kTD6TZ6jirkuCI6a22VGHG+PFBOu2qd8OCbLrFjT1hX1Tcb+3OF8Hb6AqxbxGhinkAN+LOlt8KVfueyVuO+w29JzSvX1AFgAAIABJREFUg9OmRCco4kEBvNQ+2t9wfiGEOR8F+X6pp4B7d8u7OObeY3DDSzdgVMMo7NS2U9ltUEpFJp8mAL7QJ65TXxnqZsle0LWg7PZD24rjruv9v5UQ3P767W5MuArBvDyYvL5+3J4AgO/s/h3jNcQimNoxFZ8fwZ6F6q4b9m2cMu4I/7dOduW/QqaWuOPwO3DFvle47X/U601T4Bb7TsoBRWAw4ZUofe0TAIQo+RLasqO8M5M8Bz170wU7s9m2cGH+bf+4ybshVnt81wz3wWLBseQvSdBf7AcAfGn6l2I9h/JpBMGVYd8oU4xRAK8X3dA8vBcIpQR5QlAkmnnHsz4UEBFyims/LPQN9VnvIjc9au0e3zwegClBRxVVfDJRVfINEvjJZd6IeZjWr88YFiB8Gg6byK9rdoXMMAsE4Uw58mSdJHjxEaUlbh0b3q5Y2BAp5N79e3WSTLjrCz9hUwDn73K+4EJnWWbafUU0fD8mX2RJ7dHOus5BU/IROQubQdlYAjA2v1xgWEQx+XjM75rPNWqaQty675t5BD7IbcSLa1/EpNZJieM28u9q0tAG4LeBdXDrPmoWw2dO0MetHEz4Sr5Sn8/kk13ERzeO/kj7RCjRxxeaegzgZF1XXS9D7cptK5ViNx54I87d+dzB7mYiuEom9576C/3+8RsW3jBgfeWfmO0Ji0yheOaSM/1zMovtkwLGTlg0bhGe+MwT5TNyMg2xM9J+FGDvLZ0K5tZt3Lp216K7hPJt9e48REHQkMByn2TunoC4cx2fvChcyccz+fL2ALgOE+IrJHIas0DSNb7RSR6kHhCZfAWic2gOjtQYFt9X1r8S2gYfk6/JqcVxk47DniP3/P/t3XecHVX5P/DPuW17380m2Wx205NNQkISEkISSIP0UKRDCIKKFKkiqAhIUVREBEQERTGICVWDXyw0hZ8gSK8KBBKaJJBet9x7fn/MzL1n+ty2uzd83q9XXtm9d3ZmbpmZM895znNch2Z7EQDWRSM4bs8b2Nq+FbObZ+ck6GSc32TE/t1prsxNZqwxJLXIZSKhFPtnv3jIYoflshf0eyYBXHnACQCADbsym5E0FTAPp7arB0D/VacFtKwTRxxfrWZPan9TZwxRdBqu63GeqC2qSs7E7jSKI+Hw8/Ca4RhUqWVI7ejY4brufJB1Q5M/X1erddK+sfEN23K79bfSNfs/zXNJwmeEy/dnfD/wutT32XqfNLJRu9d4qch8POTqGm8L5OrnW6OTOsgM14DzxBtBs8683smox/2Q+r55Ddc1Rsuo2zFlCkvtvA44BIH1+5ZPqps89hKm05HXCBMjozoSigQ+r0REGE3l2vY5XJf2Jgzy5dCPD/px8ufOyv6B/ubEwfrF3mc5rywrQyTgsETric+obxLEUKn1dIkO79mx/vqFv+LWSd/2XOZNZWbiZAxS7zUM+Zxo1Z5SCYkT207ERZMvUp5XC5SbX282wzeN9QatcbFy0UrbY/nqiQ3Xm2/Y3YZ2GRfbkIgry3p/v4w9fvqox/WAlfk1PHzkw6bfI/qwwK3KrK3p1FBxMr7ZXID6hXJ7tmdp1C+YnjvGjWdcrxMYSbQDeuPD+hnPGTin2/YLcBmue9JqYPYlwCXrgXHHJB92yq6c2DgRy0cvz+9Opsnoje9oGIP7374/+fh+fffLy74aR48RUFQ7Q9bvWp/z7XUHo4B/NBz1HPLiRkJgR9vxQKQIu5QZElcsWJGzfcxGVEkJu39zauIb67C45PEpgF+1fxh4/WnV5HOrkSusv/oP1zWeCSs3Nx2eNXj9XTZVy+A1rocdlrvIq6df7dkh5nQd+1rrMttjQbJI1A6nTn2rbtsOOmJBZWShlobjiAPY1rU77axyqxeLUzer725912PJIMzX320Os1ZnXRYgWRcr2HBdp/c/31kuXm0EY2/e17Pks83U7zINr9ZCMWvbN6FPSR8Mrh5sWralzz72/UmeCxK2NqXX0EZhabtaa8iaRk0oPw+oGICSSAnOnnC267rzwTEQqQdm1HbjHkhERCRw4Gla/2mezxuZfE7nj2E1w9IqByE8Jt4I6SOhrOU6jfNjJtTtuZXjWV2hfddLI8HarE5rUd/rCydd6Pq3xveo1iH5u1J4fF6mxAGvq5P9OfX6oAY6recdoXc8vtm5GW9vftt9XxReQT7jPZnRNMNzn1UV0bLka+BwXdqbMMiXQ2qG07o95jo/bjXajm7VhvG5DXtMrcB/+943IHqGnMcFO4hkRT6f/akrqUNpWrU59P0KG8MAvDdguqgoPY2GXAwz8tpuwrJ7TVFz5qHxPo+uz3x4TbpCjW3mfXB5D42mrfpNkFJ61rcxlrUGm40iwtaZyIz3p2tnqh5lJj2j6ksoipi3fcOLN6S9vpzSd65Tr+8Vju8GoiX4y3t/wfPrnwegzU776vJXc17DyHfXYPmOnv8fYPBBthli1+9cjy3tW7p13zJlfPqfHXgZbnrppjxtJdUcNUb27ejcgfveus+01OAq801goTCCfJncrBuBmHiVFlxShzNZSyJ0OyNLQKk56HUFUcsurE1owcqRtSN9N5NWkM+lo8ops1y4PJdcRs8QDpemAj39y4N1JBpWLjZ3OBmTaBidQdZMvqVDlnpfRx2aj2oAbnofLUMlUJBPWajLNC7cLuqT5e9Me19LwxLbQtqkU9VF1T5/47e2lLXb1ma8LuuKHy8twTfW3m97Ktsgn5Evmu7sumph+nzVt1Wzb7xIIbCzSwuAZtpp6HSEGe21j9q3YFD1IPsCSkDdOEZT2XrCIZPPO8iXPP9IidaqVtPzceX9NoVYhMCzJzyb1bDwTCQcMnydMjp3I+Fdj09af/U+MxhzrDktZS074c89k8/QYfliZDVRmqmkjaUmn2U72dQ9Vl/LB9s/cN8dY4SUw3N9PIbtq/vulcmXWt7/Cdt5JxHs/tM8CZ/7PpdGS/HQEQ/hymlX+u6x0XHcp6Q++eYwyEd7Ewb5csirQex2sgklbza8L3iBMvk8plIXlv9V81rn2R5za0jEjQQInxNzREQc99kt8GT0phk3jn69W6b3yyF7y6tAeTbDZUMu+x+2fPZuQd18sn7/3F6nMeuqGliuiFU4fg/sG7EWaXd+P4yXH+9MBQKymqlVSNN35yt9GzJfV45Yv8uhTm247oVPaD2qZdEy/Gb+b3pgz7TvoynbtLKf43Lz7puHVz/TMp6un3U97l1yL+5far/B7B20o/71HeuSj9w0O7fBPvW4DZdpmSK3vXobLn/68uTjqxavwmnjTsvpdruLcXOUSUPWmI97Sxh46qOnsqpZlC9FIpU5/Ogu95se47z1p/LUOen2ebfjn8f90+0PAORvdl2pzFrv+DfGDLhKkKW6qBp//PDjwNt1+8xTw3XtFy3PyZgcnlPbIKcMO8r2vJuQMnTUcbiusq1GS4eSwe9mWQqgVm7EJ/rsjw0lPX8NcfLUaOfZ4LPNPDSC9LsrtGtB0Ey+9RGl/nGOJgmzMsoIeNfk0jvT9HbNrIGzcrcDUiIhBF7Z+YF/rULj25ncV3uQL+oxFFogNZOqU7tfFqeCz8G73/PH+pkPKB9gaytKALtDIZ9JN8xHtd93SR2uu+1Ecydb2pNiqJPFWSc+0n/vyNMIG/vEG5Z2esDAufp2GZ2M6V6PnF5hbVULnjzmSd+NegX5rBP5fXnsl02/q/UVrd8RoWTULhq8yH0byufjd65vrmhGLBzzDLarRtaMxN8/+DuAVEco0d6AQb4c8gzyuRXUDngSCpJ95jVctwTazY/TXoyoHWHqJb5z4Z04a/xZjutJnqtLvBuc4VDY8TUvaHaexTOmZ2kZ9a/SCQg5T7yRWZDP7+bX+LysjS/rBVD4BG27g9t3xmgeSQD7tVfgxWUv+g9zNWZgtPZYuyyegMAD5WX4LVJZYpncIBsNojaxFvheqmbH0yX2i/zhQw9Pe/25IVGMdmDjO6Yi1UXhIpT7zBKdLxVFEdt39Lrnr8PYO1K1hba2b01mUAypGoI5A+dgRO0IDKsZ1o17GpzxbT7/rVQGp39dqfRJABjzBYT1c9xH2z8yPd9W11awvb3qcN10GUGCMzf8Gac9cho27dnk8xfdx7jxLhF6A33/M/HcJq1ulHV2ddV70dQ5qTxajsqYTy1Yj460WAJorWzFaft4B4C9s9Rd2gnGcF3L+XfwEXd4bkvl9p011tnpdB31aCI6ZR1GlG0Y5/tAmXzKteGpEnugQA0IFIWLMLnvZCxsnOK7P+bngYXvXo11UW0fs6mTmutOPHXPP4Vz+ZRsZzoXkHirKIYbq7Vrku8sqA7f0yCzHGfCOO6swQHT7gD4MBLBO1veQUNJg+9x5roeh5egvlL/kh96uRPjOyuE7Zj2qs8olOUdh8Iq6+oNQT6rkmjqe2i0z/5cXor73n8Yn+3+zO3PbPwz+fTyHOEyrNr1XvLxI4cfietnXZ/OLiPkGVDMfZBPnTjQK5jpd71xY8yerrapg9RWd3qFZeX9UF3snNWsfkaeHT76mn9fqX3vP9zhXgLDGkQXSsD24Bbn+0N1G0DwLGQ/xhoriquT2/5we/DyHUS9HYN8OeQ9tMVFiV6812cIihACp4873bOX8eH/PeX6XJPYqK3H5fl7ltyT/LmlosW1d8nIDhKjDzU9fsHECxz22f6q9+2zL74z6Ajb48bFZLee+ZVOkM9p6FnII+DpFOQbubMIJ7WdhG9P8a4jaAS5Po5E8JHSwx2xzq7m00t53czrkj/nquG8o9NckNl1uK4yY60QAYNv+jIhS0FwtwaMBHBpg3l4UWZDfbR9HR9aAxgFp2c5f0ZXTLsig/VnTn17z51cDsTbgX7jk4GAngyCRPSZzFS/fu3XALRe8JmrZmL6yunJ53JRND7/pO2mp62uzXHJTCVHCpbUJM8Ta7auST4/sCK7Omg9rUOvkZlZJl8CEsC6zs053qvcKTaCfP1Tw4ePGXGMbTnjhsEI+AD+JSIA73NYsQQePPxBzGqe5bk+x2BU/wkAgESd82zHxt/Y2hijgk+E4J7Jp72mSbC3LbzeE6f2jhrkM46fYDX5UtvZGdJ+M47tcfXjTMtKSPxq3q9wVHJ2U39d8URyP9bqgd2Blb3nWFaHjz/62YuOy0xqnJTlNrR3YF2nVjbD90ZZ/+zb85ThpCqOFOPV5a/i+FHHey5n1HHevGdz1rOpn9ySyhpSy9g4zbJsDnZY/sZhPyo8gjem4bpOs+sqWWfWsjC5dNLgQ/0Xgr2cj1Ow+e1YgPIPaTZzE3qh7q5omSmAddnUy3Ja/sR4i7u6LZMv5a5FdwVejzqxnlPma5DsRscgn8e9ltS3eenY09HpsXrj+/yRfm61Toqj7rutY1YZFeYV9DTVrg34WfktZ3wypbHyZKfBp7s/df8DogLDIF83ca+1o51gZbF/j84Z48/Av0/8d0bbj0vrMEtgetN0fPeA79qW9cryMBodwtILfPKYk23LOhZjhYBwyL4xUvONm1C/4bqNpY3Jn516obwyEJwahyEJXLjfhagr8a57Y0wIcmlDHeY3pzLLxpaYhxCFfVo0ao9VNBTF5VMv91zey6jaUQDss665NYLvrFQDOuk1bGz1pFyGba+N2b9DXlkwftR3s0O50Z7UOAkrF6/EDw/8YcbrzlTyvRBAn7BeKL20Luubj1wImUrZm63ZsgYb92w0PZbtULDuEALQrgwLuX7m9Rn3hLsx1v6S6MI7W96xPf/jmT+2PVZIjPNi0GLfKr9zWs/SPrmRUv/MoiVorWzFIS2HOHZiGI1/o97YT2f9NNBWvOrXRqw1pwIPbRRATPs8YiXOmb/G/qodVOkOnXQL8kX1WXprEcary181Ped1HXUKeKplK4xng+yldU0CwNT+U/HYUY9hTsuc5MyH2vq0NYai5mCD1w1dZ1z7qwSAn9Vo7YVsMuOMLRnBSWtN2vRX6L7vt8+7HY8e9SiG1thrD6e1CcvvfsNSDXuUfasp6rnrxMZw6luSyaRBVkVhJTClDNEri3h3MHfpNXiTgQuHz67co9MsGi5KfledzifqcZ0ozXKyFQ+T++wbaDlrkMp03KTTfLQs65vJp78PXdGy5Pt018LgQTHzxrS/L0+4n8/+p3Ta37vk3sy247Rp63la+b6kk01cntiW/DnjIJ/DvZ3npHD6vjeU1CMS9mjXWhInqmLm4zOkdM/aMvmUAK7XcW0cM+nUQxZx70kljQ6MsmhZskZrbyxDQpSpnr8b/ZxQe6oXDlqY1bqeOOYJx8cvHfc1j7/SM/CU683P5/4cRwyzZ9V5FWV/oFy7CVmzZY3j86sWr8IlUy4BAIRC9q+XgEDIoQe5XX97jOEafsPJSqOl+N707wFwvlA6bduQTfFoYdn3ingCByWaMLLYXN8nHGCgxR8O/QPmDJyD5aOX4wvDv5DxPhl1LKw3LdYbtPMnnAsA+HV1JW6oqYIU9jbatQddi1vm3uK6LXuvpP8t3IJBWo2hTIbrGg1q1dvKUIBDhx6K0XWjk9voXtq7F0EcR7x4MgBgTWJ3MlDdk0IA4i4t8Ftetn++vSEw6UdAYrfSQJ7TkvsZi7Xi9ALLPn3M8fna4tqcb7M7LR+9HF8c80UcN+q4tP825HBOayxtdMx86SlLurRZvld99gLWblsbaLj8ARVDMHtgsKyw5opmXDXtquTvBw44MPlzJOAdr3WmX/UU6l7WwyWTLw1u19SI3rHVpQcU1OuIV+DMKSNeDfJFSvXZTwPss9tWGkq162o0HE3N4Ky/X6FY8EB1XL+OvOXQ+ZSNiAhh1eJVWLlopf/CATi1GiY1TkKf0j5ZrztsGafqN1zX6MRSg3xjG8a6LZ536gQg2QT5Oo1JepQgn+hMzRReEvUO/rZ36UE+vTSHrNeyb2+YlSojUepx3imq6GuaeAMAHj7yYTyw9AEA5okuEnma6OT2ebejNODEJdYgVUk40+B4eh3KxnDdeLQC7255FyERyvj7ZwRkK6T9/TRGHL1WpLXtHznykZxmCtrbzJllDIaV2nVGZqNbkO+iSRciJEK4aL+LzNsuMbdfVixYgYOaD/LYqt6hEoqgtc4j+B1O7UcYApcfcLnrvlszB4UyiYpXp60xjL4jHrx9LeLtns/v1K8hZZGynHQcEPU2vf/Obi8RUhpYTnUO0umVd8u6aSp3Lq4PpLIwOkLOFxh1pjmvYMzfS7UL/Htb33N8vq2uDceM1IZHWQvMGkJhexBxh75bZ08425ZN4GZe6zycO+Fcx+FYIY/XkNXsutbhqgKIImwbnhsk62VI9RBcP+t6z4yCBfXuM94alrUtw3Uzr7MFj603aDGlUX9bdZU2XNeyrnmt8zCtaZptG0YgNVUbSvt/aL3/sOqh1VoGQiYNpw795uyh8jKMHTQQO4XA/4akbqzTmRk614z3ohRaHcktoRAOe+aS5PNesxXnW0gI16E+j7z/iO2xTBue3UlAYkMk+4kPvLjnP2oKIePRS2m0FOdPPD+jmjYhy+3KGePOwCNHPYLzJp6Xux3MkPXbe9W7WiaG21Ak9dwYq0xvltpDhx6avIacOubU5OPWTD63AFkkFMGPD7hKXVL5yacmn3LTb83suuIA73IFaibf8rblqcermgEAnX21meCfOOYJPHvCs56vAXC+jqolOsJ6oCMRYJTCqLA54OC01WRbybjxtARSvIbSd8lUJl8udcg42urafEcA+DE+3x/Wms8vN86+MfDQND+Vxeb6k0Fr8u2Oae2ToJl/+aLmKQ2qGpTxeoygpakze9TS5I9+Wc57urSAhTACG3pQUJ0IJGZ5r6b0nZz8uThcnBzSb8xK3resb/J4Ng3XzUH7ZnTdaNtj+/XdDzHf2oOamc0zAaQCMKZOgLTaDbbpdT0ZmXzxWBn+tu5v2b0X+rpKpX/b3/e4CEA9Zq33dpkezVGhvf4QRKq2rnJOV4N8c1sOxssnvYwlQ5ZY9iv1+u/e/6rk98+VMSFUOOo66aC23tRzRw9abGsnqUE+WyAvYJBvUKV2zBv3E4F0eQf5dhkjx6Klyc99eI1zyQyiQsQgX5bcarh9ddxXTb+b3ugAPfeGcpfetohDULDEo2fO2otrpfbyG/tkfQ0A0KXvbpBsOFFqz3oRIoSQw0V0ewaVrGPhGE4de6pjL7dXIC+r2XUt+74jFAKkPZwZJJMviLmNE/33SYRwcMvB9tlefWbbTQBalDKA2w65DTfOvhExS4C2OOL/97ObZ+M3839juiEOylj7/9ODy385+hac9/Slyefj0jsdvzvEhNa4WjIgFWTfp2Ef3Dzn5p7aJYQgAn0DR9Rogddc3Ujm07s5zsJx5fJWxEKxgp1wIxds57Re+JWRAG6vSg2VM0oZWKm7Hstg2OajRz2K1YetxoTGCZhXqmWXpNOYMmfR+NcacqrJN9FybTh8mPfEQ+p39/Txpyd/Njr1OhPaubQ4Upy8kfcaruu8n6lrjNFGkAHOLVPDliwKx8kRzOspspTp+Pncn7uu34gPdEd9uWyoZS4iIpIMsOSC2vHoOURPJ/Rr/Yt6LOuBQx/I2b5kQv0eXTntyozX0+GQydeltFX9gjzGMWh8Hx0noLF0oqjBlVg4hhkDZuCVk15x7Pg0DddNI7D14yHO2dluww9jPhmLhmlN0/DySS9jbL12nmuubE49aTmefnTQj9xXlGYTP66/9N1F6ZeWsG9b23jEIchn/fzyf43P7Bw0uE77XkZECJv3aHVx1aCYGuSL6PdztpmElW3HqvxrkiY7VELRwG3EkjL7rOVhJURfaa0/r2TmeSU8lEZLsWrxKnxvxvcC7QcAiJpgw6GNzsB7l9yL2+fdHnj9RL0dg3xZOnbksY6PL2tbZvo9rJwfnWpRuNWnuGfJPfjxQfY6UCUOi3sG+TLoBTtz/Jm27IDdIXMDx0vIqVZeSCDkUJNvW47b3kFm11UzUIJuXjhkITolYUYzqF/12wW/tT0WCccyrjdnDfLt6tpl+n13GhmN9SX1phuO5OevfK9OG/Ml29/tU78PWqpaMLFxYkbDpFstmYKXP3u16feES03AbqG/BUX6zNVblJpBy9uW56RHOFPhRBwJCHQBWLn4Sryw/gXH5c4YfwZmNs/E+RPP794dzMDHShbfSW0n5WUbAkCnw+MrFqzA88uez8s2C0UIEq8Wp87dvSn7U92XnyjZUGotN9Pye1L1jTLJaqwvqU9mEzWGte0ZZ9MgEymVKTW7pNLR4pb97pXx/8tDfhko6KHevKpZTMbjTrMzphv8Vxc3arAGGaXQYakn5bVVY33FSibfvn329cymi+t/sz1Hs3EHKVOR1vocXvCV0zMPZDlSPoc39Jmn09GvzH2kSHfLZqKoPfpoFvW4b1eG9fmN9Bg3oNq0nNPxHrW0E9UOViOo7joxmtJxmc4QQrdjRg3y3bvkXvxi7i8AALE0JrcLiRA+2fkJgFTHoJP5rfNdnzOG3xqCzq67NWAw0nNd+ozVIZ9rVjQUDTC7sj/1emTLPMvwsmlk8oVFKDmyRl13p0y1XIxzurVjX/3ODenrX5Mxoa9HhFLFKFpCJWjVaxt+oekgnDH+DNPrdcqENToIIyJse14drut3vWmra0vr2LcOT3ZjJNOMqB3BYbu0V2GQL0caSlK9F2XRMlvacVlUKZitXNzqirWG6enjToeTARUDcEjrIbbHnU6FxR4nvx0eNeoMd8y/A+dMOMf0mNu07Hvie3zX53zCdp54Y0+Oe9itDbWnjnsK39n/O6bnThlzCo4oMoZqBdu+dcIRAIAEpOXx0qLU+oL2DO7bZ18UWwoDxyLFWDBogXcPaUDrd643/b4jlP2tirqGU8aag3wNsWr8btHv8toz2qOZfMawH0tY6NQxpzoes90plOhCAsBP9jsCV79+G5b/ZbnjcjObZ+LG2TdiQMWA7t3BDJQqAd0L97swb9vZHDY3jA8ccKD/sJbPAetQ5t4U5DNYE5PVMhQm5amJEqzZyemK6J0XQhoZQtq1QJ0cyqrU5Vrt9o4awUqnLPQp/abgsKGH+e6n2smiZu4nM/ni9vB2uqUt1CFdRl3cIEHP3dJcZ8npfTD206iZpg4f9SuWXl+mfcYPVmk3cNmWUujIcXvF+g4tHrwYiwcHnzk53a18feLXfZdW22+zm2dnVcu4NzFq8rkF+fwYpUq8zn8RS/1L9b30aw8ZGXPfmvIt3HrwrYH3y23yFvXYGFE7Agc0HQAgvSAfACweshgREcE+DfukNqm8B2eMP8Pz7xPx9Dpkx/TVgkEbS7P/3iX0dqJj3VSlo/rM8WdmvS1te6l12uoIlmVYX1PPtC4NFeHbU76N+5beZxoWO69lXvJn4ztmnCNPHHUigPSv2bv6aENXS0vrYZyVJ0eqUaL/fHTTLJw+7nTT99spSGoM1x1e2t92X7g7jzWsg3ZSec0wTFTI8lvg6HPiH8f8I9lgePzoxx1PGBGBZBtL7dkujhQHrkGncjp1lXn0QDxb7J9VNKFxAiY0TjA9Fk84B1HcMiRUbkE+Y8hrTTyOu476GxbcvyDnjeaQ5UJTEatI3iCpjdV0txpxCPJJAHHL9vYoQ9tWH7YaC+4PNjHE2PZS/LskNUuuMZRsfut8TO03FSveWIFfvPKLNPdac8SwI3DXf1Kzk7WH/Po13RmjdEPK3UnIchPQUpZerSsnbo2SCyZegF+//utAw47yxdi3YpgbKb0hIBTqakdCAPdusp9bThx1IhpKGzC2fmxBTLjR3Z4sTWUO/PDAHwaelGFvZz0Sg0xq0W1crh+2oUGGmtSQs2zrXoX1jDVjD4bWDMU1M67BgQMOxIErD3TsKCstSr13MWXWQrebkhtn34jn1z/vOJN8uua1zjP9btwQqpkgyf1J9wohQoiICLpkl5Lt7b74Q4c/BCEEHvzn94D1H7ovCG3o9Vnjz0p9USCyAAAa00lEQVQOS1Yzpbe0b/H82wE1pZDbgb8Uae/1z+b8LMCLcbdND2DuG3CGUj+72s3tLLcyLVmJdwCRMA6saUsGeoIyJvfamxQrx5IRCLts6mW+f+dXFgWwB/LUz9NvArKb5tyEddvWYUz9GN99MWkYCbwDTKwegee3/BcA0FrZiq/t+zXEZRz1JfWmxWNpZjCfOuZULGtb5pr5vGzUMsfHDXG9ky4qBTqF9M2Yqo4lgF3Axdu0UQhBZ0B33LZ+DnYsP9C+Pfljth0+hqJwEVoqW/C1fe2TIYpyLcgXSbNskNDvxSoiJSiNltpqxx029DBc+pRWyiaZyRfSZkxft20d7nzzzuR3N8h5fVbzLKzdtlbbZqxCqYlq2ivb3zl1rEX0IF9TiT3Auallf2DDM777kwnjdfpl6xd6rWUiNwzy5YA646L1QmpQT4VBerb9GOub1rgf/rn+3wCA4ph7b8Qulwk3/DhlSi0ZvASX7H+Jw9JmtUVONfkEQvpQIYFUD0pXzoN89hT/cQ3jAABzB85V9yit9TqmiieARO0gYC1QHirCjkQ7diH1vqWTJRW1HJFq3ZSqoiqcte9ZGQf5Blc7TD2f4VexGCEACVMmnzXLZEjNsMxWbuL8+Rwx/AicPObkHKw/e9WhHabfe0OGUwgC78RiQMKcoXD9rOsxu3l2QdTgszI6Aq4Yf3betqG+K9cedK0tIPJ5tl3JBj9nwjk4esTRPbg3ztTT2SEthyQz5a3UY/TP7/0ZV0zznrTCS0ToQ5qUx4ygyONHP472eDvm3jvX9DdqFlppNJy8yagrcr7ZqCupy0l28DPHP2O7kTVq2g6vthccT7cTIIwwbjvkNvz+P79PXtslJK444Ar87s3fIRwK442N2lDRaCiarO/V3jAcWP9kcj1Rh/aKEAKnjTst+btav2ln507P/QoJc9a6V+2nIPbox8IJo07Iaj2GaNj8Pqcd4Amiqx0oKsVg6+zOLkz1u3IU/MiFbAO0hpgylM+YsdNv0g0gFeQTDhmBBmsgb3yf8fjL2r8A8M7wBbQ2ZiafvyjS2qbh4lTw7MHDH3Rd3u8z/ZJlZIYQwvZahXJN8DumjNIqoxKlWHLAuVg4eKHn8kKZNGFk7cisOtviekdO2KFtps6sbJqMJQvhUBh/OvxPjs8Z35vGsr6Oz7vZrU+2sXbPp57rBezfPyOxREDg94t+73qfqrph9g2Yfbf2nvtluqnBU6cgX1lU236dQzBtY5G27u9ND15rLyjjHOY3xLd/efYJCUS9EYN83SQeLYKR8JPOTLpuRKQISHSgQukNi3nUm9md4U39Af3tPb5BC59WF1ejqbwJTeVNePaTZ5OPGw0DITOrhxSE083J0JqhtqxJ410pcqzEZed0sUhAIK6/poqiSuzY/Sl2ZZgdUhILme5UMykK78Z5wpHMvhcl+kW9K64U+w1FcMqYU3Dnq79ChxDYv/mgjNZt2ju3DB2PWbi6i3oT9GbMPvytJ4UHTALWp3pHb5l7CzriHaYZAAuNMcxqZHUugsfO1DOz04Q+n2cfR7Tzx7WTv4N5o3pXgM84Ei9s0G5eLp58ceAATDYBPgAI68e7UzjMLfNOvcGuL4tg6ZCliIQimFE7Bre+aa/NmitOQ6lG1I7AXQvvwsi6kbbn0s3WTEBiUt9JmNR3EnZ0aJ0fEhKHDzs8mYE39g5t+Jo6HNU63FYk/K/HarZUkGGN7fo1+oKJF/guG5TrcPA0DW0oB9ZqP39z8jexdMhSz+UzYUw6UlWR/g3ttP7Tcr07GTtwwIE5WU+xkuVrlJ8pcmlDq212of9stDGdZh1Wv5srFqzAqLpR+OdH/8SiwYvy3sEWdGI5a5CvpbIFkV2bsKZLy2yzlu7x3a7PcO54PAGEgDBCrrXMTbpSJYGumnaVx4L+KkpCQCdQHHHYx+b9Af22oDPAeSdbRkC5b5pBvvcTuwM3163fsYQ+sigkQmkFkJcMWYLbX7td74Qy1pk6FoyEFbXN21hmD2KXxrS/VRNiDBv3bAQQbHRYujr0ocB5yYwmKgAcq9VNupRgTS4ykIb10bLSYnqj/euTvu5Z6yPIDHdOBlUNwvTOVO9LedgedPIasnL/0vtx2yG3KY+EUKNnK2yMhPMW5Ava0BF7tGE+9djms6TGsUcrkUg29IS+3V2Wt/uaGddY3gdnccsMlk5BPnUyFKcJE04fdzp+MOMHtsdDIoQZTTMsu55ZwLlYv+CrtWyEEDhv4nlortBm7arMQwHbr+zzFaxavCrn682ESBbeBo5u0oqSz26ejf377d+De6URpeYMpmlN0wo6wAeksn3THWaUDrU2aHNFs8eSnz/GEMXaUvvseT2tSL95e6tIu3H1++zUAH229dkgtW2nU/xAvcGOCYmQCGHx4MVwmmS+OwqBj20Y69h+SPf6rN7ol0XLsHTIUtust0YmvTqiwVrjN0jDVAiBr0/6OlYtXuV/47p5bfLH0fWjA6w9mJx9Nsp5Z1bzrLwEgrbqx2+1TyaZIRaO4YxxZ+CeJfeYajjuLdSAnjFbaVXM+fM0JjoAAOzcaHrOKSNODXqM7zMeReEi3Dz35rwOezYmRrHVgHNhzVpLyATCFelNrpLOqAUjky8SMDtYKEG+IQGzT93UV2ifR1XM/lmJSOqx7R3bbc/n2vpdWm3sdIN8HfpIrTOGHJnxttPtgD53wrl47sTnTNcBqX7mlqxWALZhxACwRZ9dt85hIgxjQsAg2YXpMjK8ORyXPq96PuXkc6KsuBrYtQ4Xx1qSw0azcd3Mn+C1z17Dzs6dWL1mNYbXDM9bfa0wlBp2YfNX5qnjnnLsyTRYMweioRBG1KZm58pXMWcjGFcR9U7T3lQ/FNj4Mv5WHqzwquN73NWZvMkYWNmKj3d9YhseHbRx12XJACxyuCgfPuzwZO2N5aPtEyp4FUD+6ayf4uOdH2PxA1oWhcw0yFfWCOx8H3sq7Q2VrXEtKyMXN0BCeT9unnMzZgyY4bF09zIaNmuiqRug62dd3yuGwqo90jfNvqkH9yT3YnmctXibPunGORPOyUujs5AZQxQry9O7OekOJTHzeXJS4yTP5Xd0pobYuw3pDSquT5aUztVXvcHeo3SUyIj5JnTFghVo6IVBVTeNlalOKSEErp5+tW2Zo0YchUfefwSDq1LlI/Z0mYN870eDBZWcrn9OtAmzOtAvVpV9UFeRj2Feuai76OR/Me1GvT6N79Pp450nhNsbqENzN+72ziYytTuqtU5MacnoU+VzsjE3o+tH454l92BI9RDc+op/Zms0HMWV067EkKohOP6h49FW14b3t72ft/0bUF0E7ABKnbLpnOjnhGnVI7MeHdGllx2K+kxAaASc8smYnXjp4PSydXf1GwO8vx6tTZPT3uagykH44ugv4sjh6QUI1SHaon448PHDQG0r2rYX482d76GyMlhH6CbEAYRR61CT79oDr8XqNavz0qk6um40Thh1Ak4efbLj8+MaxmUdQCbqzRjk6ybVdSOATS9jR+OonKyvqqgK05q0IRR/q/8b+pV798BdMXwZLn1rRUbbMmXFWWJC6UxnDmgz7lXEsp+i3o8QAg8e9qCpOLeT9Uh/dtaVi1bi2P9LDTcoQhdG14/GigUrUBIpwZEPHokdkcxqe3RZMvmiDqnvKr/A7lfrp+KWz55OrS8cRUtlC4ZEarCmazMiIrNhxZOHLcZzL92Mmnr7EK/Pdn8GIP2eSidCadxNb5qe9fry4a96gPjeJff2igAfYM6w3Nsy0mLdkNlkNMTJrjJHQxRzqbY0db79xzH/cByWqhpbPxaLBi/CuRPOzfqYTST8M/lOHHUi7nzzzuTvaufWFiXgmFBqsD6w9AEMrRlqW9eNs2/MWwZ8toJ0NB7Q/wD8Zv5vTCMAzp94PiKhCLa2b8U/PvxHzvdLjlwE/Pf3mDs0t8Ngc1U2Qv3mZFsv0E1L3wnYsOEF0+yon2dqbbxrZ16L1e+s9myzCCkhhYBo0DKVjM9+a/tW27I9lfk4snZkWuWAjFm571p4F4bWDMXV/7oab256M61tntR2UqB6of3LEsAOoDgWrJNOJCQQAmochnimqythzK5rPz+pNbP97hdyYVTdKLy07KW0Exx26cN8Mzk/CCFw/iT7qJ907Ihq15yy2mE4Z/ZPcfhHT6G5f6rDJBaKoa7EucNss56iXusws/Dg6sE4d+K5We2bm3AojIsnX+z6/J0L73R9jmhvwCBfN5nSMhu/fvtulPfTZt5cPHix6UY8G34BPgA4fOo3MHjovGTR63Ts2pMAclTSQEDYbgS+OfmbOR1CY2itavVdZlyfcXht42tprXd0/WhEQxF0JrQU9NZqrUE3vs/45LCPHcgseNYlE6YWv3RpLI6qHRVo+GWdy83gkJI6rNm+GX3KM2uMnrbPaZjXMs9xMo99GvbBK5++kpMboHBVqgHWWwJoBnWoypTGSaYM1Z42rf80PPnhkzi45eBAx0EhKS3Nf4adUy1S0nTH8NG06deUITLqWPfHqjhSjGtmXJOTTce77EOWrC6afJEpyKfapgwRK4+VoyJWgW9N+ZZjgA8AZjbPzHxn8yzo8L2JjRNNvzeWNeLq6VfjpQ0v5SXI9+qWtwFo181eqRuCC9+ddhU2tW8KdHz0Ro8f/TgiIne3LGpHwOi60Rhd590GNUreGFnyw/SJxab0m5Jc5stjv4zbXr0tp/uZrkzaScYQ30v2vwR/XPPHtLZ14X4XBlq2U5/FNhywgyIx7hjg1VtQWpl9rba4EeRz6IQoiZTg5ZNexsr/rMQRw47IeltBZDKCyfjeBZkcJluzmu33FvNb5+OFDS/gS2O/hFi0BONa55ief/r4p13P/5v172SNT9ICEeVWoCuREGI+gJ8CCAP4pZTyGsvzRQB+C2AigI0AjpFSrs3trha2aU3TsHLRymS9gu/P+H6378O4hnEZDRXuiqdO3NcedG0udwkAcPyo43O+zqDOm3gefvfm79L+u7H1++CFDS8AANo7Uz252Qa2GupGAptewQX7nI7HPvkX6l0CGncvuTvQ+uKyy/HxgSV9gO3vIKQXpk1XSIScZ+sF8PO5P8dnuz/LSVCuT2kf/OigH2F6/16Yxae8vmUuwwF6yvGjju/R4yqfytPMHk7HN/b7BloqW/JWRmBv4Jcl1yNKtZo7DaL7M2iaq3cDu4CamHeGxa0H3+qY6balPjVcKBqK4qnjnsr5PuabNtc6EPKZhdHP+D7jc7I/VudMOAePrHvENNlHNkbXjc7pMC+pDwHNp+bKZjSjcLO6c1U+4Y75d2R1Dvtox0cAtAywx49+3BQ0PXvC2Th7Qv5mf8+3fGaydemd4pGA7327PuInFzV4SxtGABv+gfqGNsfnQyLU69tLlx9wOe54/Q5MaJyQ923dMPsG22P9yvvhxtk3uv6N12zNm0sqgK5dqC1Lr+YjEWXHN8gntJkEfgbgYAAfAvi3EGK1lFJNCTsVwGYp5VAhxLEAfgDgmHzscCHLR7aa1WFDD8v5rJAlQuuBa+sCDmjKLMNlSs1IPLP5PwjpxY5nN8/G8Fp7gdbulumwp5vm3IRvPXER/v7Rk3ilOtU7ZQQHnHrCgvjOvFsw44O/Y8mQJTgZ7rX1gpKNY4AN/8/2+JDGccCGpyDycMNeGavM6ey381vn52xduSSEQHNFMz7Y/kFO6zyRt3xmdC5rW5a3dRe6wVWD8e7Wd3t6Nxz1bZoKvHYLpo0OMGtjjh3YNBGta+/D10d71zua2n+q4+ND6p1vPHuLCQ3j8MKnL7s+X19Ug+kDDsIf1vwBRVXZZ93ct/Q+xyGQ2RjfZ3xOA4grF6/M2boAIKJn7B/ccnBO10t2mQZJzp94Pq57/josHLQw+ZhX4PFHB/0ItUWFlzUZFmHEZfplbPxERi0BnnkJwwfPC7S8kbFmnXk7E4smnomO8kYsHX541uvqKU3lTfjWlG95LvPoUY92y+Qh6bp02ndx80s3o7Iod/cFRORP+NVvEEJMBXC5lHKe/vs3AUBK+X1lmb/qyzwthIgA+ARAg/RY+aRJk+Rzzz2Xg5dA+faVX+6Pp6M7cWb9/vjqIv8ZYp20x9vx0Y6PTMW2e4t73roHTWVNaQcwuxJd+MYT38CytmWm+kK7OnchFo5lXSw4Fz7Y/gEW3r8Qpww/BudNvST5eGdXB+558jLMn/Q11Fbkvnj458Wuzl34eMfHrkPrKHd+9Mw1ePHTl3BXjm+wKZiOeAfiMp63mmHZemvzW2itbPXMKMibT14FGseYsnv9fP/pKxEJRXH+fhf26szRzkQnOuOdjtlPe7r2ICRCiIQieHvz272qZEGh+c+m/2BEzYheV5aCCs+7W99FPBFPDilOx5Y9W7AnvicnNZVVHfEOPPnhk5g1cFag2p1PfPgEznz0TJwy5hScN/G8nO4L2b304T9xz1t34/T9LjTVKaTCJ4R4XkrpPRsZ7ZWCBPmOBDBfSvkl/fdlAKZIKc9SlnlNX+ZD/fc1+jKfWdb1FQBfAYCBAwdOXLduXS5fC+XJp5vfxUvrHsPB47/U07tCRERERER7KSkl/u+9/8PcgXO7ZUIMor0Vg3yfX92aaiSlvBXArYCWyded26bMNdQMxsE1vS8Dj4iIiIiI9h5CiJzV0CQi+jzyz5kGPgJM1XoH6I85LqMP162CNgEHERERERERERER5VmQIN+/AQwTQgwSQsQAHAtgtWWZ1QCW6z8fCeAxr3p8RERERERERERElDu+w3WllF1CiLMA/BVAGMDtUsrXhRBXAHhOSrkawK8ArBBCvANgE7RAIBEREREREREREXWDQDX5pJQPAXjI8tilys97AByV210jIiIiIiIiIiKiIIIM1yUiIiIiIiIiIqJejEE+IiIiIiIiIiKiAscgHxERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4BvmIiIiIiIiIiIgKHIN8REREREREREREBY5BPiIiIiIiIiIiogLHIB8REREREREREVGBY5CPiIiIiIiIiIiowDHIR0REREREREREVOAY5CMiIiIiIiIiIipwDPIREREREREREREVOAb5iIiIiIiIiIiIChyDfERERERERERERAWOQT4iIiIiIiIiIqICxyAfERERERERERFRgWOQj4iIiIiIiIiIqMAxyEdERERERERERFTgGOQjIiIiIiIiIiIqcAzyERERERERERERFTgG+YiIiIiIiIiIiAocg3xEREREREREREQFjkE+IiIiIiIiIiKiAscgHxERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4IaXsmQ0L8SmAdT2y8dyrB/BZT+8EEWWNxzLR3oHHMtHegccy0d6Bx3L3a5FSNvT0TlD367Eg395ECPGclHJST+8HEWWHxzLR3oHHMtHegccy0d6BxzJR9+FwXSIiIiIiIiIiogLHIB8REREREREREVGBY5AvN27t6R0gopzgsUy0d+CxTLR34LFMtHfgsUzUTViTj4iIiIiIiIiIqMAxk4+IiIiIiIiIiKjAMchHRERERERERERU4Bjky4IQYr4Q4r9CiHeEEBf39P4QkZkQ4nYhxAYhxGvKY7VCiIeFEG/r/9fojwshxA368fyKEGKC8jfL9eXfFkIs74nXQvR5JoRoFkI8LoR4QwjxuhDiHP1xHs9EBUQIUSyEeFYI8bJ+LH9Xf3yQEOIZ/ZhdJYSI6Y8X6b+/oz/fqqzrm/rj/xVCzOuZV0T0+SaECAshXhRC/En/nccyUQ9jkC9DQogwgJ8BWACgDcBxQoi2nt0rIrL4DYD5lscuBvColHIYgEf13wHtWB6m//sKgJ8DWhABwGUApgCYDOAyI5BARN2mC8AFUso2APsDOFO/5vJ4Jios7QBmSynHARgPYL4QYn8APwDwEynlUACbAZyqL38qgM364z/Rl4N+/B8LYDS06/zNetuciLrXOQDeVH7nsUzUwxjky9xkAO9IKd+VUnYAWAng0B7eJyJSSCmfALDJ8vChAO7Qf74DwGHK47+Vmn8BqBZC9AMwD8DDUspNUsrNAB6GPXBIRHkkpfyflPIF/eft0G4omsDjmaig6MfkDv3XqP5PApgN4F79ceuxbBzj9wKYI4QQ+uMrpZTtUsr3ALwDrW1ORN1ECDEAwCIAv9R/F+CxTNTjGOTLXBOAD5TfP9QfI6LerVFK+T/9508ANOo/ux3TPNaJehF9iM++AJ4Bj2eigqMP73sJwAZogfY1ALZIKbv0RdTjMnnM6s9vBVAHHstEvcH1AL4BIKH/Xgcey0Q9jkE+IvrcklJKaBkERFQAhBDlAO4DcK6Ucpv6HI9nosIgpYxLKccDGAAtY2dkD+8SEaVJCLEYwAYp5fM9vS9EZMYgX+Y+AtCs/D5Af4yIerf1+rA96P9v0B93O6Z5rBP1AkKIKLQA3++klPfrD/N4JipQUsotAB4HMBXakPqI/pR6XCaPWf35KgAbwWOZqKdNA7BUCLEWWtmq2QB+Ch7LRD2OQb7M/RvAMH0GoRi0gqGre3ifiMjfagDGjJrLAfxRefwkfVbO/QFs1YcB/hXAIUKIGr1A/yH6Y0TUTfS6Pb8C8KaU8jrlKR7PRAVECNEghKjWfy4BcDC0GpuPAzhSX8x6LBvH+JEAHtOzdlcDOFafsXMQtEl2nu2eV0FEUspvSikHSClbod0HPyalPAE8lol6XMR/EXIipewSQpwF7eYgDOB2KeXrPbxbRKQQQvwewEwA9UKID6HNqnkNgLuFEKcCWAfgaH3xhwAshFbwdxeALwKAlHKTEOJKaIF9ALhCSmmdzIOI8msagGUAXtVreQHAt8DjmajQ9ANwhz57ZgjA3VLKPwkh3gCwUghxFYAXoQX1of+/QgjxDrSJtI4FACnl60KIuwG8AW327TOllPFufi1EZHcReCwT9SihBdCJiIiIiIiIiIioUHG4LhERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4BvmIiIiIiIiIiIgKHIN8REREREREREREBe7/A0aQbHgzm4UDAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Descriptive statistics"
      ],
      "metadata": {
        "id": "nwBbVRuL8KS2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#statistic finding mean\n",
        "df.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N2P4d6HyD5z1",
        "outputId": "e71b0635-0fba-4155-80cb-5b2694fcb189"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Length            0.523992\n",
              "Diameter          0.407881\n",
              "Height            0.139516\n",
              "Whole weight      0.828742\n",
              "Shucked weight    0.359367\n",
              "Viscera weight    0.180594\n",
              "Shell weight      0.238831\n",
              "Rings             9.933684\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.median()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OQgEsXWeELDj",
        "outputId": "0af1f080-1bf7-4a8b-ae47-9dde790d2f68"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Length            0.5450\n",
              "Diameter          0.4250\n",
              "Height            0.1400\n",
              "Whole weight      0.7995\n",
              "Shucked weight    0.3360\n",
              "Viscera weight    0.1710\n",
              "Shell weight      0.2340\n",
              "Rings             9.0000\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.mode()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        },
        "id": "_QyAaHvUEvV7",
        "outputId": "e80c2740-5db3-413c-8eb8-8cb6cc187d4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Sex  Length  Diameter  Height  Whole weight  Shucked weight  \\\n",
              "0    M   0.550      0.45    0.15        0.2225           0.175   \n",
              "1  NaN   0.625       NaN     NaN           NaN             NaN   \n",
              "\n",
              "   Viscera weight  Shell weight  Rings  \n",
              "0          0.1715         0.275    9.0  \n",
              "1             NaN           NaN    NaN  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-24bb07aa-af67-4891-9108-6fbd10c619bb\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>M</td>\n",
              "      <td>0.550</td>\n",
              "      <td>0.45</td>\n",
              "      <td>0.15</td>\n",
              "      <td>0.2225</td>\n",
              "      <td>0.175</td>\n",
              "      <td>0.1715</td>\n",
              "      <td>0.275</td>\n",
              "      <td>9.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>NaN</td>\n",
              "      <td>0.625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-24bb07aa-af67-4891-9108-6fbd10c619bb')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-24bb07aa-af67-4891-9108-6fbd10c619bb button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-24bb07aa-af67-4891-9108-6fbd10c619bb');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Missing values"
      ],
      "metadata": {
        "id": "CIVh0M2-8QTZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "bool_series = pd.isnull(df[\"Sex\"])\n",
        "df[bool_series]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49
        },
        "id": "NOJFZ_4IFlcG",
        "outputId": "4e4145aa-dc39-41b1-81f3-da8d4b3e952c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Empty DataFrame\n",
              "Columns: [Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings]\n",
              "Index: []"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-68b9217c-4fae-4e16-9024-d0da07312914\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-68b9217c-4fae-4e16-9024-d0da07312914')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-68b9217c-4fae-4e16-9024-d0da07312914 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-68b9217c-4fae-4e16-9024-d0da07312914');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.info())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yKtM1AfLOS-Y",
        "outputId": "71916c96-a34f-45ac-96a2-af1f84da8b0f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 4177 entries, 0 to 4176\n",
            "Data columns (total 9 columns):\n",
            " #   Column          Non-Null Count  Dtype  \n",
            "---  ------          --------------  -----  \n",
            " 0   Sex             4177 non-null   object \n",
            " 1   Length          4177 non-null   float64\n",
            " 2   Diameter        4177 non-null   float64\n",
            " 3   Height          4177 non-null   float64\n",
            " 4   Whole weight    4177 non-null   float64\n",
            " 5   Shucked weight  4177 non-null   float64\n",
            " 6   Viscera weight  4177 non-null   float64\n",
            " 7   Shell weight    4177 non-null   float64\n",
            " 8   Rings           4177 non-null   int64  \n",
            "dtypes: float64(7), int64(1), object(1)\n",
            "memory usage: 293.8+ KB\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "qbmZdgNkXKJ6",
        "outputId": "76ddaa09-d0bd-4cdf-d862-4eaa31f8d0b9"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        Sex  Length  Diameter  Height  Whole weight  Shucked weight  \\\n",
              "0     False   False     False   False         False           False   \n",
              "1     False   False     False   False         False           False   \n",
              "2     False   False     False   False         False           False   \n",
              "3     False   False     False   False         False           False   \n",
              "4     False   False     False   False         False           False   \n",
              "...     ...     ...       ...     ...           ...             ...   \n",
              "4172  False   False     False   False         False           False   \n",
              "4173  False   False     False   False         False           False   \n",
              "4174  False   False     False   False         False           False   \n",
              "4175  False   False     False   False         False           False   \n",
              "4176  False   False     False   False         False           False   \n",
              "\n",
              "      Viscera weight  Shell weight  Rings  \n",
              "0              False         False  False  \n",
              "1              False         False  False  \n",
              "2              False         False  False  \n",
              "3              False         False  False  \n",
              "4              False         False  False  \n",
              "...              ...           ...    ...  \n",
              "4172           False         False  False  \n",
              "4173           False         False  False  \n",
              "4174           False         False  False  \n",
              "4175           False         False  False  \n",
              "4176           False         False  False  \n",
              "\n",
              "[4177 rows x 9 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4172</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4173</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4174</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4175</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4176</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>4177 rows × 9 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9WAbwRTpjcNX",
        "outputId": "bc336670-a807-4f1b-c80e-48f6aefa76ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sex               0\n",
              "Length            0\n",
              "Diameter          0\n",
              "Height            0\n",
              "Whole weight      0\n",
              "Shucked weight    0\n",
              "Viscera weight    0\n",
              "Shell weight      0\n",
              "Rings             0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "outliers"
      ],
      "metadata": {
        "id": "IpBU0SKx8ctz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.displot(df['Sex'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 386
        },
        "id": "uF7viPnzjiY2",
        "outputId": "f44987b3-a150-4d40-d6ea-1fb344f0e3a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.FacetGrid at 0x7f06a2a95d90>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 360x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(x='Sex',y='Rings',data=df)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "id": "qivTUF_IjuTO",
        "outputId": "b84318d5-b304-44f1-c044-66123748bab4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f069f444b90>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data_tips=pd.get_dummies(df)\n",
        "data_tips"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 468
        },
        "id": "MkZyvnwkj7OE",
        "outputId": "9da22cf1-0261-4354-a677-9f73cc24cc1c"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  \\\n",
              "0      0.455     0.365   0.095        0.5140          0.2245          0.1010   \n",
              "1      0.350     0.265   0.090        0.2255          0.0995          0.0485   \n",
              "2      0.530     0.420   0.135        0.6770          0.2565          0.1415   \n",
              "3      0.440     0.365   0.125        0.5160          0.2155          0.1140   \n",
              "4      0.330     0.255   0.080        0.2050          0.0895          0.0395   \n",
              "...      ...       ...     ...           ...             ...             ...   \n",
              "4172   0.565     0.450   0.165        0.8870          0.3700          0.2390   \n",
              "4173   0.590     0.440   0.135        0.9660          0.4390          0.2145   \n",
              "4174   0.600     0.475   0.205        1.1760          0.5255          0.2875   \n",
              "4175   0.625     0.485   0.150        1.0945          0.5310          0.2610   \n",
              "4176   0.710     0.555   0.195        1.9485          0.9455          0.3765   \n",
              "\n",
              "      Shell weight  Rings  Sex_F  Sex_I  Sex_M  \n",
              "0           0.1500     15      0      0      1  \n",
              "1           0.0700      7      0      0      1  \n",
              "2           0.2100      9      1      0      0  \n",
              "3           0.1550     10      0      0      1  \n",
              "4           0.0550      7      0      1      0  \n",
              "...            ...    ...    ...    ...    ...  \n",
              "4172        0.2490     11      1      0      0  \n",
              "4173        0.2605     10      0      0      1  \n",
              "4174        0.3080      9      0      0      1  \n",
              "4175        0.2960     10      1      0      0  \n",
              "4176        0.4950     12      0      0      1  \n",
              "\n",
              "[4177 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2a940f93-302f-423e-aca4-70bee8235a3a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "      <th>Sex_F</th>\n",
              "      <th>Sex_I</th>\n",
              "      <th>Sex_M</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.455</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.095</td>\n",
              "      <td>0.5140</td>\n",
              "      <td>0.2245</td>\n",
              "      <td>0.1010</td>\n",
              "      <td>0.1500</td>\n",
              "      <td>15</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.350</td>\n",
              "      <td>0.265</td>\n",
              "      <td>0.090</td>\n",
              "      <td>0.2255</td>\n",
              "      <td>0.0995</td>\n",
              "      <td>0.0485</td>\n",
              "      <td>0.0700</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.530</td>\n",
              "      <td>0.420</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.6770</td>\n",
              "      <td>0.2565</td>\n",
              "      <td>0.1415</td>\n",
              "      <td>0.2100</td>\n",
              "      <td>9</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.440</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.125</td>\n",
              "      <td>0.5160</td>\n",
              "      <td>0.2155</td>\n",
              "      <td>0.1140</td>\n",
              "      <td>0.1550</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.330</td>\n",
              "      <td>0.255</td>\n",
              "      <td>0.080</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.0895</td>\n",
              "      <td>0.0395</td>\n",
              "      <td>0.0550</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4172</th>\n",
              "      <td>0.565</td>\n",
              "      <td>0.450</td>\n",
              "      <td>0.165</td>\n",
              "      <td>0.8870</td>\n",
              "      <td>0.3700</td>\n",
              "      <td>0.2390</td>\n",
              "      <td>0.2490</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4173</th>\n",
              "      <td>0.590</td>\n",
              "      <td>0.440</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.9660</td>\n",
              "      <td>0.4390</td>\n",
              "      <td>0.2145</td>\n",
              "      <td>0.2605</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4174</th>\n",
              "      <td>0.600</td>\n",
              "      <td>0.475</td>\n",
              "      <td>0.205</td>\n",
              "      <td>1.1760</td>\n",
              "      <td>0.5255</td>\n",
              "      <td>0.2875</td>\n",
              "      <td>0.3080</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4175</th>\n",
              "      <td>0.625</td>\n",
              "      <td>0.485</td>\n",
              "      <td>0.150</td>\n",
              "      <td>1.0945</td>\n",
              "      <td>0.5310</td>\n",
              "      <td>0.2610</td>\n",
              "      <td>0.2960</td>\n",
              "      <td>10</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4176</th>\n",
              "      <td>0.710</td>\n",
              "      <td>0.555</td>\n",
              "      <td>0.195</td>\n",
              "      <td>1.9485</td>\n",
              "      <td>0.9455</td>\n",
              "      <td>0.3765</td>\n",
              "      <td>0.4950</td>\n",
              "      <td>12</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>4177 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2a940f93-302f-423e-aca4-70bee8235a3a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2a940f93-302f-423e-aca4-70bee8235a3a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2a940f93-302f-423e-aca4-70bee8235a3a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Split the data into dependent and independent variables"
      ],
      "metadata": {
        "id": "BR75HIpZ8ojg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x=df.iloc[:,1:8]\n",
        "y=df.iloc[:,8]\n",
        "x\n",
        "y"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bfI-u_KikEVs",
        "outputId": "e05e05f5-3c68-4099-c07d-6d4466a8d005"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       15\n",
              "1        7\n",
              "2        9\n",
              "3       10\n",
              "4        7\n",
              "        ..\n",
              "4172    11\n",
              "4173    10\n",
              "4174     9\n",
              "4175    10\n",
              "4176    12\n",
              "Name: Rings, Length: 4177, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "inde=df.iloc[1:,1:7].values"
      ],
      "metadata": {
        "id": "PP7bTNI8kbL2"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "inde"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cuIzZQ0nkegw",
        "outputId": "0d126b2f-53c8-47ee-813a-b646b553e202"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.35  , 0.265 , 0.09  , 0.2255, 0.0995, 0.0485],\n",
              "       [0.53  , 0.42  , 0.135 , 0.677 , 0.2565, 0.1415],\n",
              "       [0.44  , 0.365 , 0.125 , 0.516 , 0.2155, 0.114 ],\n",
              "       ...,\n",
              "       [0.6   , 0.475 , 0.205 , 1.176 , 0.5255, 0.2875],\n",
              "       [0.625 , 0.485 , 0.15  , 1.0945, 0.531 , 0.261 ],\n",
              "       [0.71  , 0.555 , 0.195 , 1.9485, 0.9455, 0.3765]])"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "depe=df.iloc[1:,9:].values\n",
        "depe"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "itNM19B0kkGn",
        "outputId": "83c341a3-13c3-4fe9-a881-2d52bdcf274e"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([], shape=(4176, 0), dtype=float64)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Split the data into training and testing"
      ],
      "metadata": {
        "id": "LuH2z89G8rYb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(inde,depe,test_size=0.2,random_state=5)\n",
        "x_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8M3Ls9AckuJ9",
        "outputId": "4d7d1fe4-b154-43f1-cd7a-42d2f96c5a78"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.565 , 0.435 , 0.15  , 0.99  , 0.5795, 0.1825],\n",
              "       [0.48  , 0.37  , 0.125 , 0.5435, 0.244 , 0.101 ],\n",
              "       [0.44  , 0.35  , 0.12  , 0.375 , 0.1425, 0.0965],\n",
              "       ...,\n",
              "       [0.555 , 0.43  , 0.125 , 0.7005, 0.3395, 0.1355],\n",
              "       [0.51  , 0.395 , 0.145 , 0.6185, 0.216 , 0.1385],\n",
              "       [0.595 , 0.47  , 0.155 , 1.2015, 0.492 , 0.3865]])"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LBBQkTy0k62J",
        "outputId": "4630928d-32f0-4d5b-c19b-8e2739270561"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.455 , 0.365 , 0.11  , 0.385 , 0.166 , 0.046 ],\n",
              "       [0.47  , 0.37  , 0.18  , 0.51  , 0.1915, 0.1285],\n",
              "       [0.72  , 0.575 , 0.17  , 1.9335, 0.913 , 0.389 ],\n",
              "       ...,\n",
              "       [0.275 , 0.215 , 0.075 , 0.1155, 0.0485, 0.029 ],\n",
              "       [0.39  , 0.3   , 0.09  , 0.252 , 0.1065, 0.053 ],\n",
              "       [0.585 , 0.46  , 0.165 , 1.1135, 0.5825, 0.2345]])"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Build the Model"
      ],
      "metadata": {
        "id": "FIyZrecJ80b9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import datasets\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.datasets import make_classification\n",
        "iris=datasets.load_iris()"
      ],
      "metadata": {
        "id": "s60typmplDxp"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(iris.feature_names)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "br2ZlXEFlMW4",
        "outputId": "28849331-d4fe-4251-fcad-3101343f42ff"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(iris.target_names)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4SBaT0tslVFv",
        "outputId": "39ef5f4b-59a9-4d07-e062-fda0973823e9"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['setosa' 'versicolor' 'virginica']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "iris.data\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lOvwcFHflhNX",
        "outputId": "857cf0b9-c693-4a74-8568-c90ef9cdcd6e"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[5.1, 3.5, 1.4, 0.2],\n",
              "       [4.9, 3. , 1.4, 0.2],\n",
              "       [4.7, 3.2, 1.3, 0.2],\n",
              "       [4.6, 3.1, 1.5, 0.2],\n",
              "       [5. , 3.6, 1.4, 0.2],\n",
              "       [5.4, 3.9, 1.7, 0.4],\n",
              "       [4.6, 3.4, 1.4, 0.3],\n",
              "       [5. , 3.4, 1.5, 0.2],\n",
              "       [4.4, 2.9, 1.4, 0.2],\n",
              "       [4.9, 3.1, 1.5, 0.1],\n",
              "       [5.4, 3.7, 1.5, 0.2],\n",
              "       [4.8, 3.4, 1.6, 0.2],\n",
              "       [4.8, 3. , 1.4, 0.1],\n",
              "       [4.3, 3. , 1.1, 0.1],\n",
              "       [5.8, 4. , 1.2, 0.2],\n",
              "       [5.7, 4.4, 1.5, 0.4],\n",
              "       [5.4, 3.9, 1.3, 0.4],\n",
              "       [5.1, 3.5, 1.4, 0.3],\n",
              "       [5.7, 3.8, 1.7, 0.3],\n",
              "       [5.1, 3.8, 1.5, 0.3],\n",
              "       [5.4, 3.4, 1.7, 0.2],\n",
              "       [5.1, 3.7, 1.5, 0.4],\n",
              "       [4.6, 3.6, 1. , 0.2],\n",
              "       [5.1, 3.3, 1.7, 0.5],\n",
              "       [4.8, 3.4, 1.9, 0.2],\n",
              "       [5. , 3. , 1.6, 0.2],\n",
              "       [5. , 3.4, 1.6, 0.4],\n",
              "       [5.2, 3.5, 1.5, 0.2],\n",
              "       [5.2, 3.4, 1.4, 0.2],\n",
              "       [4.7, 3.2, 1.6, 0.2],\n",
              "       [4.8, 3.1, 1.6, 0.2],\n",
              "       [5.4, 3.4, 1.5, 0.4],\n",
              "       [5.2, 4.1, 1.5, 0.1],\n",
              "       [5.5, 4.2, 1.4, 0.2],\n",
              "       [4.9, 3.1, 1.5, 0.2],\n",
              "       [5. , 3.2, 1.2, 0.2],\n",
              "       [5.5, 3.5, 1.3, 0.2],\n",
              "       [4.9, 3.6, 1.4, 0.1],\n",
              "       [4.4, 3. , 1.3, 0.2],\n",
              "       [5.1, 3.4, 1.5, 0.2],\n",
              "       [5. , 3.5, 1.3, 0.3],\n",
              "       [4.5, 2.3, 1.3, 0.3],\n",
              "       [4.4, 3.2, 1.3, 0.2],\n",
              "       [5. , 3.5, 1.6, 0.6],\n",
              "       [5.1, 3.8, 1.9, 0.4],\n",
              "       [4.8, 3. , 1.4, 0.3],\n",
              "       [5.1, 3.8, 1.6, 0.2],\n",
              "       [4.6, 3.2, 1.4, 0.2],\n",
              "       [5.3, 3.7, 1.5, 0.2],\n",
              "       [5. , 3.3, 1.4, 0.2],\n",
              "       [7. , 3.2, 4.7, 1.4],\n",
              "       [6.4, 3.2, 4.5, 1.5],\n",
              "       [6.9, 3.1, 4.9, 1.5],\n",
              "       [5.5, 2.3, 4. , 1.3],\n",
              "       [6.5, 2.8, 4.6, 1.5],\n",
              "       [5.7, 2.8, 4.5, 1.3],\n",
              "       [6.3, 3.3, 4.7, 1.6],\n",
              "       [4.9, 2.4, 3.3, 1. ],\n",
              "       [6.6, 2.9, 4.6, 1.3],\n",
              "       [5.2, 2.7, 3.9, 1.4],\n",
              "       [5. , 2. , 3.5, 1. ],\n",
              "       [5.9, 3. , 4.2, 1.5],\n",
              "       [6. , 2.2, 4. , 1. ],\n",
              "       [6.1, 2.9, 4.7, 1.4],\n",
              "       [5.6, 2.9, 3.6, 1.3],\n",
              "       [6.7, 3.1, 4.4, 1.4],\n",
              "       [5.6, 3. , 4.5, 1.5],\n",
              "       [5.8, 2.7, 4.1, 1. ],\n",
              "       [6.2, 2.2, 4.5, 1.5],\n",
              "       [5.6, 2.5, 3.9, 1.1],\n",
              "       [5.9, 3.2, 4.8, 1.8],\n",
              "       [6.1, 2.8, 4. , 1.3],\n",
              "       [6.3, 2.5, 4.9, 1.5],\n",
              "       [6.1, 2.8, 4.7, 1.2],\n",
              "       [6.4, 2.9, 4.3, 1.3],\n",
              "       [6.6, 3. , 4.4, 1.4],\n",
              "       [6.8, 2.8, 4.8, 1.4],\n",
              "       [6.7, 3. , 5. , 1.7],\n",
              "       [6. , 2.9, 4.5, 1.5],\n",
              "       [5.7, 2.6, 3.5, 1. ],\n",
              "       [5.5, 2.4, 3.8, 1.1],\n",
              "       [5.5, 2.4, 3.7, 1. ],\n",
              "       [5.8, 2.7, 3.9, 1.2],\n",
              "       [6. , 2.7, 5.1, 1.6],\n",
              "       [5.4, 3. , 4.5, 1.5],\n",
              "       [6. , 3.4, 4.5, 1.6],\n",
              "       [6.7, 3.1, 4.7, 1.5],\n",
              "       [6.3, 2.3, 4.4, 1.3],\n",
              "       [5.6, 3. , 4.1, 1.3],\n",
              "       [5.5, 2.5, 4. , 1.3],\n",
              "       [5.5, 2.6, 4.4, 1.2],\n",
              "       [6.1, 3. , 4.6, 1.4],\n",
              "       [5.8, 2.6, 4. , 1.2],\n",
              "       [5. , 2.3, 3.3, 1. ],\n",
              "       [5.6, 2.7, 4.2, 1.3],\n",
              "       [5.7, 3. , 4.2, 1.2],\n",
              "       [5.7, 2.9, 4.2, 1.3],\n",
              "       [6.2, 2.9, 4.3, 1.3],\n",
              "       [5.1, 2.5, 3. , 1.1],\n",
              "       [5.7, 2.8, 4.1, 1.3],\n",
              "       [6.3, 3.3, 6. , 2.5],\n",
              "       [5.8, 2.7, 5.1, 1.9],\n",
              "       [7.1, 3. , 5.9, 2.1],\n",
              "       [6.3, 2.9, 5.6, 1.8],\n",
              "       [6.5, 3. , 5.8, 2.2],\n",
              "       [7.6, 3. , 6.6, 2.1],\n",
              "       [4.9, 2.5, 4.5, 1.7],\n",
              "       [7.3, 2.9, 6.3, 1.8],\n",
              "       [6.7, 2.5, 5.8, 1.8],\n",
              "       [7.2, 3.6, 6.1, 2.5],\n",
              "       [6.5, 3.2, 5.1, 2. ],\n",
              "       [6.4, 2.7, 5.3, 1.9],\n",
              "       [6.8, 3. , 5.5, 2.1],\n",
              "       [5.7, 2.5, 5. , 2. ],\n",
              "       [5.8, 2.8, 5.1, 2.4],\n",
              "       [6.4, 3.2, 5.3, 2.3],\n",
              "       [6.5, 3. , 5.5, 1.8],\n",
              "       [7.7, 3.8, 6.7, 2.2],\n",
              "       [7.7, 2.6, 6.9, 2.3],\n",
              "       [6. , 2.2, 5. , 1.5],\n",
              "       [6.9, 3.2, 5.7, 2.3],\n",
              "       [5.6, 2.8, 4.9, 2. ],\n",
              "       [7.7, 2.8, 6.7, 2. ],\n",
              "       [6.3, 2.7, 4.9, 1.8],\n",
              "       [6.7, 3.3, 5.7, 2.1],\n",
              "       [7.2, 3.2, 6. , 1.8],\n",
              "       [6.2, 2.8, 4.8, 1.8],\n",
              "       [6.1, 3. , 4.9, 1.8],\n",
              "       [6.4, 2.8, 5.6, 2.1],\n",
              "       [7.2, 3. , 5.8, 1.6],\n",
              "       [7.4, 2.8, 6.1, 1.9],\n",
              "       [7.9, 3.8, 6.4, 2. ],\n",
              "       [6.4, 2.8, 5.6, 2.2],\n",
              "       [6.3, 2.8, 5.1, 1.5],\n",
              "       [6.1, 2.6, 5.6, 1.4],\n",
              "       [7.7, 3. , 6.1, 2.3],\n",
              "       [6.3, 3.4, 5.6, 2.4],\n",
              "       [6.4, 3.1, 5.5, 1.8],\n",
              "       [6. , 3. , 4.8, 1.8],\n",
              "       [6.9, 3.1, 5.4, 2.1],\n",
              "       [6.7, 3.1, 5.6, 2.4],\n",
              "       [6.9, 3.1, 5.1, 2.3],\n",
              "       [5.8, 2.7, 5.1, 1.9],\n",
              "       [6.8, 3.2, 5.9, 2.3],\n",
              "       [6.7, 3.3, 5.7, 2.5],\n",
              "       [6.7, 3. , 5.2, 2.3],\n",
              "       [6.3, 2.5, 5. , 1.9],\n",
              "       [6.5, 3. , 5.2, 2. ],\n",
              "       [6.2, 3.4, 5.4, 2.3],\n",
              "       [5.9, 3. , 5.1, 1.8]])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "iris.target"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4CKp5y-Nl9L9",
        "outputId": "9803e71a-fc4d-4618-ad4f-799005c68a6c"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
              "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
              "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=iris.data\n",
        "y=iris.target\n"
      ],
      "metadata": {
        "id": "2FQyZLsFl-7l"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x.shape\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ITeNy_zQ3zXv",
        "outputId": "0da97dd7-a3ed-4e15-a34e-01aa28096d51"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(150, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C1b7eAla3_YL",
        "outputId": "d6b80fd9-7e14-43da-a076-95d84f99e44a"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(150,)"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clf=RandomForestClassifier()\n",
        "clf.fit(x,y)\n",
        "print(clf.feature_importances_)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yhpsnDrf4F7y",
        "outputId": "6d5cbe72-85f2-4780-80ef-97d6031c731d"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.08079589 0.02397348 0.45205659 0.44317404]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict_proba(x[[0]]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vOK4sW5B4RV2",
        "outputId": "60b95043-c4cc-408f-b395-31a4c00f4f47"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 0. 0.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Train & Test the model"
      ],
      "metadata": {
        "id": "RrgHN1DP4Wn-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n",
        "x_train.shape,y_train.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RsrhhxkW4YZM",
        "outputId": "6c299d07-fcd7-40d1-c5cb-053f87e01e40"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((120, 4), (120,))"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clf.fit(x_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hrGB6qLH4epP",
        "outputId": "8bde485e-40ee-4217-a34b-5da3a09f295c"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier()"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UjqS_BVp4i5W",
        "outputId": "9d0e2f61-7b1d-4692-f627-ebb1f1abfa60"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 0. 0.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict(x_test))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "othfUqgf4nb9",
        "outputId": "0466177e-8daa-443a-c4aa-f55701923687"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2 2 0 2 0 2 1 1 2 1 1 1 2 2 0 2 1 0 2 2 0 0 1 0 2 2 0 1 1 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gFz8ZBdA4oXj",
        "outputId": "2bca2cb9-0edf-46f0-94cd-4614f8dea5d7"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2 2 0 2 0 2 1 1 2 1 1 1 2 2 0 2 1 0 2 2 0 0 1 0 2 2 0 1 1 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.score(x_test,y_test))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iaClSkjO4yRf",
        "outputId": "dbff0739-5c82-418e-d553-a95472d669d8"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MATRICES"
      ],
      "metadata": {
        "id": "I9o3eRIk8_zU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import classification_report\n",
        "model = LogisticRegression(solver='liblinear')\n",
        "model.fit(x_train, y_train)\n",
        "predicted = model.predict(x_test)\n",
        "report = classification_report(y_test, predicted)\n",
        "print(report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a7AErtab5-fs",
        "outputId": "1c768bef-b5c2-478d-e1dd-a0f58997e70b"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00         8\n",
            "           1       1.00      1.00      1.00        10\n",
            "           2       1.00      1.00      1.00        12\n",
            "\n",
            "    accuracy                           1.00        30\n",
            "   macro avg       1.00      1.00      1.00        30\n",
            "weighted avg       1.00      1.00      1.00        30\n",
            "\n"
          ]
        }
      ]
    }
  ]
}{
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
        "LOAD THE DATA"
      ],
      "metadata": {
        "id": "BKLaU35W7Nod"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import sklearn\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "df =pd.read_csv(r\"/abalone.csv\")\n",
        "df.head()\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "WpK8MZQicU23",
        "outputId": "479c957b-15df-4ff3-d92a-a08e7d1e878c"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "  Sex  Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  \\\n",
              "0   M   0.455     0.365   0.095        0.5140          0.2245          0.1010   \n",
              "1   M   0.350     0.265   0.090        0.2255          0.0995          0.0485   \n",
              "2   F   0.530     0.420   0.135        0.6770          0.2565          0.1415   \n",
              "3   M   0.440     0.365   0.125        0.5160          0.2155          0.1140   \n",
              "4   I   0.330     0.255   0.080        0.2050          0.0895          0.0395   \n",
              "\n",
              "   Shell weight  Rings  \n",
              "0         0.150     15  \n",
              "1         0.070      7  \n",
              "2         0.210      9  \n",
              "3         0.155     10  \n",
              "4         0.055      7  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-e0f586fa-2f51-4001-a40c-eb2341fdd93f\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>M</td>\n",
              "      <td>0.455</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.095</td>\n",
              "      <td>0.5140</td>\n",
              "      <td>0.2245</td>\n",
              "      <td>0.1010</td>\n",
              "      <td>0.150</td>\n",
              "      <td>15</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>M</td>\n",
              "      <td>0.350</td>\n",
              "      <td>0.265</td>\n",
              "      <td>0.090</td>\n",
              "      <td>0.2255</td>\n",
              "      <td>0.0995</td>\n",
              "      <td>0.0485</td>\n",
              "      <td>0.070</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>F</td>\n",
              "      <td>0.530</td>\n",
              "      <td>0.420</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.6770</td>\n",
              "      <td>0.2565</td>\n",
              "      <td>0.1415</td>\n",
              "      <td>0.210</td>\n",
              "      <td>9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>M</td>\n",
              "      <td>0.440</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.125</td>\n",
              "      <td>0.5160</td>\n",
              "      <td>0.2155</td>\n",
              "      <td>0.1140</td>\n",
              "      <td>0.155</td>\n",
              "      <td>10</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>I</td>\n",
              "      <td>0.330</td>\n",
              "      <td>0.255</td>\n",
              "      <td>0.080</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.0895</td>\n",
              "      <td>0.0395</td>\n",
              "      <td>0.055</td>\n",
              "      <td>7</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-e0f586fa-2f51-4001-a40c-eb2341fdd93f')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-e0f586fa-2f51-4001-a40c-eb2341fdd93f button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-e0f586fa-2f51-4001-a40c-eb2341fdd93f');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "UNIVARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "oNagCnv57w3K"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.hist(df['Length'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 352
        },
        "id": "oEkkM1Tt819W",
        "outputId": "75492939-7c5d-4765-a0ee-29d87083aa48"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(array([   7.,   60.,  147.,  304.,  460.,  778., 1051., 1017.,  324.,\n",
              "          29.]),\n",
              " array([0.075, 0.149, 0.223, 0.297, 0.371, 0.445, 0.519, 0.593, 0.667,\n",
              "        0.741, 0.815]),\n",
              " <a list of 10 Patch objects>)"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "BIVARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "gVX2LoNU74e2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "plt.scatter(df.Length, df.Height)\n",
        "plt.xlabel('Length')\n",
        "plt.ylabel('Height')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "x41QPzxw86AO",
        "outputId": "7320829f-6ec7-4459-f3d4-f3e685799a34"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0, 0.5, 'Height')"
            ]
          },
          "metadata": {},
          "execution_count": 14
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.stripplot(x='Diameter',y='Height',data=df,palette='rainbow')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "OzzSG4SN_ARA",
        "outputId": "48cf6890-1258-4053-d416-dc32fc247dc9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f0df2ea5fd0>"
            ]
          },
          "metadata": {},
          "execution_count": 18
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MULTI VARIENT ANALYSIS"
      ],
      "metadata": {
        "id": "5FWz6VzE8Htr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "ax = df[[\"Whole weight\",\"Shucked weight\",\"Viscera weight\"]].plot(figsize=(20,15))\n",
        "ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 588
        },
        "id": "w4E79VZGBa-6",
        "outputId": "1190e065-0e5d-4db1-9e77-5b6811adb753"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1440x1080 with 1 Axes>"
            ],
            "image/png": "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Iv8ZYp20x9vx0Y6PTMW2e4t73roHTWVNaQcwuxJd+MYT38CytmWm+kK7OnchFo5lXSw4Fz7Y/gEW3r8Qpww/BudNvST5eGdXB+558jLMn/Q11Fbkvnj458Wuzl34eMfHrkPrKHd+9Mw1ePHTl3BXjm+wKZiOeAfiMp63mmHZemvzW2itbPXMKMibT14FGseYsnv9fP/pKxEJRXH+fhf26szRzkQnOuOdjtlPe7r2ICRCiIQieHvz272qZEGh+c+m/2BEzYheV5aCCs+7W99FPBFPDilOx5Y9W7AnvicnNZVVHfEOPPnhk5g1cFag2p1PfPgEznz0TJwy5hScN/G8nO4L2b304T9xz1t34/T9LjTVKaTCJ4R4XkrpPRsZ7ZWCBPmOBDBfSvkl/fdlAKZIKc9SlnlNX+ZD/fc1+jKfWdb1FQBfAYCBAwdOXLduXS5fC+XJp5vfxUvrHsPB47/U07tCRERERER7KSkl/u+9/8PcgXO7ZUIMor0Vg3yfX92aaiSlvBXArYCWyded26bMNdQMxsE1vS8Dj4iIiIiI9h5CiJzV0CQi+jzyz5kGPgJM1XoH6I85LqMP162CNgEHERERERERERER5VmQIN+/AQwTQgwSQsQAHAtgtWWZ1QCW6z8fCeAxr3p8RERERERERERElDu+w3WllF1CiLMA/BVAGMDtUsrXhRBXAHhOSrkawK8ArBBCvANgE7RAIBEREREREREREXWDQDX5pJQPAXjI8tilys97AByV210jIiIiIiIiIiKiIIIM1yUiIiIiIiIiIqJejEE+IiIiIiIiIiKiAscgHxERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4BvmIiIiIiIiIiIgKHIN8REREREREREREBY5BPiIiIiIiIiIiogLHIB8REREREREREVGBY5CPiIiIiIiIiIiowDHIR0REREREREREVOAY5CMiIiIiIiIiIipwDPIREREREREREREVOAb5iIiIiIiIiIiIChyDfERERERERERERAWOQT4iIiIiIiIiIqICxyAfERERERERERFRgWOQj4iIiIiIiIiIqMAxyEdERERERERERFTgGOQjIiIiIiIiIiIqcAzyERERERERERERFTgG+YiIiIiIiIiIiAocg3xEREREREREREQFjkE+IiIiIiIiIiKiAscgHxERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4IaXsmQ0L8SmAdT2y8dyrB/BZT+8EEWWNxzLR3oHHMtHegccy0d6Bx3L3a5FSNvT0TlD367Eg395ECPGclHJST+8HEWWHxzLR3oHHMtHegccy0d6BxzJR9+FwXSIiIiIiIiIiogLHIB8REREREREREVGBY5AvN27t6R0gopzgsUy0d+CxTLR34LFMtHfgsUzUTViTj4iIiIiIiIiIqMAxk4+IiIiIiIiIiKjAMchHRERERERERERU4Bjky4IQYr4Q4r9CiHeEEBf39P4QkZkQ4nYhxAYhxGvKY7VCiIeFEG/r/9fojwshxA368fyKEGKC8jfL9eXfFkIs74nXQvR5JoRoFkI8LoR4QwjxuhDiHP1xHs9EBUQIUSyEeFYI8bJ+LH9Xf3yQEOIZ/ZhdJYSI6Y8X6b+/oz/fqqzrm/rj/xVCzOuZV0T0+SaECAshXhRC/En/nccyUQ9jkC9DQogwgJ8BWACgDcBxQoi2nt0rIrL4DYD5lscuBvColHIYgEf13wHtWB6m//sKgJ8DWhABwGUApgCYDOAyI5BARN2mC8AFUso2APsDOFO/5vJ4Jios7QBmSynHARgPYL4QYn8APwDwEynlUACbAZyqL38qgM364z/Rl4N+/B8LYDS06/zNetuciLrXOQDeVH7nsUzUwxjky9xkAO9IKd+VUnYAWAng0B7eJyJSSCmfALDJ8vChAO7Qf74DwGHK47+Vmn8BqBZC9AMwD8DDUspNUsrNAB6GPXBIRHkkpfyflPIF/eft0G4omsDjmaig6MfkDv3XqP5PApgN4F79ceuxbBzj9wKYI4QQ+uMrpZTtUsr3ALwDrW1ORN1ECDEAwCIAv9R/F+CxTNTjGOTLXBOAD5TfP9QfI6LerVFK+T/9508ANOo/ux3TPNaJehF9iM++AJ4Bj2eigqMP73sJwAZogfY1ALZIKbv0RdTjMnnM6s9vBVAHHstEvcH1AL4BIKH/Xgcey0Q9jkE+IvrcklJKaBkERFQAhBDlAO4DcK6Ucpv6HI9nosIgpYxLKccDGAAtY2dkD+8SEaVJCLEYwAYp5fM9vS9EZMYgX+Y+AtCs/D5Af4yIerf1+rA96P9v0B93O6Z5rBP1AkKIKLQA3++klPfrD/N4JipQUsotAB4HMBXakPqI/pR6XCaPWf35KgAbwWOZqKdNA7BUCLEWWtmq2QB+Ch7LRD2OQb7M/RvAMH0GoRi0gqGre3ifiMjfagDGjJrLAfxRefwkfVbO/QFs1YcB/hXAIUKIGr1A/yH6Y0TUTfS6Pb8C8KaU8jrlKR7PRAVECNEghKjWfy4BcDC0GpuPAzhSX8x6LBvH+JEAHtOzdlcDOFafsXMQtEl2nu2eV0FEUspvSikHSClbod0HPyalPAE8lol6XMR/EXIipewSQpwF7eYgDOB2KeXrPbxbRKQQQvwewEwA9UKID6HNqnkNgLuFEKcCWAfgaH3xhwAshFbwdxeALwKAlHKTEOJKaIF9ALhCSmmdzIOI8msagGUAXtVreQHAt8DjmajQ9ANwhz57ZgjA3VLKPwkh3gCwUghxFYAXoQX1of+/QgjxDrSJtI4FACnl60KIuwG8AW327TOllPFufi1EZHcReCwT9SihBdCJiIiIiIiIiIioUHG4LhERERERERERUYFjkI+IiIiIiIiIiKjAMchHRERERERERERU4BjkIyIiIiIiIiIiKnAM8hERERERERERERU4BvmIiIiIiIiIiIgKHIN8REREREREREREBe7/A0aQbHgzm4UDAAAAAElFTkSuQmCC\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Descriptive statistics"
      ],
      "metadata": {
        "id": "nwBbVRuL8KS2"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#statistic finding mean\n",
        "df.mean()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "N2P4d6HyD5z1",
        "outputId": "e71b0635-0fba-4155-80cb-5b2694fcb189"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Length            0.523992\n",
              "Diameter          0.407881\n",
              "Height            0.139516\n",
              "Whole weight      0.828742\n",
              "Shucked weight    0.359367\n",
              "Viscera weight    0.180594\n",
              "Shell weight      0.238831\n",
              "Rings             9.933684\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 22
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.median()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OQgEsXWeELDj",
        "outputId": "0af1f080-1bf7-4a8b-ae47-9dde790d2f68"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Length            0.5450\n",
              "Diameter          0.4250\n",
              "Height            0.1400\n",
              "Whole weight      0.7995\n",
              "Shucked weight    0.3360\n",
              "Viscera weight    0.1710\n",
              "Shell weight      0.2340\n",
              "Rings             9.0000\n",
              "dtype: float64"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.mode()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 112
        },
        "id": "_QyAaHvUEvV7",
        "outputId": "e80c2740-5db3-413c-8eb8-8cb6cc187d4e"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Sex  Length  Diameter  Height  Whole weight  Shucked weight  \\\n",
              "0    M   0.550      0.45    0.15        0.2225           0.175   \n",
              "1  NaN   0.625       NaN     NaN           NaN             NaN   \n",
              "\n",
              "   Viscera weight  Shell weight  Rings  \n",
              "0          0.1715         0.275    9.0  \n",
              "1             NaN           NaN    NaN  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-24bb07aa-af67-4891-9108-6fbd10c619bb\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>M</td>\n",
              "      <td>0.550</td>\n",
              "      <td>0.45</td>\n",
              "      <td>0.15</td>\n",
              "      <td>0.2225</td>\n",
              "      <td>0.175</td>\n",
              "      <td>0.1715</td>\n",
              "      <td>0.275</td>\n",
              "      <td>9.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>NaN</td>\n",
              "      <td>0.625</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "      <td>NaN</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-24bb07aa-af67-4891-9108-6fbd10c619bb')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-24bb07aa-af67-4891-9108-6fbd10c619bb button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-24bb07aa-af67-4891-9108-6fbd10c619bb');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 24
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Missing values"
      ],
      "metadata": {
        "id": "CIVh0M2-8QTZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "bool_series = pd.isnull(df[\"Sex\"])\n",
        "df[bool_series]"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 49
        },
        "id": "NOJFZ_4IFlcG",
        "outputId": "4e4145aa-dc39-41b1-81f3-da8d4b3e952c"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Empty DataFrame\n",
              "Columns: [Sex, Length, Diameter, Height, Whole weight, Shucked weight, Viscera weight, Shell weight, Rings]\n",
              "Index: []"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-68b9217c-4fae-4e16-9024-d0da07312914\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-68b9217c-4fae-4e16-9024-d0da07312914')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-68b9217c-4fae-4e16-9024-d0da07312914 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-68b9217c-4fae-4e16-9024-d0da07312914');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(df.info())"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yKtM1AfLOS-Y",
        "outputId": "71916c96-a34f-45ac-96a2-af1f84da8b0f"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 4177 entries, 0 to 4176\n",
            "Data columns (total 9 columns):\n",
            " #   Column          Non-Null Count  Dtype  \n",
            "---  ------          --------------  -----  \n",
            " 0   Sex             4177 non-null   object \n",
            " 1   Length          4177 non-null   float64\n",
            " 2   Diameter        4177 non-null   float64\n",
            " 3   Height          4177 non-null   float64\n",
            " 4   Whole weight    4177 non-null   float64\n",
            " 5   Shucked weight  4177 non-null   float64\n",
            " 6   Viscera weight  4177 non-null   float64\n",
            " 7   Shell weight    4177 non-null   float64\n",
            " 8   Rings           4177 non-null   int64  \n",
            "dtypes: float64(7), int64(1), object(1)\n",
            "memory usage: 293.8+ KB\n",
            "None\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 424
        },
        "id": "qbmZdgNkXKJ6",
        "outputId": "76ddaa09-d0bd-4cdf-d862-4eaa31f8d0b9"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "        Sex  Length  Diameter  Height  Whole weight  Shucked weight  \\\n",
              "0     False   False     False   False         False           False   \n",
              "1     False   False     False   False         False           False   \n",
              "2     False   False     False   False         False           False   \n",
              "3     False   False     False   False         False           False   \n",
              "4     False   False     False   False         False           False   \n",
              "...     ...     ...       ...     ...           ...             ...   \n",
              "4172  False   False     False   False         False           False   \n",
              "4173  False   False     False   False         False           False   \n",
              "4174  False   False     False   False         False           False   \n",
              "4175  False   False     False   False         False           False   \n",
              "4176  False   False     False   False         False           False   \n",
              "\n",
              "      Viscera weight  Shell weight  Rings  \n",
              "0              False         False  False  \n",
              "1              False         False  False  \n",
              "2              False         False  False  \n",
              "3              False         False  False  \n",
              "4              False         False  False  \n",
              "...              ...           ...    ...  \n",
              "4172           False         False  False  \n",
              "4173           False         False  False  \n",
              "4174           False         False  False  \n",
              "4175           False         False  False  \n",
              "4176           False         False  False  \n",
              "\n",
              "[4177 rows x 9 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Sex</th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4172</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4173</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4174</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4175</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4176</th>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "      <td>False</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>4177 rows × 9 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-3f006ba6-10c8-4bdd-bd13-2293cfa1b410');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "df.isnull().sum()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9WAbwRTpjcNX",
        "outputId": "bc336670-a807-4f1b-c80e-48f6aefa76ce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Sex               0\n",
              "Length            0\n",
              "Diameter          0\n",
              "Height            0\n",
              "Whole weight      0\n",
              "Shucked weight    0\n",
              "Viscera weight    0\n",
              "Shell weight      0\n",
              "Rings             0\n",
              "dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "outliers"
      ],
      "metadata": {
        "id": "IpBU0SKx8ctz"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "sns.displot(df['Sex'])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 386
        },
        "id": "uF7viPnzjiY2",
        "outputId": "f44987b3-a150-4d40-d6ea-1fb344f0e3a1"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<seaborn.axisgrid.FacetGrid at 0x7f06a2a95d90>"
            ]
          },
          "metadata": {},
          "execution_count": 10
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 360x360 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "sns.boxplot(x='Sex',y='Rings',data=df)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 298
        },
        "id": "qivTUF_IjuTO",
        "outputId": "b84318d5-b304-44f1-c044-66123748bab4"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.axes._subplots.AxesSubplot at 0x7f069f444b90>"
            ]
          },
          "metadata": {},
          "execution_count": 11
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ],
            "image/png": "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\n"
          },
          "metadata": {
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "data_tips=pd.get_dummies(df)\n",
        "data_tips"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 468
        },
        "id": "MkZyvnwkj7OE",
        "outputId": "9da22cf1-0261-4354-a677-9f73cc24cc1c"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "      Length  Diameter  Height  Whole weight  Shucked weight  Viscera weight  \\\n",
              "0      0.455     0.365   0.095        0.5140          0.2245          0.1010   \n",
              "1      0.350     0.265   0.090        0.2255          0.0995          0.0485   \n",
              "2      0.530     0.420   0.135        0.6770          0.2565          0.1415   \n",
              "3      0.440     0.365   0.125        0.5160          0.2155          0.1140   \n",
              "4      0.330     0.255   0.080        0.2050          0.0895          0.0395   \n",
              "...      ...       ...     ...           ...             ...             ...   \n",
              "4172   0.565     0.450   0.165        0.8870          0.3700          0.2390   \n",
              "4173   0.590     0.440   0.135        0.9660          0.4390          0.2145   \n",
              "4174   0.600     0.475   0.205        1.1760          0.5255          0.2875   \n",
              "4175   0.625     0.485   0.150        1.0945          0.5310          0.2610   \n",
              "4176   0.710     0.555   0.195        1.9485          0.9455          0.3765   \n",
              "\n",
              "      Shell weight  Rings  Sex_F  Sex_I  Sex_M  \n",
              "0           0.1500     15      0      0      1  \n",
              "1           0.0700      7      0      0      1  \n",
              "2           0.2100      9      1      0      0  \n",
              "3           0.1550     10      0      0      1  \n",
              "4           0.0550      7      0      1      0  \n",
              "...            ...    ...    ...    ...    ...  \n",
              "4172        0.2490     11      1      0      0  \n",
              "4173        0.2605     10      0      0      1  \n",
              "4174        0.3080      9      0      0      1  \n",
              "4175        0.2960     10      1      0      0  \n",
              "4176        0.4950     12      0      0      1  \n",
              "\n",
              "[4177 rows x 11 columns]"
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-2a940f93-302f-423e-aca4-70bee8235a3a\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Length</th>\n",
              "      <th>Diameter</th>\n",
              "      <th>Height</th>\n",
              "      <th>Whole weight</th>\n",
              "      <th>Shucked weight</th>\n",
              "      <th>Viscera weight</th>\n",
              "      <th>Shell weight</th>\n",
              "      <th>Rings</th>\n",
              "      <th>Sex_F</th>\n",
              "      <th>Sex_I</th>\n",
              "      <th>Sex_M</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.455</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.095</td>\n",
              "      <td>0.5140</td>\n",
              "      <td>0.2245</td>\n",
              "      <td>0.1010</td>\n",
              "      <td>0.1500</td>\n",
              "      <td>15</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.350</td>\n",
              "      <td>0.265</td>\n",
              "      <td>0.090</td>\n",
              "      <td>0.2255</td>\n",
              "      <td>0.0995</td>\n",
              "      <td>0.0485</td>\n",
              "      <td>0.0700</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.530</td>\n",
              "      <td>0.420</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.6770</td>\n",
              "      <td>0.2565</td>\n",
              "      <td>0.1415</td>\n",
              "      <td>0.2100</td>\n",
              "      <td>9</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.440</td>\n",
              "      <td>0.365</td>\n",
              "      <td>0.125</td>\n",
              "      <td>0.5160</td>\n",
              "      <td>0.2155</td>\n",
              "      <td>0.1140</td>\n",
              "      <td>0.1550</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.330</td>\n",
              "      <td>0.255</td>\n",
              "      <td>0.080</td>\n",
              "      <td>0.2050</td>\n",
              "      <td>0.0895</td>\n",
              "      <td>0.0395</td>\n",
              "      <td>0.0550</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4172</th>\n",
              "      <td>0.565</td>\n",
              "      <td>0.450</td>\n",
              "      <td>0.165</td>\n",
              "      <td>0.8870</td>\n",
              "      <td>0.3700</td>\n",
              "      <td>0.2390</td>\n",
              "      <td>0.2490</td>\n",
              "      <td>11</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4173</th>\n",
              "      <td>0.590</td>\n",
              "      <td>0.440</td>\n",
              "      <td>0.135</td>\n",
              "      <td>0.9660</td>\n",
              "      <td>0.4390</td>\n",
              "      <td>0.2145</td>\n",
              "      <td>0.2605</td>\n",
              "      <td>10</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4174</th>\n",
              "      <td>0.600</td>\n",
              "      <td>0.475</td>\n",
              "      <td>0.205</td>\n",
              "      <td>1.1760</td>\n",
              "      <td>0.5255</td>\n",
              "      <td>0.2875</td>\n",
              "      <td>0.3080</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4175</th>\n",
              "      <td>0.625</td>\n",
              "      <td>0.485</td>\n",
              "      <td>0.150</td>\n",
              "      <td>1.0945</td>\n",
              "      <td>0.5310</td>\n",
              "      <td>0.2610</td>\n",
              "      <td>0.2960</td>\n",
              "      <td>10</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4176</th>\n",
              "      <td>0.710</td>\n",
              "      <td>0.555</td>\n",
              "      <td>0.195</td>\n",
              "      <td>1.9485</td>\n",
              "      <td>0.9455</td>\n",
              "      <td>0.3765</td>\n",
              "      <td>0.4950</td>\n",
              "      <td>12</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>4177 rows × 11 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-2a940f93-302f-423e-aca4-70bee8235a3a')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "        \n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "      \n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-2a940f93-302f-423e-aca4-70bee8235a3a button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-2a940f93-302f-423e-aca4-70bee8235a3a');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n",
              "  "
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Split the data into dependent and independent variables"
      ],
      "metadata": {
        "id": "BR75HIpZ8ojg"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x=df.iloc[:,1:8]\n",
        "y=df.iloc[:,8]\n",
        "x\n",
        "y"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bfI-u_KikEVs",
        "outputId": "e05e05f5-3c68-4099-c07d-6d4466a8d005"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       15\n",
              "1        7\n",
              "2        9\n",
              "3       10\n",
              "4        7\n",
              "        ..\n",
              "4172    11\n",
              "4173    10\n",
              "4174     9\n",
              "4175    10\n",
              "4176    12\n",
              "Name: Rings, Length: 4177, dtype: int64"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "inde=df.iloc[1:,1:7].values"
      ],
      "metadata": {
        "id": "PP7bTNI8kbL2"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "inde"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cuIzZQ0nkegw",
        "outputId": "0d126b2f-53c8-47ee-813a-b646b553e202"
      },
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.35  , 0.265 , 0.09  , 0.2255, 0.0995, 0.0485],\n",
              "       [0.53  , 0.42  , 0.135 , 0.677 , 0.2565, 0.1415],\n",
              "       [0.44  , 0.365 , 0.125 , 0.516 , 0.2155, 0.114 ],\n",
              "       ...,\n",
              "       [0.6   , 0.475 , 0.205 , 1.176 , 0.5255, 0.2875],\n",
              "       [0.625 , 0.485 , 0.15  , 1.0945, 0.531 , 0.261 ],\n",
              "       [0.71  , 0.555 , 0.195 , 1.9485, 0.9455, 0.3765]])"
            ]
          },
          "metadata": {},
          "execution_count": 19
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "depe=df.iloc[1:,9:].values\n",
        "depe"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "itNM19B0kkGn",
        "outputId": "83c341a3-13c3-4fe9-a881-2d52bdcf274e"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([], shape=(4176, 0), dtype=float64)"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Split the data into training and testing"
      ],
      "metadata": {
        "id": "LuH2z89G8rYb"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(inde,depe,test_size=0.2,random_state=5)\n",
        "x_train"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "8M3Ls9AckuJ9",
        "outputId": "4d7d1fe4-b154-43f1-cd7a-42d2f96c5a78"
      },
      "execution_count": 20,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.565 , 0.435 , 0.15  , 0.99  , 0.5795, 0.1825],\n",
              "       [0.48  , 0.37  , 0.125 , 0.5435, 0.244 , 0.101 ],\n",
              "       [0.44  , 0.35  , 0.12  , 0.375 , 0.1425, 0.0965],\n",
              "       ...,\n",
              "       [0.555 , 0.43  , 0.125 , 0.7005, 0.3395, 0.1355],\n",
              "       [0.51  , 0.395 , 0.145 , 0.6185, 0.216 , 0.1385],\n",
              "       [0.595 , 0.47  , 0.155 , 1.2015, 0.492 , 0.3865]])"
            ]
          },
          "metadata": {},
          "execution_count": 20
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x_test"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LBBQkTy0k62J",
        "outputId": "4630928d-32f0-4d5b-c19b-8e2739270561"
      },
      "execution_count": 23,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.455 , 0.365 , 0.11  , 0.385 , 0.166 , 0.046 ],\n",
              "       [0.47  , 0.37  , 0.18  , 0.51  , 0.1915, 0.1285],\n",
              "       [0.72  , 0.575 , 0.17  , 1.9335, 0.913 , 0.389 ],\n",
              "       ...,\n",
              "       [0.275 , 0.215 , 0.075 , 0.1155, 0.0485, 0.029 ],\n",
              "       [0.39  , 0.3   , 0.09  , 0.252 , 0.1065, 0.053 ],\n",
              "       [0.585 , 0.46  , 0.165 , 1.1135, 0.5825, 0.2345]])"
            ]
          },
          "metadata": {},
          "execution_count": 23
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Build the Model"
      ],
      "metadata": {
        "id": "FIyZrecJ80b9"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn import datasets\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.datasets import make_classification\n",
        "iris=datasets.load_iris()"
      ],
      "metadata": {
        "id": "s60typmplDxp"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(iris.feature_names)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "br2ZlXEFlMW4",
        "outputId": "28849331-d4fe-4251-fcad-3101343f42ff"
      },
      "execution_count": 25,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(iris.target_names)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4SBaT0tslVFv",
        "outputId": "39ef5f4b-59a9-4d07-e062-fda0973823e9"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['setosa' 'versicolor' 'virginica']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "iris.data\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "lOvwcFHflhNX",
        "outputId": "857cf0b9-c693-4a74-8568-c90ef9cdcd6e"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[5.1, 3.5, 1.4, 0.2],\n",
              "       [4.9, 3. , 1.4, 0.2],\n",
              "       [4.7, 3.2, 1.3, 0.2],\n",
              "       [4.6, 3.1, 1.5, 0.2],\n",
              "       [5. , 3.6, 1.4, 0.2],\n",
              "       [5.4, 3.9, 1.7, 0.4],\n",
              "       [4.6, 3.4, 1.4, 0.3],\n",
              "       [5. , 3.4, 1.5, 0.2],\n",
              "       [4.4, 2.9, 1.4, 0.2],\n",
              "       [4.9, 3.1, 1.5, 0.1],\n",
              "       [5.4, 3.7, 1.5, 0.2],\n",
              "       [4.8, 3.4, 1.6, 0.2],\n",
              "       [4.8, 3. , 1.4, 0.1],\n",
              "       [4.3, 3. , 1.1, 0.1],\n",
              "       [5.8, 4. , 1.2, 0.2],\n",
              "       [5.7, 4.4, 1.5, 0.4],\n",
              "       [5.4, 3.9, 1.3, 0.4],\n",
              "       [5.1, 3.5, 1.4, 0.3],\n",
              "       [5.7, 3.8, 1.7, 0.3],\n",
              "       [5.1, 3.8, 1.5, 0.3],\n",
              "       [5.4, 3.4, 1.7, 0.2],\n",
              "       [5.1, 3.7, 1.5, 0.4],\n",
              "       [4.6, 3.6, 1. , 0.2],\n",
              "       [5.1, 3.3, 1.7, 0.5],\n",
              "       [4.8, 3.4, 1.9, 0.2],\n",
              "       [5. , 3. , 1.6, 0.2],\n",
              "       [5. , 3.4, 1.6, 0.4],\n",
              "       [5.2, 3.5, 1.5, 0.2],\n",
              "       [5.2, 3.4, 1.4, 0.2],\n",
              "       [4.7, 3.2, 1.6, 0.2],\n",
              "       [4.8, 3.1, 1.6, 0.2],\n",
              "       [5.4, 3.4, 1.5, 0.4],\n",
              "       [5.2, 4.1, 1.5, 0.1],\n",
              "       [5.5, 4.2, 1.4, 0.2],\n",
              "       [4.9, 3.1, 1.5, 0.2],\n",
              "       [5. , 3.2, 1.2, 0.2],\n",
              "       [5.5, 3.5, 1.3, 0.2],\n",
              "       [4.9, 3.6, 1.4, 0.1],\n",
              "       [4.4, 3. , 1.3, 0.2],\n",
              "       [5.1, 3.4, 1.5, 0.2],\n",
              "       [5. , 3.5, 1.3, 0.3],\n",
              "       [4.5, 2.3, 1.3, 0.3],\n",
              "       [4.4, 3.2, 1.3, 0.2],\n",
              "       [5. , 3.5, 1.6, 0.6],\n",
              "       [5.1, 3.8, 1.9, 0.4],\n",
              "       [4.8, 3. , 1.4, 0.3],\n",
              "       [5.1, 3.8, 1.6, 0.2],\n",
              "       [4.6, 3.2, 1.4, 0.2],\n",
              "       [5.3, 3.7, 1.5, 0.2],\n",
              "       [5. , 3.3, 1.4, 0.2],\n",
              "       [7. , 3.2, 4.7, 1.4],\n",
              "       [6.4, 3.2, 4.5, 1.5],\n",
              "       [6.9, 3.1, 4.9, 1.5],\n",
              "       [5.5, 2.3, 4. , 1.3],\n",
              "       [6.5, 2.8, 4.6, 1.5],\n",
              "       [5.7, 2.8, 4.5, 1.3],\n",
              "       [6.3, 3.3, 4.7, 1.6],\n",
              "       [4.9, 2.4, 3.3, 1. ],\n",
              "       [6.6, 2.9, 4.6, 1.3],\n",
              "       [5.2, 2.7, 3.9, 1.4],\n",
              "       [5. , 2. , 3.5, 1. ],\n",
              "       [5.9, 3. , 4.2, 1.5],\n",
              "       [6. , 2.2, 4. , 1. ],\n",
              "       [6.1, 2.9, 4.7, 1.4],\n",
              "       [5.6, 2.9, 3.6, 1.3],\n",
              "       [6.7, 3.1, 4.4, 1.4],\n",
              "       [5.6, 3. , 4.5, 1.5],\n",
              "       [5.8, 2.7, 4.1, 1. ],\n",
              "       [6.2, 2.2, 4.5, 1.5],\n",
              "       [5.6, 2.5, 3.9, 1.1],\n",
              "       [5.9, 3.2, 4.8, 1.8],\n",
              "       [6.1, 2.8, 4. , 1.3],\n",
              "       [6.3, 2.5, 4.9, 1.5],\n",
              "       [6.1, 2.8, 4.7, 1.2],\n",
              "       [6.4, 2.9, 4.3, 1.3],\n",
              "       [6.6, 3. , 4.4, 1.4],\n",
              "       [6.8, 2.8, 4.8, 1.4],\n",
              "       [6.7, 3. , 5. , 1.7],\n",
              "       [6. , 2.9, 4.5, 1.5],\n",
              "       [5.7, 2.6, 3.5, 1. ],\n",
              "       [5.5, 2.4, 3.8, 1.1],\n",
              "       [5.5, 2.4, 3.7, 1. ],\n",
              "       [5.8, 2.7, 3.9, 1.2],\n",
              "       [6. , 2.7, 5.1, 1.6],\n",
              "       [5.4, 3. , 4.5, 1.5],\n",
              "       [6. , 3.4, 4.5, 1.6],\n",
              "       [6.7, 3.1, 4.7, 1.5],\n",
              "       [6.3, 2.3, 4.4, 1.3],\n",
              "       [5.6, 3. , 4.1, 1.3],\n",
              "       [5.5, 2.5, 4. , 1.3],\n",
              "       [5.5, 2.6, 4.4, 1.2],\n",
              "       [6.1, 3. , 4.6, 1.4],\n",
              "       [5.8, 2.6, 4. , 1.2],\n",
              "       [5. , 2.3, 3.3, 1. ],\n",
              "       [5.6, 2.7, 4.2, 1.3],\n",
              "       [5.7, 3. , 4.2, 1.2],\n",
              "       [5.7, 2.9, 4.2, 1.3],\n",
              "       [6.2, 2.9, 4.3, 1.3],\n",
              "       [5.1, 2.5, 3. , 1.1],\n",
              "       [5.7, 2.8, 4.1, 1.3],\n",
              "       [6.3, 3.3, 6. , 2.5],\n",
              "       [5.8, 2.7, 5.1, 1.9],\n",
              "       [7.1, 3. , 5.9, 2.1],\n",
              "       [6.3, 2.9, 5.6, 1.8],\n",
              "       [6.5, 3. , 5.8, 2.2],\n",
              "       [7.6, 3. , 6.6, 2.1],\n",
              "       [4.9, 2.5, 4.5, 1.7],\n",
              "       [7.3, 2.9, 6.3, 1.8],\n",
              "       [6.7, 2.5, 5.8, 1.8],\n",
              "       [7.2, 3.6, 6.1, 2.5],\n",
              "       [6.5, 3.2, 5.1, 2. ],\n",
              "       [6.4, 2.7, 5.3, 1.9],\n",
              "       [6.8, 3. , 5.5, 2.1],\n",
              "       [5.7, 2.5, 5. , 2. ],\n",
              "       [5.8, 2.8, 5.1, 2.4],\n",
              "       [6.4, 3.2, 5.3, 2.3],\n",
              "       [6.5, 3. , 5.5, 1.8],\n",
              "       [7.7, 3.8, 6.7, 2.2],\n",
              "       [7.7, 2.6, 6.9, 2.3],\n",
              "       [6. , 2.2, 5. , 1.5],\n",
              "       [6.9, 3.2, 5.7, 2.3],\n",
              "       [5.6, 2.8, 4.9, 2. ],\n",
              "       [7.7, 2.8, 6.7, 2. ],\n",
              "       [6.3, 2.7, 4.9, 1.8],\n",
              "       [6.7, 3.3, 5.7, 2.1],\n",
              "       [7.2, 3.2, 6. , 1.8],\n",
              "       [6.2, 2.8, 4.8, 1.8],\n",
              "       [6.1, 3. , 4.9, 1.8],\n",
              "       [6.4, 2.8, 5.6, 2.1],\n",
              "       [7.2, 3. , 5.8, 1.6],\n",
              "       [7.4, 2.8, 6.1, 1.9],\n",
              "       [7.9, 3.8, 6.4, 2. ],\n",
              "       [6.4, 2.8, 5.6, 2.2],\n",
              "       [6.3, 2.8, 5.1, 1.5],\n",
              "       [6.1, 2.6, 5.6, 1.4],\n",
              "       [7.7, 3. , 6.1, 2.3],\n",
              "       [6.3, 3.4, 5.6, 2.4],\n",
              "       [6.4, 3.1, 5.5, 1.8],\n",
              "       [6. , 3. , 4.8, 1.8],\n",
              "       [6.9, 3.1, 5.4, 2.1],\n",
              "       [6.7, 3.1, 5.6, 2.4],\n",
              "       [6.9, 3.1, 5.1, 2.3],\n",
              "       [5.8, 2.7, 5.1, 1.9],\n",
              "       [6.8, 3.2, 5.9, 2.3],\n",
              "       [6.7, 3.3, 5.7, 2.5],\n",
              "       [6.7, 3. , 5.2, 2.3],\n",
              "       [6.3, 2.5, 5. , 1.9],\n",
              "       [6.5, 3. , 5.2, 2. ],\n",
              "       [6.2, 3.4, 5.4, 2.3],\n",
              "       [5.9, 3. , 5.1, 1.8]])"
            ]
          },
          "metadata": {},
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "iris.target"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4CKp5y-Nl9L9",
        "outputId": "9803e71a-fc4d-4618-ad4f-799005c68a6c"
      },
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n",
              "       0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,\n",
              "       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
              "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,\n",
              "       2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])"
            ]
          },
          "metadata": {},
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "x=iris.data\n",
        "y=iris.target\n"
      ],
      "metadata": {
        "id": "2FQyZLsFl-7l"
      },
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "x.shape\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ITeNy_zQ3zXv",
        "outputId": "0da97dd7-a3ed-4e15-a34e-01aa28096d51"
      },
      "execution_count": 31,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(150, 4)"
            ]
          },
          "metadata": {},
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "y.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C1b7eAla3_YL",
        "outputId": "d6b80fd9-7e14-43da-a076-95d84f99e44a"
      },
      "execution_count": 32,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(150,)"
            ]
          },
          "metadata": {},
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clf=RandomForestClassifier()\n",
        "clf.fit(x,y)\n",
        "print(clf.feature_importances_)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yhpsnDrf4F7y",
        "outputId": "6d5cbe72-85f2-4780-80ef-97d6031c731d"
      },
      "execution_count": 34,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[0.08079589 0.02397348 0.45205659 0.44317404]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict_proba(x[[0]]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vOK4sW5B4RV2",
        "outputId": "60b95043-c4cc-408f-b395-31a4c00f4f47"
      },
      "execution_count": 35,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 0. 0.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Train & Test the model"
      ],
      "metadata": {
        "id": "RrgHN1DP4Wn-"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)\n",
        "x_train.shape,y_train.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RsrhhxkW4YZM",
        "outputId": "6c299d07-fcd7-40d1-c5cb-053f87e01e40"
      },
      "execution_count": 36,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "((120, 4), (120,))"
            ]
          },
          "metadata": {},
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "clf.fit(x_train,y_train)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hrGB6qLH4epP",
        "outputId": "8bde485e-40ee-4217-a34b-5da3a09f295c"
      },
      "execution_count": 37,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier()"
            ]
          },
          "metadata": {},
          "execution_count": 37
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "UjqS_BVp4i5W",
        "outputId": "9d0e2f61-7b1d-4692-f627-ebb1f1abfa60"
      },
      "execution_count": 38,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[1. 0. 0.]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.predict(x_test))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "othfUqgf4nb9",
        "outputId": "0466177e-8daa-443a-c4aa-f55701923687"
      },
      "execution_count": 39,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2 2 0 2 0 2 1 1 2 1 1 1 2 2 0 2 1 0 2 2 0 0 1 0 2 2 0 1 1 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(y_test)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "gFz8ZBdA4oXj",
        "outputId": "2bca2cb9-0edf-46f0-94cd-4614f8dea5d7"
      },
      "execution_count": 40,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[2 2 0 2 0 2 1 1 2 1 1 1 2 2 0 2 1 0 2 2 0 0 1 0 2 2 0 1 1 1]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(clf.score(x_test,y_test))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "iaClSkjO4yRf",
        "outputId": "dbff0739-5c82-418e-d553-a95472d669d8"
      },
      "execution_count": 41,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "MATRICES"
      ],
      "metadata": {
        "id": "I9o3eRIk8_zU"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import classification_report\n",
        "model = LogisticRegression(solver='liblinear')\n",
        "model.fit(x_train, y_train)\n",
        "predicted = model.predict(x_test)\n",
        "report = classification_report(y_test, predicted)\n",
        "print(report)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a7AErtab5-fs",
        "outputId": "1c768bef-b5c2-478d-e1dd-a0f58997e70b"
      },
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       1.00      1.00      1.00         8\n",
            "           1       1.00      1.00      1.00        10\n",
            "           2       1.00      1.00      1.00        12\n",
            "\n",
            "    accuracy                           1.00        30\n",
            "   macro avg       1.00      1.00      1.00        30\n",
            "weighted avg       1.00      1.00      1.00        30\n",
            "\n"
          ]
        }
      ]
    }
  ]
}