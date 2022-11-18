{\rtf1\ansi\ansicpg1252\deff0\deflang1033{\fonttbl{\f0\fnil\fcharset0 Calibri;}}
{\colortbl ;\red0\green0\blue255;}
{\*\generator Msftedit 5.41.21.2510;}\viewkind4\uc1\pard\sa200\sl276\slmult1\lang9\f0\fs22\{\par
  "nbformat": 4,\par
  "nbformat_minor": 0,\par
  "metadata": \{\par
    "colab": \{\par
      "provenance": [],\par
      "collapsed_sections": []\par
    \},\par
    "kernelspec": \{\par
      "name": "python3",\par
      "display_name": "Python 3"\par
    \},\par
    "language_info": \{\par
      "name": "python"\par
    \}\par
  \},\par
  "cells": [\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "# Basic Python"\par
      ],\par
      "metadata": \{\par
        "id": "McSxJAwcOdZ1"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 1. Split this string"\par
      ],\par
      "metadata": \{\par
        "id": "CU48hgo4Owz5"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "s = \\"Hi there Sam!\\""\par
      ],\par
      "metadata": \{\par
        "id": "s07c7JK7Oqt-"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "T = s.split()\\n",\par
        "print(T)"\par
      ],\par
      "metadata": \{\par
        "id": "6mGVa3SQYLkb",\par
        "outputId": "25278b4d-6302-4983-9673-fba4fe9a574e",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "['Hi', 'there', 'Sam!']\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 2. Use .format() to print the following string. \\n",\par
        "\\n",\par
        "### Output should be: The diameter of Earth is 12742 kilometers."\par
      ],\par
      "metadata": \{\par
        "id": "GH1QBn8HP375"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "planet = \\"Earth\\"\\n",\par
        "diameter = 12742"\par
      ],\par
      "metadata": \{\par
        "id": "_ZHoml3kPqic"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "planet = \\"Earth\\"\\n",\par
        "diameter = 12742\\n",\par
        "print('The diameter of \{\} is \{\} kilometers.'.format(planet,diameter));"\par
      ],\par
      "metadata": \{\par
        "id": "HyRyJv6CYPb4",\par
        "outputId": "ded3e1fd-ce42-4aed-ab7b-1435cf4b1d29",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "The diameter of Earth is 12742 kilometers.\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 3. In this nest dictionary grab the word \\"hello\\""\par
      ],\par
      "metadata": \{\par
        "id": "KE74ZEwkRExZ"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "d = \{'k1':[1,2,3,\{'tricky':['oh','man','inception',\{'target':[1,2,3,'hello']\}]\}]\}"\par
      ],\par
      "metadata": \{\par
        "id": "fcVwbCc1QrQI"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "d = \{'k1':[1,2,3,\{'tricky':['oh','man','inception',\{'target':[1,2,3,'hello']\}]\}]\}\\n",\par
        "d['k1'][3]['tricky'][3]['target'][3]"\par
      ],\par
      "metadata": \{\par
        "id": "MvbkMZpXYRaw",\par
        "outputId": "57926d9e-4000-4e31-ab94-2cf98cfe9667",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 ",\par
          "height": 36\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "execute_result",\par
          "data": \{\par
            "text/plain": [\par
              "'hello'"\par
            ],\par
            "application/vnd.google.colaboratory.intrinsic+json": \{\par
              "type": "string"\par
            \}\par
          \},\par
          "metadata": \{\},\par
          "execution_count": 10\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "# Numpy"\par
      ],\par
      "metadata": \{\par
        "id": "bw0vVp-9ddjv"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np"\par
      ],\par
      "metadata": \{\par
        "id": "LLiE_TYrhA1O"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 4.1 Create an array of 10 zeros? \\n",\par
        "## 4.2 Create an array of 10 fives?"\par
      ],\par
      "metadata": \{\par
        "id": "wOg8hinbgx30"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np\\n",\par
        "array=np.zeros(10)\\n",\par
        "print(\\"An array of 10 zeros:\\")\\n",\par
        "print(array)"\par
      ],\par
      "metadata": \{\par
        "id": "NHrirmgCYXvU",\par
        "outputId": "b79c0955-b041-4561-ce50-87b2177bc71c",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "An array of 10 zeros:\\n",\par
            "[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np\\n",\par
        "array=np.ones(10)*5\\n",\par
        "print(\\"An array of 10 fives:\\")\\n",\par
        "print(array)"\par
      ],\par
      "metadata": \{\par
        "id": "e4005lsTYXxx",\par
        "outputId": "9f790aec-0be2-4b2a-d073-932572ce83b1",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "An array of 10 fives:\\n",\par
            "[5. 5. 5. 5. 5. 5. 5. 5. 5. 5.]\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 5. Create an array of all the even integers from 20 to 35"\par
      ],\par
      "metadata": \{\par
        "id": "gZHHDUBvrMX4"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np\\n",\par
        "even =np.arange(20,35,2)\\n",\par
        "print(\\"Array of all the even integers from 20 to 35\\")\\n",\par
        "print(even) "\par
      ],\par
      "metadata": \{\par
        "id": "oAI2tbU2Yag-",\par
        "outputId": "b62e3bf3-3a2b-4b80-a47e-a1b3194fdb96",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "Array of all the even integers from 20 to 35\\n",\par
            "[20 22 24 26 28 30 32 34]\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 6. Create a 3x3 matrix with values ranging from 0 to 8"\par
      ],\par
      "metadata": \{\par
        "id": "NaOM308NsRpZ"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np\\n",\par
        "matrix =  np.arange(0, 9).reshape(3,3)\\n",\par
        "print(matrix)"\par
      ],\par
      "metadata": \{\par
        "id": "tOlEVH7BYceE",\par
        "outputId": "635128f9-7b98-4e04-e323-688c1ec7b3c8",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "[[0 1 2]\\n",\par
            " [3 4 5]\\n",\par
            " [6 7 8]]\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 7. Concatenate a and b \\n",\par
        "## a = np.array([1, 2, 3]), b = np.array([4, 5, 6])"\par
      ],\par
      "metadata": \{\par
        "id": "hQ0dnhAQuU_p"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import numpy as np\\n",\par
        "a = np.array([1, 2, 3])\\n",\par
        "b = np.array([4, 5, 6])\\n",\par
        "con = np.concatenate((a,b))\\n",\par
        "print(con)"\par
      ],\par
      "metadata": \{\par
        "id": "rAPSw97aYfE0"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "# Pandas"\par
      ],\par
      "metadata": \{\par
        "id": "dlPEY9DRwZga"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 8. Create a dataframe with 3 rows and 2 columns"\par
      ],\par
      "metadata": \{\par
        "id": "ijoYW51zwr87"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import pandas as pd\\n"\par
      ],\par
      "metadata": \{\par
        "id": "T5OxJRZ8uvR7"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import pandas as pd\\n",\par
        "data = \{\\n",\par
        "  \\"Name\\": [\\"hari\\",\\"ram\\", \\"naveen\\"],\\n",\par
        "  \\"Age\\": [20, 21, 21]\\n",\par
        "\}\\n",\par
        "df = pd.DataFrame(data)\\n",\par
        "print(df) "\par
      ],\par
      "metadata": \{\par
        "id": "xNpI_XXoYhs0",\par
        "outputId": "ce4b39bd-04ce-42fb-ef89-429560ce0737",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "     Name  Age\\n",\par
            "0    hari   20\\n",\par
            "1     ram   21\\n",\par
            "2  naveen   21\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023"\par
      ],\par
      "metadata": \{\par
        "id": "UXSmdNclyJQD"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import pandas as pd \\n",\par
        "datee = pd.date_range(start ='1-1-2023', end ='02-10-2023')  \\n",\par
        "for ser in datee:\\n",\par
        "    print(ser)"\par
      ],\par
      "metadata": \{\par
        "id": "dgyC0JhVYl4F",\par
        "outputId": "52347ada-dc07-4baa-a0ac-ba0f68d8cab6",\par
        "colab": \{\par
          "base_uri": "{\field{\*\fldinst{HYPERLINK "https://localhost:8080/"}}{\fldrslt{\ul\cf1 https://localhost:8080/}}}\f0\fs22 "\par
        \}\par
      \},\par
      "execution_count": null,\par
      "outputs": [\par
        \{\par
          "output_type": "stream",\par
          "name": "stdout",\par
          "text": [\par
            "2023-01-01 00:00:00\\n",\par
            "2023-01-02 00:00:00\\n",\par
            "2023-01-03 00:00:00\\n",\par
            "2023-01-04 00:00:00\\n",\par
            "2023-01-05 00:00:00\\n",\par
            "2023-01-06 00:00:00\\n",\par
            "2023-01-07 00:00:00\\n",\par
            "2023-01-08 00:00:00\\n",\par
            "2023-01-09 00:00:00\\n",\par
            "2023-01-10 00:00:00\\n",\par
            "2023-01-11 00:00:00\\n",\par
            "2023-01-12 00:00:00\\n",\par
            "2023-01-13 00:00:00\\n",\par
            "2023-01-14 00:00:00\\n",\par
            "2023-01-15 00:00:00\\n",\par
            "2023-01-16 00:00:00\\n",\par
            "2023-01-17 00:00:00\\n",\par
            "2023-01-18 00:00:00\\n",\par
            "2023-01-19 00:00:00\\n",\par
            "2023-01-20 00:00:00\\n",\par
            "2023-01-21 00:00:00\\n",\par
            "2023-01-22 00:00:00\\n",\par
            "2023-01-23 00:00:00\\n",\par
            "2023-01-24 00:00:00\\n",\par
            "2023-01-25 00:00:00\\n",\par
            "2023-01-26 00:00:00\\n",\par
            "2023-01-27 00:00:00\\n",\par
            "2023-01-28 00:00:00\\n",\par
            "2023-01-29 00:00:00\\n",\par
            "2023-01-30 00:00:00\\n",\par
            "2023-01-31 00:00:00\\n",\par
            "2023-02-01 00:00:00\\n",\par
            "2023-02-02 00:00:00\\n",\par
            "2023-02-03 00:00:00\\n",\par
            "2023-02-04 00:00:00\\n",\par
            "2023-02-05 00:00:00\\n",\par
            "2023-02-06 00:00:00\\n",\par
            "2023-02-07 00:00:00\\n",\par
            "2023-02-08 00:00:00\\n",\par
            "2023-02-09 00:00:00\\n",\par
            "2023-02-10 00:00:00\\n"\par
          ]\par
        \}\par
      ]\par
    \},\par
    \{\par
      "cell_type": "markdown",\par
      "source": [\par
        "## 10. Create 2D list to DataFrame\\n",\par
        "\\n",\par
        "lists = [[1, 'aaa', 22],\\n",\par
        "         [2, 'bbb', 25],\\n",\par
        "         [3, 'ccc', 24]]"\par
      ],\par
      "metadata": \{\par
        "id": "ZizSetD-y5az"\par
      \}\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]"\par
      ],\par
      "metadata": \{\par
        "id": "_XMC8aEt0llB"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \},\par
    \{\par
      "cell_type": "code",\par
      "source": [\par
        "import pandas as pd    \\n",\par
        "lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]   \\n",\par
        "df = pd.DataFrame(lists, columns =['Tag', 'letters', 'numbers']) \\n",\par
        "print(df )"\par
      ],\par
      "metadata": \{\par
        "id": "knH76sDKYsVX"\par
      \},\par
      "execution_count": null,\par
      "outputs": []\par
    \}\par
  ]\par
\}\par
}
 
