import os
# Python os system function allows us to run a command in the Python script, just like if I was running it in my shell. 
# For example: import os currentFiles = os.system("users > users.txt")

import cv2
# Sincronizador de edgebox(aquela moldura vermelha)
# "Visão computacional e processamento de imagens"
# "OpenCV provides a real-time optimized Computer Vision library, tools, and hardware."

import numpy as np
# NumPy is a Python library used for working with arrays. It also has functions for working in domain
# of linear algebra, fourier transform, and matrices.

import matplotlib.pyplot as plt
# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python

import tensorflow as tf
# É um sistema para criação e treinamento de redes neurais para detectar e decifrar padrões e correlações

treinar_novo_modelo = True

if treinar_novo_modelo:
    # Carrega direto do tensorflow(não precisa carregar o csv)
    mnist = tf.keras.datasets.mnist

    # Vai dividir em duas data
    # train data é o que é utilizado para treinar  modelo
    # test data é so pra acessar o modelo

    # ?x é a imagem em si a escrita de entrada em si
    # ?y é só a classificação(o número do digito)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # agora vai NORMALIZAR a escala de cinza que vai de 0 a 255 para 0 a 1
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Define o modelo como sequencial
    model =tf.keras.models.Sequential()
    # ?Adiciona camadas(
    # ao invés de ser uma grade, fica uma forma é 28*28 = 784 Sequencial)
    model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
    # A função de ativação é reLU
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    # Camada de saida((softmax garante que cada neuronio irá apontar para 1 so resltado))
    # softmax maximiza a probabilidade de cada digito ser classificado corretamente
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)
    model.save('handwritten.model')
    # treinar_novo_modelo = False

else:
    model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        # Tem que inverter pois o default é white on black e não black on white
        img = np.invert(np.array([img]))
        np.array([img])
        prediction = model.predict(img)
        print(f"O numero é provavelmente {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Erro!")
    finally:
        image_number += 1

# Utilizar o colab
# fixar a questao to treinamento/teste
#https://www.youtube.com/watch?v=bte8Er0QhDg