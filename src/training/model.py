# Importe o TensorFlow e o módulo de datasets do Keras
import tensorflow as tf

from keras import layers
from tensorflow import keras

from shared import utils as utils

def trainModel():
    FashionMNIST = tf.keras.datasets.fashion_mnist

    (imagens_treino, labels_treino), (imagens_teste, labels_teste) = FashionMNIST.load_data()

    utils.printHead()

    print("| INFORMAÇÕES DO DATASET                              ")
    print("|                                                     ")
    print("| Formato das imagens de treino:", imagens_treino.shape)
    print("| Quantidade de labels de treino:", len(labels_treino) )
    print("| Formato das imagens de teste:", imagens_teste.shape  )
    print("| Quantidade de labels de teste:", len(labels_teste)   )

    utils.printHead()


    model = keras.Sequential([
        layers.Input(shape=(784,)), 
        layers.Dense(128, activation='relu'), 
        layers.Dense(10, activation='softmax') # Camada de saída
    ])

    utils.printHead()

    model.summary()

    utils.printHead()