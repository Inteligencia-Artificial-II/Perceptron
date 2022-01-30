import matplotlib.pyplot as plt
from src.UI import render_ui
import numpy as np

class Perceptron:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # establecemos los limites de la gráfica
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        # guarda la información de los puntos sobre la gráfica
        self.data = []
        # puntos para evaluar
        self.test_data = []
        # bandera para evaluar datos despues del entrenamiento
        self.is_training = True

        self.gens = 100 # epocas o generaciones máximas
        self.W = [] # pesos
        self.lr = 0.1 # tasa de aprendizaje

        # llama a la interfaz gráfica
        render_ui(self)

    def set_point(self, event):
        right_click = 1
        # el cluster guarda tanto la clase como el simbolo que graficará 
        cluster, color = (('o', 0), 'b') if (event.button == right_click) else (('x',1), 'r')

        # evitamos puntos que se encuentren fuera del plano
        if (event.xdata == None or event.ydata == None): return

        # guardamos una tupla con los valore X y Y, así como su clase correspondiente
        if (self.is_training): 
            self.data.append(((event.xdata, event.ydata), cluster[1]))
            plt.plot(event.xdata, event.ydata, cluster[0], color=color)
        else:
            self.test_data.append((event.xdata, event.ydata))
            plt.plot(event.xdata, event.ydata, 'o', color='k')

        self.fig.canvas.draw()

    def run(self):
        self.is_training = False
        self.container_before.grid_remove()
        self.container_after.grid(row=2, columnspan=4)
        self.lr = self.learning_rate.get()
        self.gens = self.max_iter.get()
        self.train()

        # llamar al algoritmo de entrenamiento

    def init_weights(self):
        print("Inicializando pesos")
    
    def evaluate(self):
        print("Test data: ", self.test_data)
        # llamar al algoritmo de entrenamiento


    def train(self):
        # obtenemos los valores de X
        aux_arr = [[self.data[i][0][0], self.data[i][0][1]] for i in range(len(self.data))]
        X = np.array(aux_arr[:])
        del aux_arr
        # obtenemos los valores de Y (las clases)
        aux_arr = [self.data[i][1] for i in range(len(self.data))]
        Y = np.array(aux_arr[:])

        m, n = X.shape
        y = np.zeros((m))
        v = np.empty((m))
        conv = []

        # Normalización
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        xNorm = []
        for i in range(n):
            xNorm.append((X[:,i] - mu[i])/sigma[i])
        X = np.transpose(np.array(xNorm))
        b = -0.5
        W = [0.2, 0.5]
        a = 0.03
        it = 0
        error = np.sum((np.abs(Y-y)))


        while(error != 0):
            for i in range(m):
                v[i] = np.dot(W, X[i]) + b
                if v[i] > 0:
                    y[i] = 1
                else:
                    y[i] = 0
                if (Y[i] - y[i]) != 0:
                    W = W + a*X[i]
                    b = b + a
                error = np.sum(np.abs(Y-y))
                conv.append(error)
                it += 1
        
        print("si me salio jefa: ", W)

