from asyncio.windows_events import NULL
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
        # guarda los clusters
        self.Y = []
        self.X = []
        # puntos para evaluar
        self.test_data = []
        # bandera para evaluar datos despues del entrenamiento
        self.is_training = True

        # Parámetros para el algoritmos
        self.gens = 100 # epocas o generaciones máximas
        self.W = [] # pesos
        self.lr = 0.1 # tasa de aprendizaje

        # llama a la interfaz gráfica
        render_ui(self)

    def set_point(self, event):
        right_click = 1
        # el cluster guarda tanto la clase como el simbolo que graficará 
        cluster, color = (('o', 0), 'b') if (event.button == right_click) else (('x', 1), 'r')

        # evitamos puntos que se encuentren fuera del plano
        if (event.xdata == None or event.ydata == None): return

        # guardamos una tupla con los valore X y Y, así como su clase correspondiente
        print(f"points: {self.X}")
        print(f"clusters: {self.Y}")
        if (self.is_training):
            self.Y = np.append(self.Y, cluster[1])
            # Solución temporal
            if (len(self.X) == 0):
                self.X = np.array([[event.xdata, event.ydata]])
            else:
                self.X = np.append(self.X, [[event.xdata, event.ydata]], axis=0)
            plt.plot(event.xdata, event.ydata, cluster[0], color=color)
        else:
            # Solución temporal
            if (len(self.test_data) == 0):
                self.test_data = np.array([[event.xdata, event.ydata]])
            else:
                self.test_data = np.append(self.test_data, [[event.xdata, event.ydata]], axis=0)
            plt.plot(event.xdata, event.ydata, 'o', color='k')
        # print(f"points: {self.X}")
        # print(f"clusters: {self.Y}")
        # print(f"test data: {self.test_data}")
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
        print("Inicializando pesos...")
        # Sacamos el random para los pesos
        self.W = np.random.uniform(0.5, 0.8, self.X.shape[1])
        # Agregamos el bias al final del array de pesos (W = n + 1)
        self.W = np.append(self.W, np.random.uniform(-1, 1))
        print(f"W: {self.W}")
    
    def evaluate(self):
        print("Test data: ", self.test_data)
        # llamar al algoritmo de entrenamiento

    def train(self):
        m, n = self.X.shape
        y = np.zeros((m))
        v = np.empty((m))
        conv = []

        # Normalización
        mu = np.mean(self.X, axis=0)
        sigma = np.std(self.X, axis=0)
        xNorm = []
        for i in range(n):
            xNorm.append((self.X[:,i] - mu[i])/sigma[i])
        self.X = np.transpose(np.array(xNorm))
        a = 0.03
        it = 0
        error = np.sum((np.abs(self.Y - y)))

        print(f"pruebas: {self.W[-1]}, {self.W[: -1]}")
        print(f"pruebas: {self.W[-1].shape}, {self.W[: -1].shape}")

        while(error != 0):
            for i in range(m):
                v[i] = np.dot(self.W[: -1], self.X[i]) + self.W[-1]
                if v[i] > 0:
                    y[i] = 1
                else:
                    y[i] = 0
                if (self.Y[i] - y[i]) != 0:
                    self.W[:-1] = self.W[:-1] + a * self.X[i]
                    self.W[-1] = self.W[-1] + a
                error = np.sum(np.abs(self.Y - y))
                conv.append(error)
                it += 1
        
        print("Valor de W: ", self.W)