import matplotlib.pyplot as plt
from tkinter import NORMAL, DISABLED, messagebox
from src.UI import render_gui, render_conv
import numpy as np

class Perceptron:
    def __init__(self):
        self.conv_fig = plt.figure(2)
        self.conv_ax = self.conv_fig.add_subplot(111)
        plt.xlabel('Número de épocas')
        plt.ylabel('Error acumulado')

        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        # establecemos los limites de la gráfica
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])

        self.Y = [] # guarda los clusters
        self.X = [] # guarda los puntos de entrenamiento
        self.test_data = [] # guarda los puntos para evaluar

        # bandera para evaluar datos despues del entrenamiento
        self.is_training = True

        # Parámetros para el algoritmos
        self.epochs = 0 # epocas o generaciones máximas
        self.W = [] # pesos
        self.lr = 0.0 # tasa de aprendizaje

        # Error acumulado
        self.acum_error = []

        self.iter = None

        # llama a la interfaz gráfica
        render_gui(self)

    def set_point(self, event):
        right_click = 1
        # el cluster guarda tanto la clase como el simbolo que graficará 
        cluster = 1 if (event.button == right_click) else 0

        # evitamos puntos que se encuentren fuera del plano
        if (event.xdata == None or event.ydata == None): return

        if (self.epochs == self.iter): return

        # guardamos una tupla con las coordenadas X y Y capturadas por el canvas,
        # así como su clase correspondiente
        if (self.is_training):
            # se capturan los datos para entrenar
            self.Y = np.append(self.Y, cluster)
            if (len(self.X) == 0):
                self.X = np.array([[event.xdata, event.ydata]])
            else:
                self.X = np.append(self.X, [[event.xdata, event.ydata]], axis=0)
            self.plot_point((event.xdata, event.ydata), cluster)
        else:
            # se capturan los datos para evaluar
            if (len(self.test_data) == 0):
                self.test_data = np.array([[event.xdata, event.ydata]])
            else:
                self.test_data = np.append(self.test_data, [[event.xdata, event.ydata]], axis=0)
            self.plot_point((event.xdata, event.ydata))

        # si se ingresan como minimo 2 puntos de clases distintas,
        # se habilita el botón para inicializar pesos
        if (len(np.unique(self.Y)) > 1):
            self.weight_btn["state"] = NORMAL

        self.fig.canvas.draw()

    def plot_point(self, point: tuple, cluster=None):
        """Toma un array de tuplas y las añade los puntos en la figura con el
        color de su cluster"""
        plt.figure(1)
        if (cluster == None):
            plt.plot(point[0], point[1], 'o', color='k')
        else:
            color = 'b' if cluster == 1 else 'r'
            shape = 'o' if cluster == 1 else 'x'
            plt.plot(point[0], point[1], shape, color=color)

    def plot_training_data(self):
        """Grafica los datos de entrenamiento"""
        plt.figure(1)
        for i in range(len(self.Y)):
            self.plot_point(self.X[i], self.Y[i])

    def clear_plot(self, figure = 1):
        """Borra los puntos del canvas"""
        plt.figure(figure)
        plt.cla()
        if figure == 1:
            self.ax.set_xlim([-5, 5])
            self.ax.set_ylim([-5, 5])
        self.fig.canvas.draw()

    def run(self):
        """es ejecutada cuando el botón de «entrenar» es presionado"""
        # obtenemos los datos de la interfaz gráfica
        try:
            self.lr = float(self.learning_rate.get())
            if self.lr <= 0 or self.lr >= 1:
                messagebox.showwarning("Error", "El tasa de aprendizaje debe de ser mayor a 0 y menor a 1")
                self.max_iter.delete(0, 'end')
                self.learning_rate.delete(0, 'end')
                return
        except:
            if self.learning_rate.get() == "":
                self.lr = 0.3
            else:
                messagebox.showwarning("Error", "Asegurese de ingresar datos númericos validos")
                self.max_iter.delete(0, 'end')
                self.learning_rate.delete(0, 'end')
                return
        try:
            self.epochs = int(self.max_iter.get())
            if self.epochs <= 0:
                messagebox.showwarning("Error", "No pueden haber épocas menores o iguales a cero")
                self.max_iter.delete(0, 'end')
                self.learning_rate.delete(0, 'end')
                return
        except:
            if (self.max_iter.get() == ""):
                self.epochs = 25
            else:
                messagebox.showwarning("Error", "Asegurese de ingresar datos númericos validos")
                self.max_iter.delete(0, 'end')
                self.learning_rate.delete(0, 'end')
                return

        self.is_training = False
        self.container_before.grid_remove()
        self.container_after.grid(row=2, columnspan=4)

        # mandamos a entrenar al algoritmo
        self.train()

    def init_weights(self):
        """Se ejecuta al presionar el botón «inicializar pesos»"""
        # Sacamos el random para los pesos
        self.W = np.random.uniform(-1, 1, self.X.shape[1] + 1)

        # habilitamos el botón para iniciar el algoritmo
        self.run_btn["state"] = NORMAL

        # gráficamos la recta inicial que separará los datos
        self.x1Line = np.linspace(-5, 5, 100)
        self.clear_plot()
        self.plot_training_data()
        self.plot_line('r')


    def evaluate(self):
        """Toma los datos de prueba y los categoriza"""
        # Se posicionan los datos que se graficarán
        self.clear_plot()
        self.plot_training_data()
        self.plot_line('b')

        # obtenemos las clases correctas del set de datos de prueba
        for i in self.test_data:
            res = np.dot(self.W[: -1], i) + self.W[-1]
            cluster = 1 if res > 0 else 0
            self.plot_point(i, cluster)

        # gráficamos todos los datos en el plano
        self.fig.canvas.draw()

    def train(self):
        """Entrena el perceptrón"""
        m, _ = self.X.shape
        pw = 0
        done = False
        y = np.zeros_like(self.Y)
        self.iter = 0

        while(not done):
            done = True
            for i in range(m):
                net = np.dot(self.W[: -1], self.X[i, :]) + self.W[-1]
                pw = 1 if net > 0 else 0
                error = self.Y[i] - pw
                y[i] = pw
                if error != 0:
                    done = False
                    self.W[:-1] = self.W[:-1] + np.multiply((self.lr * error), self.X[i, :])
                    self.W[-1] = self.W[-1] + self.lr * error
                    # gráficamos la recta que separa los datos
                    self.plot_line('g')
            self.acum_error.append(np.sum(np.abs(self.Y - y)))
            self.iter += 1
            if (self.iter == self.epochs):
                self.is_converge['text'] = "Límite de epocas alcanzada (set de datos sin solución)"
                done = True

        if (self.iter != self.epochs):
            self.is_converge['text'] = f'El set de datos convergió en {self.iter} epocas'
            self.analyse["state"] = NORMAL
        self.plot_line('b')
        render_conv(self)
        plt.figure(2)
        plt.plot([ i for i in range(1, len(self.acum_error) + 1) ], self.acum_error)
        plt.figure(1)

    def plot_line(self, color):
        """gráfica la recta que clasifica los datos del plano"""
        plt.figure(1)
        self.x2Line = (-self.W[0] * self.x1Line - self.W[-1]) / self.W[1]
        plt.plot(self.x1Line, self.x2Line, color=color)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def restart(self):
        """devuelve los valores y elementos gráficos a su estado inicial"""
        # cierra la ventana de la gráfica de convergencia
        self.conv_window.destroy()
        self.clear_plot(2)
        plt.figure(1)
        self.acum_error.clear()
        self.container_before.grid(row=2, columnspan=4)
        self.container_after.grid_remove()
        self.X = []
        self.Y = []
        self.test_data = []
        self.is_training = True
        self.epochs = 0
        self.W = []
        self.lr = 0.0
        self.iter = None
        self.analyse["state"] = DISABLED
        self.weight_btn["state"] = DISABLED
        self.run_btn["state"] = DISABLED
        self.learning_rate.delete(0, 'end')
        self.max_iter.delete(0, 'end')
        self.clear_plot()
