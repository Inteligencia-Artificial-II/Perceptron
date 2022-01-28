import matplotlib.pyplot as plt
from src.UI import render_ui

class Perceptron:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # establecemos los limites de la gráfica
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        # guarda la información de los puntos sobre la gráfica
        self.data = []
        # llama a la interfaz gráfica
        render_ui(self)

    def set_point(self, event):
        right_click = 1
        # el cluster guarda tanto la clase como el simbolo que graficará 
        cluster, color = (('o', 0), 'b') if (event.button == right_click) else (('x',1), 'r')

        # evitamos puntos que se encuentren fuera del plano
        if (event.xdata == None or event.ydata == None): return

        # guardamos una tupla con los valore X y Y, así como su clase correspondiente
        self.data.append(((event.xdata, event.ydata), cluster[1]))

        plt.plot(event.xdata, event.ydata, cluster[0], color=color)
        self.fig.canvas.draw()

    def run(self):
        print("Iteraciones máximas: ", self.max_iter.get())
        print("Tasa de aprendizaje: ", self.learning_rate.get())
        print("Data: ", self.data)

    def init_weights(self):
        print("Inicializando pesos")
