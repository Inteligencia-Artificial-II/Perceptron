from tkinter import Tk, Frame, Label, Entry, Button
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Definimos la interfaz gráfica
def render_ui(self):
    self.window = Tk()
    self.window.title('Perceptrón')
    self.window.geometry("600x600")


    Label(self.window, text="Perceptrón", font=("Arial", 20)).grid(row=0, columnspan=4)
    # añade el gráfico de matplotlib a la interfaz de tkinter
    FigureCanvasTkAgg(self.fig, self.window).get_tk_widget().grid(row=1, columnspan=4)

    # contendrá un segmento de la interfaz
    self.container = Frame(self.window)

    Label(self.container, text="Epocas máximas:").grid(row=0, column=0)
    self.max_iter = Entry(self.container)
    self.max_iter.grid(row=0, column=1)

    Label(self.container, text="Tasa de aprendizaje:").grid(row=1, column=0)
    self.learning_rate = Entry(self.container)
    self.learning_rate.grid(row=1, column=1)

    Button(self.container, text="Inicializar pesos", command=self.init_weights).grid(row=0, column=2)
    Button(self.container, text="Analizar", command=self.run).grid(row=1, column=2)

    self.container.grid(row=2, columnspan=4)

    # escucha los eventos del mouse sobre el gráfico
    self.fig.canvas.mpl_connect('button_press_event', self.set_point)

    # termina el programa al hacer click en la X roja de la ventana
    self.window.protocol('WM_DELETE_WINDOW', lambda: quit())
    self.window.mainloop()