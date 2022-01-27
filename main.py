import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig = plt.figure()
ax = fig.add_subplot(111)
# establecemos los limites de la gráfica
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])

data = []

def set_point(event):
    # el cluster guarda tanto la clase como el simbolo que graficará 
    cluster, color = (('o', 0), 'b') if (event.button == 1) else (('x',1), 'r')

    # guardamos una tupla con los valore X y Y, así como su clase correspondiente
    data.append(((event.xdata, event.ydata), cluster[1]))

    plt.plot(event.xdata, event.ydata, cluster[0], color=color)
    fig.canvas.draw()

def run():
    print("Iteraciones máximas: ", max_iter.get())
    print("Data: ", data)

window = tkinter.Tk()
window.geometry("600x600")

# añade el gráfico de matplotlib a la interfaz de tkinter
FigureCanvasTkAgg(fig, window).get_tk_widget().pack()

tkinter.Label(window, text="Max iter: ").pack()
max_iter = tkinter.Entry(window)
max_iter.pack()

tkinter.Button(window, text="Run", command=run).pack()

# escucha los eventos del mouse sobre el gráfico
fig.canvas.mpl_connect('button_press_event', set_point)
# termina el programa al hacer click en la X roja de la ventana
window.protocol('WM_DELETE_WINDOW', lambda: quit())
window.mainloop()


