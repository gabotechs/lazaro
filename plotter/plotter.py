import typing as T
import matplotlib.pyplot as plt
plt.switch_backend("tkagg")
plt.ion()
plt.show(block=False)


class Plotter:
    def __init__(self):
        self.fig: plt.Figure = plt.figure()
        self.ax: plt.Axes = self.fig.add_subplot(111)

    def plot(self, raw_data: T.List[T.Union[int, float]], aliasing: float = .0):
        data = []
        for y in raw_data:
            data.append(y if not len(data) else data[-1]*aliasing+y*(1-aliasing))
        self.ax.clear()
        self.ax.plot(data)
        self.fig.canvas.draw()
