import typing as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as pltColors

plt.switch_backend("tkagg")
plt.ion()
plt.show(block=False)


class Renderer:
    def __init__(self, title: str, size: T.Union[int, T.Tuple[int, int]]):
        self.title = title
        if isinstance(size, int):
            size = (size, 1)
        self.size = size
        self.fig: T.Union[None, plt.Figure] = None
        self.ax: T.List[plt.Axes] = []

    def init(self):
        self.fig = plt.figure(len(plt.get_fignums()) + 1)
        self.fig.suptitle(self.title)
        self.ax: T.List[plt.Axes] = []

        for i in range(self.size[0] * self.size[1]):
            self.ax.append(self.fig.add_subplot(self.size[0], self.size[1], i + 1))

    def render(self, index: int, title: str, img: np.ndarray):
        if not self.fig:
            self.init()
        self.ax[index].imshow(img)
        self.ax[index].set_title(title, loc="left", y=0.1)
        self.ax[index].set_axis_off()
        self.fig.canvas.draw()
