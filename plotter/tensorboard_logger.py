import typing as T
import time
import subprocess
import threading
import webbrowser
from torch.utils.tensorboard import SummaryWriter


class TensorBoardThread(threading.Thread):
    def __init__(self, cmd: str):
        super(TensorBoardThread, self).__init__()
        self.cmd: str = cmd
        self.process: T.Union[None, subprocess.Popen] = None

    def run(self):
        try:
            self.process = subprocess.Popen(self.cmd.split(), shell=False)
            time.sleep(2)
            webbrowser.open_new("http://localhost:6006")
        except FileNotFoundError:
            print("tensorboard not found")


class TensorBoard(SummaryWriter):
    def __init__(self, path: str):
        super(TensorBoard, self).__init__(path)
        self.path: str = path
        self.tb_thread = TensorBoardThread("tensorboard --logdir="+path)
        self.tb_thread.start()

    def __del__(self):
        self.tb_thread.process.kill()
        self.tb_thread.join()
