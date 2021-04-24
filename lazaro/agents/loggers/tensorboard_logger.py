import time
import subprocess
import webbrowser
import threading
from torch.utils.tensorboard import SummaryWriter


class TensorBoard(SummaryWriter):
    def __init__(self, path: str):
        super(TensorBoard, self).__init__(path)
        self.process = None
        threading.Thread(target=lambda: time.sleep(3) or self.launch(path)).start()

    def launch(self, path):
        try:
            self.process = subprocess.Popen(["tensorboard", "--logdir=" + path, "--load_fast=false"], shell=False)
            threading.Thread(target=lambda: time.sleep(1) or webbrowser.open("http://localhost:6006")).start()
        except FileNotFoundError:
            print("tensorboard is not installed")

    def __del__(self):
        if self.process:
            self.process.kill()
