class RandomExplorerParams:
    def __init__(self, init_ep: float, final_ep: float, decay_ep: float):
        self.init_ep: float = init_ep
        self.final_ep: float = final_ep
        self.decay_ep: float = decay_ep
