import argparse


class Args:
    def __init__(self):
        parser = argparse.ArgumentParser()
        #  here goes your arguments
        #  ex:
        #  parser.add_argument('--name')
        kargs = parser.parse_known_args()[0]
        #  here you assign arguments to typed attributes of class Args
        #  ex:
        #  self.name: str = kargs.name


args = None


def get_args() -> Args:
    global args
    if args is None:
        args = Args()
    return args
