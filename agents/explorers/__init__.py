import typing as T
from .random_explorer import RandomExplorer
from .noisy_explorer import NoisyExplorer
from .models import RandomExplorerParams, NoisyExplorerParams
AnyExplorer = T.Union[RandomExplorer, NoisyExplorer]
