import logging
import os
import sys
from abc import ABC


class BaseObject(ABC):
    def __init__(self):
        self.log: logging.Logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter('%(asctime)s %(module)-20s %(levelname)-5s %(message)s')
        handler.setFormatter(formatter)
        if not self.log.hasHandlers():
            self.log.addHandler(handler)
        self.log.setLevel(self._get_debug_level())

    def _get_debug_level(self) -> int:
        level = logging.ERROR
        debug_info = os.getenv('LZ_DEBUG', '0')
        for debug_element_level in debug_info.split(","):
            split_debug_element_level = debug_element_level.split(":")
            if len(split_debug_element_level) == 2:
                element_name = split_debug_element_level[0]
                debug_level_str = split_debug_element_level[1]
                if element_name == self.__class__.__name__ and debug_level_str.isnumeric():
                    debug_level = int(debug_level_str)
                else:
                    continue

            elif len(split_debug_element_level) == 1 and split_debug_element_level[0].isnumeric():
                debug_level = int(split_debug_element_level[0])

            else:
                print(f"incorrect debug specification {debug_element_level}")
                continue

            if debug_level == 0:
                level = logging.ERROR
            elif debug_level == 1:
                level = logging.WARNING
            elif debug_level == 2:
                level = logging.INFO
            elif debug_level == 3:
                level = logging.DEBUG
        return level
