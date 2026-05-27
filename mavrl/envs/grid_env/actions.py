from enum import IntEnum
import numpy as np

class Action(IntEnum):
    # counter-clockwise order
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    STAY = 4

ACTION_DELTA = {
    Action.RIGHT: np.array([0, 1]),
    Action.UP: np.array([-1, 0]),
    Action.LEFT: np.array([0, -1]),
    Action.DOWN: np.array([1, 0]),
    Action.STAY: np.array([0, 0])
}