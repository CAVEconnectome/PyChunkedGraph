from typing import List, Union, Dict, Set

import numpy as np


class ChunkTask:
    def __init__(self, layer: int, coords: np.ndarray, parent_id: str = None):
        self._layer: int = layer
        self._coords: np.ndarray = coords
        self._id: str = ChunkTask.get_id(layer, coords)
        self._children: Set = set()
        self._children_coords: List = None
        self._completed: bool = False
        self._parent_id: str = parent_id
        self._dependencies = 0

    @staticmethod
    def get_id(layer: int, coords: np.ndarray):
        return f"{layer}_{'_'.join(map(str, coords))}"

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

    @property
    def parent_id(self) -> str:
        return self._parent_id

    @parent_id.setter
    def parent_id(self, value: str):
        self._parent_id = value

    @property
    def layer(self) -> int:
        return self._layer

    @layer.setter
    def layer(self, value: int):
        self._layer = value

    @property
    def coords(self) -> np.ndarray:
        return self._coords

    @coords.setter
    def coords(self, value: np.ndarray):
        self._coords = value

    @property
    def children(self) -> List:
        return self._children

    @children.setter
    def children(self, value: List):
        self._children = value

    @property
    def children_coords(self) -> List:
        return self._children_coords

    @children_coords.setter
    def children_coords(self, value: List):
        self._children_coords = value        

    @property
    def dependencies(self) -> int:
        return self._dependencies

    def remove_child(self, child_id: str) -> None:
        self.children.remove(child_id)
        self._dependencies -= 1

    def add_child(self, child_id: str):
        self.children.add(child_id)
        self._dependencies += 1
