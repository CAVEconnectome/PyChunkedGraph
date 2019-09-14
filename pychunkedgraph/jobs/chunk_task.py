from typing import List, Union

import numpy as np


class ChunkTask:
    def __init__(self, layer: int, coords: np.ndarray):
        self._layer: int = layer
        self._coords: np.ndarray = coords
        self._id: str = "_".join(coords)
        self._children: List = []
        self._completed: bool = False

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, value: str):
        self._id = value

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
    def completed(self) -> bool:
        return self._completed

    @completed.setter
    def completed(self, value: bool):
        self._completed = value

    def get_child(self, id: str) -> Union[ChunkTask, None]:
        for child in self.children:
            if child.id == id:
                return child
        return None

    def add_child(self, child_chunk_task: ChunkTask):
        self.children.append(child_chunk_task)

    def add_children(self, child_chunk_tasks: List[ChunkTask]):
        self.children += child_chunk_tasks

