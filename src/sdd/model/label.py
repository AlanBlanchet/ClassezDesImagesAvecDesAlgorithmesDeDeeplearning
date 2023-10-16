from typing import overload

import torch
from attrs import define, field


@define
class LabelMap:
    labels: list[str] = field(repr=False)
    _label2id: dict[str, int] = field(init=False)
    _id2label: dict[int, str] = field(init=False)

    def __attrs_post_init__(self):
        self._label2id = {label: i for i, label in enumerate(sorted(set(self.labels)))}
        self._id2label = {v: k for k, v in self._label2id.items()}

    def __get(self, item):
        if isinstance(item, str):
            return self._label2id[item]
        return self._id2label[item]

    @overload
    def __getitem__(self, item: int) -> str:
        ...

    @overload
    def __getitem__(self, item: str) -> int:
        ...

    @overload
    def __getitem__(self, item: list[int]) -> list[str]:
        ...

    @overload
    def __getitem__(self, item: list[str]) -> list[int]:
        ...

    @overload
    def __getitem__(self, item: torch.Tensor) -> list[str]:
        ...

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.__get(i) for i in item]
        elif isinstance(item, torch.Tensor):
            return [self.__get(i.item()) for i in item]
        return self.__get(item)

    def __len__(self):
        return len(self._id2label)


if __name__ == "__main__":
    labels = ["A", "B", "C", "A"]

    label_map = LabelMap(labels)
    print(label_map)

    output = label_map[0]
    print(output)
    output = label_map["A"]
    print(output)
    output = label_map[[0, 1]]
    print(output)
    output = label_map[["A", "B"]]
    print(output)
