import torch

from sdd.model.label import LabelMap


class BB_Collator:
    def __init__(self, label_map: LabelMap, box_max_amount: int):
        self.label_map = label_map
        self.box_max_amount = box_max_amount

    def __call__(self, batch):
        B = len(batch)
        annot_max = torch.zeros(B, self.box_max_amount, 4)

        for i, elem in enumerate(batch):
            for annot_name in ["annotations", "annotations_unscaled"]:
                annots = annot_max.clone()
                annot = elem[annot_name]
                s = annot.shape
                annots[i, : s[0], : s[1]] = annot
                elem[annot_name] = annots[i]

            annotations_mask = torch.zeros(
                self.box_max_amount,
            )
            annotations_mask[: s[0]] = 1
            elem["annotations_mask"] = annotations_mask

            target = torch.tensor(self.label_map[elem["target"]])

            elem["target"] = target.unsqueeze(dim=-1)[
                torch.zeros((self.box_max_amount,), dtype=torch.int)
            ]

        return torch.utils.data.default_collate(batch)


if __name__ == "__main__":
    collator = BB_Collator(6)

    batch = []

    a = torch.tensor([1, 5])

    batch.append({"annotations": torch.ones((2, 4)), "target": torch.tensor(5)})
    batch.append({"annotations": torch.zeros((3, 4)), "target": torch.tensor(2)})

    collated = collator(batch)

    for col in collated.keys():
        print(f"{col}", collated[col].shape)
