from __future__ import annotations

from pathlib import Path
from typing import Literal

import ase
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from ..util import optional_import_error_message
from .base import DatasetConfigBase


@data_registry.register
class OMAT24DatasetConfig(DatasetConfigBase):
    type: Literal["omat24"] = "omat24"
    """Discriminator for the OMAT24 dataset."""

    src: Path
    """The path to the OMAT24 dataset."""

    @override
    def create_dataset(self):
        return OMAT24Dataset(self)


class OMAT24Dataset(Dataset[ase.Atoms]):
    def __init__(self, config: OMAT24DatasetConfig):
        super().__init__()
        self.config = config

        with optional_import_error_message("fairchem"):
            from fairchem.core.datasets import AseDBDataset  # type: ignore[reportMissingImports] # noqa

        self.dataset = AseDBDataset(config={"src": str(self.config.src)})

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        atoms = self.dataset.get_atoms(idx)
        return atoms

    def __len__(self) -> int:
        return len(self.dataset)
