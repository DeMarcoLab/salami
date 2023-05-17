from dataclasses import dataclass
from pathlib import Path

from fibsem import utils as futils
from fibsem.structures import (
    BeamSettings,
    FibsemDetectorSettings,
    ImageSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemRectangle,
    MicroscopeState,
    Point,
)
from fibsem.patterning import FibsemMillingStage


@dataclass
class SalamiSettings:
    n_steps: int
    step_size: float
    image: list[tuple[ImageSettings, BeamSettings, FibsemDetectorSettings]] = None
    mill: FibsemMillingStage = None
    _align: bool = True
    _milling: bool = True
    _neutralise: bool = True

    def __to_dict__(self):

        return {
            "n_steps": self.n_steps,
            "step_size": self.step_size,
            "image": [
                (image.__to_dict__(), beam.__to_dict__(), detector.__to_dict__())
                for image, beam, detector in self.image
            ],
            "mill": self.mill.__to_dict__() if self.mill is not None else None,
            "_align": self._align,
            "_milling": self._milling,
            "_neutralise": self._neutralise,
        }

    @classmethod
    def __from_dict__(cls, data):

        image = [
            (
                ImageSettings.__from_dict__(image),
                BeamSettings.__from_dict__(beam),
                FibsemDetectorSettings.__from_dict__(detector),
            )
            for image, beam, detector in data["image"]
        ]
        return cls(
            n_steps=data["n_steps"],
            step_size=data["step_size"],
            image=image,
            mill=FibsemMillingStage.__from_dict__(data["mill"])
            if data["mill"] is not None
            else None,
            _align=data["_align"],
            _milling=data["_milling"],
            _neutralise=data["_neutralise"],
        )


import os
import yaml


class Experiment:
    def __init__(self, path: Path = None, name: str = "default") -> None:

        self.name: str = name
        self.path: Path = futils.make_logging_directory(path=path, name=name)
        self.log_path: Path = futils.configure_logging(
            path=self.path, log_filename="logfile"
        )

        self.settings: SalamiSettings = None
        self.positions: list[MicroscopeState] = []

    def __to_dict__(self) -> dict:

        return {
            "name": self.name,
            "path": self.path,
            "settings": self.settings.__to_dict__()
            if self.settings is not None
            else None,
            "positions": [lamella.__to_dict__() for lamella in self.positions],
        }

    def save(self) -> None:
        """Save the sample data to yaml file"""

        with open(os.path.join(self.path, f"{self.name}.yaml"), "w") as f:
            yaml.safe_dump(self.__to_dict__(), f, indent=4)

    def __repr__(self) -> str:

        return f"""Experiment: 
        Path: {self.path}
        Settings: {self.settings}
        Positions: {len(self.positions)}
        """

    @staticmethod
    def load(fname: Path) -> "Experiment":
        """Load a sample from disk."""

        # read and open existing yaml file
        path = Path(fname).with_suffix(".yaml")
        if os.path.exists(path):
            with open(path, "r") as f:
                sample_dict = yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"No file with name {path} found.")

        # create sample
        path = os.path.dirname(sample_dict["path"])
        name = sample_dict["name"]
        experiment = Experiment(path=path, name=name)

        experiment.settings = SalamiSettings.__from_dict__(sample_dict["settings"])

        return experiment
