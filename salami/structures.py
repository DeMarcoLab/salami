from dataclasses import dataclass


@dataclass
class SalamiSettings:
    n_steps: int
    step_size: float
    _align: bool = True
    _milling: bool = True
    _neutralise: bool = True
