from dataclasses import dataclass
from fibsem.structures import FibsemMillingSettings, FibsemPatternSettings

@dataclass
class SalamiSettings:
    n_steps: int
    step_size: float
    milling: FibsemMillingSettings = None
    pattern: FibsemPatternSettings = None
    _align: bool = True
    _milling: bool = True
    _neutralise: bool = True
