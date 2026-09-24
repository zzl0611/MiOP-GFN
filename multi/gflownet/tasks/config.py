from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SEHTaskConfig:
    pass  # SEH just uses a temperature conditional


@dataclass
class SEHMOOTaskConfig:
    """Config for the SEHMOOTask"""
    use_steer_thermometer: bool = False
    preference_type: Optional[str] = "dirichlet"
    focus_type: Optional[str] = None
    focus_dirs_listed: Optional[List[List[float]]] = None
    focus_cosim: float = 0.0
    focus_limit_coef: float = 1.0
    focus_model_training_limits: Optional[Tuple[float, float]] = None
    focus_model_state_space_res: Optional[int] = None
    max_train_it: Optional[int] = None
    n_valid: int = 15
    n_valid_repeats: int = 128
    objectives: List[str] = field(default_factory=lambda: ["seh", "qed", "sa", "mw"])




@dataclass
class QM9TaskConfig:
    h5_path: str = "./data/qm9/qm9.h5"
    model_path: str = "./data/qm9/qm9_model.pt"


@dataclass
class TasksConfig:
    qm9: QM9TaskConfig = QM9TaskConfig()
    seh: SEHTaskConfig = SEHTaskConfig()
    seh_moo: SEHMOOTaskConfig = SEHMOOTaskConfig()
    