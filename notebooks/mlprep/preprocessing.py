"""Version-compatible preprocessing factories used across the notebooks."""

from packaging.version import Version
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder


def make_target_encoder(seed: int = 7, n_splits: int = 5) -> TargetEncoder:
    """Return a shuffled cross-fitted TargetEncoder without version warnings."""
    if Version(sklearn.__version__) >= Version("1.9"):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return TargetEncoder(cv=cv)
    return TargetEncoder(cv=n_splits, shuffle=True, random_state=seed)
