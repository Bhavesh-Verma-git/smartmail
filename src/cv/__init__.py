"""
src/cv
------
CV tailoring and generation package.
"""
from src.cv.models import ProjectSelectionResult, TailoredCVResult
from src.cv.selector import ProjectSelector, select_projects
from src.cv.generator import CVGenerator, generate_tailored_cv

__all__ = [
    "ProjectSelectionResult",
    "TailoredCVResult",
    "ProjectSelector",
    "select_projects",
    "CVGenerator",
    "generate_tailored_cv",
]
