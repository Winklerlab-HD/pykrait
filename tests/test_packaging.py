import tomllib
from pathlib import Path

import pytest

requirements = pytest.importorskip("packaging.requirements")

PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"


def test_dependencies_are_valid_requirements():
    project = tomllib.loads(PYPROJECT.read_text())["project"]
    dependencies = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        dependencies += extra
    for dependency in dependencies:
        # raises InvalidRequirement for invalid syntax, e.g. "not" in a marker
        requirements.Requirement(dependency)
