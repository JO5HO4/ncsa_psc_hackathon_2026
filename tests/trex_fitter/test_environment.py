"""Host environment contract, independent of analysis execution."""

import tomllib
from pathlib import Path


def test_uv_dependencies_keep_heavy_tools_optional():
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    dependencies = " ".join(project["project"]["dependencies"])
    extras = project["project"]["optional-dependencies"]
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    assert "pydantic" in dependencies
    assert "coffea" not in dependencies
    assert any(item.startswith("coffea==") for item in extras["coffea"])
    assert any(item.startswith("atlas-schema==") for item in extras["atlas-schema"])
    assert any(item.startswith("uproot") for item in extras["inputs"])
    assert project["tool"]["uv"]["package"] is False
