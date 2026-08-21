from pathlib import Path
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_v030_version_contract() -> None:
    pyproject = tomllib.loads(
        (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (REPOSITORY_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert pyproject["project"]["version"] == "0.3.0"
    assert "`v0.3.0`" in readme
    assert "## v0.3.0" in changelog
