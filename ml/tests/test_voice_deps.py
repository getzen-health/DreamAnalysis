from pathlib import Path

_REQ = Path(__file__).resolve().parent.parent / "requirements.txt"


def test_funasr_in_requirements():
    """funasr must be listed in ml/requirements.txt."""
    content = _REQ.read_text()
    assert "funasr" in content


def test_modelscope_in_requirements():
    """modelscope must be listed in ml/requirements.txt."""
    content = _REQ.read_text()
    assert "modelscope" in content
