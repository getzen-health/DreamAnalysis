from pathlib import Path

_ML_DIR = Path(__file__).resolve().parent.parent
_REQ = _ML_DIR / "requirements.txt"
_REQ_DEPLOY = _ML_DIR / "requirements.deploy.txt"


def test_funasr_in_requirements():
    """funasr must be listed in ml/requirements.txt."""
    content = _REQ.read_text()
    assert "funasr" in content


def test_modelscope_in_requirements():
    """modelscope must be listed in ml/requirements.txt."""
    content = _REQ.read_text()
    assert "modelscope" in content


def test_librosa_in_requirements():
    """librosa must be in both requirements.txt and requirements.deploy.txt."""
    assert "librosa" in _REQ.read_text()
    assert "librosa" in _REQ_DEPLOY.read_text()


def test_soundfile_in_requirements():
    """soundfile must be in both requirements.txt and requirements.deploy.txt."""
    assert "soundfile" in _REQ.read_text()
    assert "soundfile" in _REQ_DEPLOY.read_text()


def test_librosa_importable():
    """librosa must be importable in the current environment."""
    import librosa  # noqa: F401


def test_soundfile_importable():
    """soundfile must be importable in the current environment."""
    import soundfile  # noqa: F401
