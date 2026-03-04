def test_funasr_in_requirements():
    req = open("requirements.txt").read()
    assert "funasr" in req

def test_modelscope_in_requirements():
    req = open("requirements.txt").read()
    assert "modelscope" in req
