import pygem


def test_version_string():
    # simple test to check that the verion number is available
    assert type(pygem.__version__) == str
