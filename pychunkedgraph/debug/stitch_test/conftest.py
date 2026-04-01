import pytest


def pytest_addoption(parser):
    parser.addoption("--e2e-mode", default="multiwave", choices=["multiwave", "wave", "single"])


@pytest.fixture(scope="session", autouse=True)
def bigtable_emulator():
    """Override parent conftest's autouse bigtable_emulator fixture.
    test_e2e.py manages its own emulator lifecycle."""
    yield
