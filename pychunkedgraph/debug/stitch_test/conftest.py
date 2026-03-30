import pytest


@pytest.fixture(scope="session", autouse=True)
def bigtable_emulator():
    """Override parent conftest's autouse bigtable_emulator fixture.
    test_e2e.py manages its own emulator lifecycle."""
    yield
