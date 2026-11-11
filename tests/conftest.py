def pytest_addoption(parser):
    parser.addoption('--bundle', default=None, help='Explicit trusted local bundle')
