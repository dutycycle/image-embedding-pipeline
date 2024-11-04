def pytest_addoption(parser):
    parser.addoption(
        "--use-real-clip",
        action="store_true",
        default=False,
        help="Use real CLIP model instead of mock",
    )
