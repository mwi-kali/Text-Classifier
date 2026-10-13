from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("text-classifier")
except PackageNotFoundError:  #
    __version__ = "2.0.0"
