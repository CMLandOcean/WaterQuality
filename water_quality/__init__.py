# __all__ = ["atmosphere","data","helper","model","surface","water"]

import os
import importlib

subdirectories = [name for name in os.listdir(os.path.dirname(__file__))
                  if os.path.isdir(os.path.join(os.path.dirname(__file__), name))]

for subdirectory in subdirectories:
    package_path = f"{__name__}.{subdirectory}"
    package = importlib.import_module(package_path)
    globals().update(vars(package))