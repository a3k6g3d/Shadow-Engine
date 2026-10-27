"""Import this before transformers in the reference scripts.  The machine's global torchaudio does not match its torch
(its DLL fails to load) and transformers imports it at module level for audio helpers the text model never uses, so an
empty stand-in is registered first.  Nothing here touches the global install."""
import importlib.machinery
import sys
import types

for _name in ("torchaudio", "torchvision"):
    if _name not in sys.modules:
        try:
            __import__(_name)
        except Exception:
            sys.modules.pop(_name, None)
            for _k in [k for k in sys.modules if k.startswith(_name + ".")]:
                sys.modules.pop(_k, None)
            _m = types.ModuleType(_name)
            _m.__spec__ = importlib.machinery.ModuleSpec(_name, None)
            _m.__path__ = []
            sys.modules[_name] = _m
