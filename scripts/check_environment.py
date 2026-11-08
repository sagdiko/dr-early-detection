import importlib
packages = ["torch", "torchvision", "timm", "cv2", "albumentations", "sklearn", "scipy"]
for package in packages:
    try:
        module = importlib.import_module(package)
        print(f"{package}: {getattr(module, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{package}: MISSING ({exc})")
