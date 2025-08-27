try:
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Location: {torch.__file__}")
    import sys
    print(f"\nPython Executable: {sys.executable}")
except ImportError:
    print("PyTorch is not installed.")
except Exception as e:
    print(f"An error occurred: {e}")
