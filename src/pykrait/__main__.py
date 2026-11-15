"""
Copyright © 2026 Phaleanopsis
"""

# src/pykrait/__main__.py
import sys

try:
    from pykrait.gui.app import AppController
except ImportError as e:
    if getattr(e, "name", None) and e.name.startswith("PySide6"):
        sys.exit(
            "The pyKrait GUI needs the optional GUI dependencies. "
            'Install them with: pip install "pykrait[gui]"'
        )
    raise

if __name__ == "__main__":
    app_controller = AppController()
    app_controller.run()
