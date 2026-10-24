"""
src/ui/__init__.py
------------------
SmartMail Local Functional Testing UI package.
"""
def create_app(*args, **kwargs):
    from src.ui.app import create_app as _create_app
    return _create_app(*args, **kwargs)

__all__ = ["create_app"]
