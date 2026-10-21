"""Shared test setup."""

import sys

import pytest


@pytest.fixture(autouse=True, scope="session")
def _ignore_outside_quit_requests():
    """Keep a macOS "quit" request from ending the test run early.

    Every Tk window opened by Python on macOS belongs to one "Python" app. When anything asks that app to
    quit (another process, or the Dock), Tk's default handler exits the interpreter with status 0, so pytest
    stops partway through and still reports success. The tests never need that handler, so each Tk
    interpreter made during the run ignores the request instead.
    """
    if sys.platform != "darwin":
        yield
        return
    try:
        import tkinter
    except ImportError:
        yield
        return
    real_init = tkinter.Tk.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        try:
            self.tk.eval("proc ::tk::mac::Quit {args} {}")
        except tkinter.TclError:
            pass

    tkinter.Tk.__init__ = init
    yield
    tkinter.Tk.__init__ = real_init
