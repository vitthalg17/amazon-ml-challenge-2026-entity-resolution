"""Import-time compatibility shims (imported by config.py, i.e. before any other module).

Windows Smart App Control / App Control policies can refuse to load individual compiled
extension files. scipy.optimize's trust-region solver (_trlib) is one of them on some
machines; scikit-learn imports scipy.optimize indirectly (sklearn -> scipy.stats ->
scipy.optimize), so one blocked file breaks `import sklearn`. The pipeline never uses that
solver, so if (and only if) it cannot be loaded, an empty stand-in module takes its place and
the blocked file is simply not loaded. Nothing changes on machines where it loads normally.
"""
import sys
import types


def _stub_blocked_trlib():
    try:
        import scipy.optimize._trlib  # noqa: F401
    except ImportError as e:
        if "_trlib" not in str(e) and "Application Control" not in str(e):
            raise
        stub = types.ModuleType("scipy.optimize._trlib")

        def get_trlib_quadratic_subproblem(*args, **kwargs):
            raise RuntimeError("scipy trust-region solver unavailable on this machine "
                               "(blocked by the OS application-control policy)")

        stub.get_trlib_quadratic_subproblem = get_trlib_quadratic_subproblem
        sys.modules["scipy.optimize._trlib"] = stub


_stub_blocked_trlib()
