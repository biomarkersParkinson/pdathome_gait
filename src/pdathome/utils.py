from IPython import get_ipython

# Check if you're in a Jupyter Notebook
def in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other types of interactive shells
    except NameError:
        return False  # Not in an interactive environment