# Thin copy/launcher of 014_time_features_seedbag5.py for counted valid scoring after the dev screen.
# The implementation lives in the sibling solution file written in the previous iteration.
import os
import runpy

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    runpy.run_path(os.path.join(here, '014_time_features_seedbag5.py'), run_name='__main__')
