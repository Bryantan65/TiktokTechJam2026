"""Safety tests for agent tools."""
import ast
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_write_solution_rejects_path_traversal():
    from agent.tools import write_solution
    result = json.loads(write_solution('../harness/evil.py', 'print("pwned")'))
    assert 'error' in result
    assert 'solutions/' in result['error'] or 'NNN_name.py' in result['error']


def test_write_solution_rejects_bad_filename():
    from agent.tools import write_solution
    result = json.loads(write_solution('evil.py', 'print("hi")'))
    assert 'error' in result


def test_write_solution_rejects_syntax_error():
    from agent.tools import write_solution
    result = json.loads(write_solution('002_test.py', 'def f(\n'))
    assert 'error' in result
    assert 'SyntaxError' in result['error']


def test_write_solution_accepts_valid():
    from agent.tools import write_solution
    code = 'print("hello")\n'
    result = json.loads(write_solution('099_test_valid.py', code))
    assert result['status'] == 'ok'
    # clean up
    path = os.path.join(os.path.dirname(__file__), '..', 'solutions', '099_test_valid.py')
    if os.path.exists(path):
        os.remove(path)


def test_read_solution_rejects_outside_solutions():
    from agent.tools import read_solution
    result = json.loads(read_solution('../harness/run.py'))
    assert 'error' in result


if __name__ == '__main__':
    test_write_solution_rejects_path_traversal()
    test_write_solution_rejects_bad_filename()
    test_write_solution_rejects_syntax_error()
    test_write_solution_accepts_valid()
    test_read_solution_rejects_outside_solutions()
    print('All tool safety tests passed.')
