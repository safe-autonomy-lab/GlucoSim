"""Declared factory outcomes shared by capture, coverage, and migration checks."""
import ast
import hashlib
import inspect
import json
import textwrap


REJECTION = {
    'types': ['t2d_no_pump'],
    'exception': 'ValueError',
    'message': 'use_pump=False requires basal=0',
    'origin': 'build_patient_params.resolved_delivery_guard',
}


def expected_rejection(case, kind):
    """Only the specifically approved factory rejection is a valid declaration."""
    spec = case.get('expected_rejection')
    if spec is None:
        return None
    if spec != REJECTION:
        raise ValueError('Unsupported expected_rejection declaration')
    return dict(spec) if kind in spec['types'] else None


def factory_record_stages(case, kind):
    if expected_rejection(case, kind) is not None:
        return ('patient_rejection', 'created_rejection')
    return ('patient', 'created', 'tuned')


def rejection_digest(stage, spec):
    if stage not in ('patient_rejection', 'created_rejection') or spec != REJECTION:
        raise ValueError('Invalid rejection record')
    payload = {'version': 1, 'outcome': 'expected_rejection', 'stage': stage, 'expectation': spec}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_factory_rejection(call, owner, spec):
    """Require the exact exception at the actual construction guard, not upstream.

    Inspect the uniquely matching raise statement rather than hard-coding its
    line number. A same-message loader/calibration exception is never accepted.
    """
    if spec != REJECTION:
        raise ValueError('Unsupported expected_rejection declaration')
    if (owner.__module__ != 'glucosim.simglucose.core.parameter_builder'
            or owner.__qualname__ != 'build_patient_params'):
        raise AssertionError('Unexpected factory guard owner')
    source, first_line = inspect.getsourcelines(owner)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    raises = [node for node in ast.walk(tree)
              if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
              and isinstance(node.exc.func, ast.Name) and node.exc.func.id == 'ValueError'
              and len(node.exc.args) == 1 and isinstance(node.exc.args[0], ast.Constant)
              and node.exc.args[0].value == spec['message']]
    if len(raises) != 1:
        raise AssertionError('Expected one identifiable factory delivery guard')
    expected_line = first_line + raises[0].lineno - 1
    try:
        call()
    except ValueError as error:
        leaf = error.__traceback__
        while leaf.tb_next is not None:
            leaf = leaf.tb_next
        if (type(error) is not ValueError or str(error) != spec['message']
                or leaf.tb_frame.f_code is not owner.__code__
                or leaf.tb_lineno != expected_line):
            raise AssertionError('Exception did not match the declared factory rejection') from error
    else:
        raise AssertionError('Expected factory rejection, but construction succeeded')
