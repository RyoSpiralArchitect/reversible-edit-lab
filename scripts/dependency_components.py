"""Experimental, pinned-GPT-2 dependency boundaries and author-code adapters.

This is not a cache for arbitrary Transformer architectures. The caller freezes
tokenization, configuration, contexts, covariance, code, device, seed and eval
mode, and binds them into the contract. Model tensors are hashed at lookup time.
"""
import ast
import copy
import re


def prefix_dependency(name, layer):
    """Whether a GPT-2 tensor can affect the input of layer's MLP c_proj.

    Includes nonpersistent attention buffers when called on named_buffers().
    Unknown tensor names fail closed instead of silently excluding them.
    """
    if name.startswith(('transformer.wte.', 'transformer.wpe.')):
        return True
    if name.startswith(('transformer.ln_f.', 'lm_head.')):
        return False
    match = re.fullmatch(r'transformer\.h\.(\d+)\.(.+)', name)
    if not match:
        raise ValueError('unknown GPT-2 tensor: ' + name)
    index, suffix = int(match[1]), match[2]
    if not suffix.startswith(('ln_1.', 'attn.', 'ln_2.', 'mlp.c_fc.', 'mlp.c_proj.')):
        raise ValueError('unknown GPT-2 block tensor: ' + name)
    return index < layer or (index == layer and not suffix.startswith('mlp.c_proj.'))


def _function(source, name):
    matches = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError('expected one author function: ' + name)
    return copy.deepcopy(matches[0])


def adapt_rome(source, namespace, target_call):
    """Keep the author's arithmetic, expose target *before* factor division.

    The returned v factor is never treated as the latent target. Current output
    subtraction and dot-product normalization always run on the live model.
    """
    node = _function(source, 'compute_v')
    boundary = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'target']
    if len(boundary) != 1:
        raise ValueError('author target boundary changed')
    index = boundary[0]
    inner = ast.parse('def _optimize_target():\n    pass').body[0]
    inner.body = node.body[:index + 1] + [ast.Return(value=ast.Name(id='target', ctx=ast.Load()))]
    suffix = node.body[index + 1:]
    # Only diagnostic printing references the now-inner target_init variable.
    suffix = [n for n in suffix if not (isinstance(n, ast.Expr)
              and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
              and n.value.func.id == 'print' and any(isinstance(x, ast.Name)
              and x.id == 'target_init' for x in ast.walk(n)))]
    node.body = [inner] + ast.parse('target = _target_call(layer, _optimize_target)').body + suffix
    env = dict(namespace, _target_call=target_call)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<pinned-rome-target-adapter>', 'exec'), env)
    return env['compute_v']


def adapt_memit(source, namespace, compute_z, compute_ks, solve_call):
    """Expose the key-side linear solve without changing its expression/order."""
    node = _function(source, 'execute_memit')
    class SolveAdapter(ast.NodeTransformer):
        count = 0
        def visit_Call(self, call):
            if ast.unparse(call.func) == 'torch.linalg.solve':
                self.count += 1
                wrapper = ast.parse('_solve_call(layer, layer_ks, cov, lambda: PLACEHOLDER)').body[0].value
                wrapper.args[-1].body = call
                return ast.copy_location(wrapper, call)
            return self.generic_visit(call)
    adapter = SolveAdapter()
    node = adapter.visit(node)
    if adapter.count != 1:
        raise ValueError('author solve boundary changed')
    env = dict(namespace, compute_z=compute_z, compute_ks=compute_ks, _solve_call=solve_call)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 '<pinned-memit-solve-adapter>', 'exec'), env)
    return env['execute_memit']
