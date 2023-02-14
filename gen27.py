# -*- coding: utf-8 -*-
"""
Python2.7 相当
"""
from typed_ast import ast27


def _snake_case(name):
    ret = []
    for c in name:
        if c.isupper():
            ret.append('_' + c.lower())
        elif ret:
            ret.append(c)
        else:
            ret.append('_' + c)
    return ''.join(ret)


EXPR_ATOM = (
    ast27.Name, ast27.Str, ast27.Num,
    ast27.List, ast27.ListComp,
    ast27.Dict, ast27.DictComp,
    ast27.Set, ast27.SetComp,
    ast27.Tuple, ast27.GeneratorExp,
    ast27.Repr, ast27.Yield,
)
EXPR_PRIMARY = (
    ast27.Attribute,
    ast27.Subscript,
    ast27.Call,
)


def expr_order(node):
    if not isinstance(node, ast27.expr):
        raise TypeError('expr needed: %s' % node)
    if isinstance(node, EXPR_ATOM) or isinstance(node, EXPR_PRIMARY):
        return 1
    if isinstance(node, ast27.BinOp):
        if isinstance(node.op, ast27.Pow):
            return 2
    if isinstance(node, ast27.UnaryOp):
        if isinstance(node.op, (ast27.UAdd, ast27.USub, ast27.Invert)):
            return 3
    if isinstance(node, ast27.BinOp):
        if isinstance(node.op, (ast27.Mult, ast27.Div, ast27.FloorDiv, ast27.Mod)):
            return 4
        if isinstance(node.op, (ast27.Add, ast27.Sub)):
            return 5
        if isinstance(node.op, (ast27.LShift, ast27.RShift)):
            return 6
        if isinstance(node.op, ast27.BitAnd):
            return 7
        if isinstance(node.op, ast27.BitXor):
            return 8
        if isinstance(node.op, ast27.BitOr):
            return 9
        raise ValueError('Unknown BinOp: %s' % node.op)
    if isinstance(node, ast27.Compare):
        return 10
    if isinstance(node, ast27.UnaryOp):
        if isinstance(node.op, ast27.Not):
            return 11
        raise ValueError('Unknown UnaryOp: %s' % node.op)
    if isinstance(node, ast27.BoolOp):
        if isinstance(node.op, ast27.And):
            return 12
        if isinstance(node.op, ast27.Or):
            return 13
        raise ValueError('Unknown BoolOp: %s' % node.op)
    if isinstance(node, ast27.IfExp):
        return 14
    if isinstance(node, ast27.Lambda):
        return 15
    raise ValueError('Unknown Expr: %s' % node)


class Context:

    def __init__(self):
        self.stack = []

    def push(self, cls):
        self.stack.append(cls)

    def pop(self):
        self.stack.pop()

    def count(self, cls):
        return self.stack.count(cls)

    def parent(self):
        try:
            return self.stack[-2]
        except IndexError:
            return


class CodeGen27:

    def __init__(self):
        self._un_impl = set()
        self.ctx = Context()

    def indent(self, level):
        return '    ' * level

    def _default(self, node, level):
        key = node.__class__
        if key not in self._un_impl:
            self._un_impl.add(key)
            print(key)
        return ''

    def generate(self, node, level):
        if not isinstance(node, ast27.AST):
            raise TypeError('expect AST: %s' % node)
        method = 'visit' + _snake_case(node.__class__.__name__)
        self.ctx.push(node.__class__)
        s = getattr(self, method, self._default)(node, level)
        self.ctx.pop()
        return s

    def generate_paren(self, base, child, level):
        child_str = self.generate(child, level)
        if expr_order(base) < expr_order(child):
            return '(%s)' % child_str
        return child_str

    def _gen_docstring(self, node):
        if not isinstance(node, (ast27.Module, ast27.ClassDef, ast27.FunctionDef)):
            return None
        if not node.body:
            return None
        if not isinstance(node.body[0], ast27.Expr):
            return None
        if not isinstance(node.body[0].value, ast27.Str):
            return None
        s = node.body[0].value
        if s.has_b:
            b = 'b'
        else:
            b = ''
        if isinstance(s.s, bytes):
            s = s.s.decode('utf-8')
        else:
            s = s.s
            b = 'u'
        return '{b}{q}{body}{q}'.format(b=b, q='"""', body=s)

    def visit_expression(self, node, level):
        return self.generate(node.body, level)

    def visit_module(self, node, level):
        src = []
        doc = self._gen_docstring(node)
        if doc is not None:
            src.append(doc)
            body = node.body[1:]
        else:
            body = node.body
        for stmt in body:
            s = self.generate(stmt, level)
            src.append(s)
        # TODO: support type_ignores
        return '\n'.join(src)

    def visit_interactive(self, node, level):
        return self.visit_module(node, level)

    def visit_suite(self, node, level):
        return self.visit_module(node, level)

    def visit_assert(self, node, level):
        base = '{indent}assert {test}'.format(
            indent=self.indent(level),
            test=self.generate(node.test, level),
        )
        if node.msg is None:
            return base
        return '{base}, {msg}'.format(
            base=base,
            msg=self.generate(node.msg, level),
        )

    def visit_assign(self, node, level):
        targets = ' = '.join(self.generate(t, level) for t in node.targets)
        value = self.generate(node.value, level)
        if node.type_comment:
            type_comment = '  # type: ' + node.type_comment
        else:
            type_comment = ''
        return '{indent}{target} = {value}{type_comment}'.format(
            indent=self.indent(level),
            target=targets,
            value=value,
            type_comment=type_comment,
        )

    def visit_aug_assign(self, node, level):
        target = self.generate(node.target, level)
        value = self.generate(node.value, level)
        op = {
            ast27.Add: '+=',
            ast27.Sub: '-=',
            ast27.Mult: '*=',
            ast27.Div: '/=',
            ast27.FloorDiv: '//=',
            ast27.Mod: '%=',
            ast27.Pow: '**=',
            ast27.RShift: '>>=',
            ast27.LShift: '<<=',
            ast27.BitAnd: '&=',
            ast27.BitXor: '^=',
            ast27.BitOr: '|=',
        }[node.op.__class__]
        return '{indent}{target} {op} {value}'.format(
            indent=self.indent(level),
            target=target,
            op=op,
            value=value,
        )

    def visit_break(self, node, level):
        return '{indent}break'.format(indent=self.indent(level))

    def visit_class_def(self, node, level):
        src = []
        for deco in node.decorator_list:
            src.append('{indent}@{deco}'.format(
                indent=self.indent(level),
                deco=self.generate(deco, level)
            ))

        if node.bases:
            bases = '(' + ', '.join(self.generate(base, level) for base in node.bases) + ')'
        else:
            bases = ''
        class_def = '{indent}class {name}{bases}:'.format(
            indent=self.indent(level),
            name=node.name,
            bases=bases,
        )
        src.append(class_def)

        doc = self._gen_docstring(node)
        if doc is not None:
            src.append(self.indent(level + 1) + doc)
            body = node.body[1:]
        else:
            body = node.body
        for stmt in body:
            s = self.generate(stmt, level + 1)
            src.append(s)
        return '\n'.join(src)

    def visit_continue(self, node, level):
        return '{indent}continue'.format(indent=self.indent(level))

    def visit_delete(self, node, level):
        return '{indent}del {targets}'.format(
            indent=self.indent(level),
            targets=', '.join(self.generate(t, level) for t in node.targets),
        )

    def visit_exec(self, node, level):
        e = 'exec ' + self.generate(node.body, level)
        if node.globals:
            e = '{e} in {globals}'.format(
                e=e,
                globals=self.generate(node.globals, level),
            )
        if node.locals:
            assert node.globals is not None
            e = '{e}, {locals}'.format(
                e=e,
                locals=self.generate(node.locals, level)
            )
        return e

    def visit_expr(self, node, level):
        return '{indent}{value}'.format(
            indent=self.indent(level),
            value=self.generate(node.value, level),
        )

    def visit_for(self, node, level):
        src = []
        target = self.generate(node.target, level)
        it = self.generate(node.iter, level)
        if node.type_comment:
            type_comment = '  # type: ' + node.type_comment
        else:
            type_comment = ''
        src.append('{indent}for {target} in {iter}:{type_comment}'.format(
            indent=self.indent(level),
            target=target,
            iter=it,
            type_comment=type_comment,
        ))
        for stmt in node.body:
            s = self.generate(stmt, level + 1)
            src.append(s)
        if node.orelse:
            src.append('{indent}else:'.format(indent=self.indent(level)))
            for stmt in node.orelse:
                s = self.generate(stmt, level + 1)
                src.append(s)
        return '\n'.join(src)

    def visit_function_def(self, node, level):
        src = []
        for deco in node.decorator_list:
            src.append('{indent}@{deco}'.format(
                indent=self.indent(level),
                deco=self.generate(deco, level)
            ))

        func_def = '{indent}def {name}({arguments}):'.format(
            indent=self.indent(level),
            name=node.name,
            arguments=self.generate(node.args, level),
        )
        if node.type_comment:
            func_def = '{func_def}  # type: {comment}'.format(
                func_def=func_def,
                comment=node.type_comment
            )
        src.append(func_def)

        doc = self._gen_docstring(node)
        if doc is not None:
            src.append(self.indent(level + 1) + doc)
            body = node.body[1:]
        else:
            body = node.body
        for stmt in body:
            s = self.generate(stmt, level + 1)
            src.append(s)
        return '\n'.join(src)

    def visit_global(self, node, level):
        return '{indent}global {names}'.format(
            indent=self.indent(level),
            names=', '.join(node.names),
        )

    def visit_if(self, node, level):
        src = []
        while True:
            if src:
                if_keyword = 'elif'
            else:
                if_keyword = 'if'
            src.append('{indent}{if_keyword} {test}:'.format(
                indent=self.indent(level),
                if_keyword=if_keyword,
                test=self.generate(node.test, level),
            ))
            for stmt in node.body:
                src.append(self.generate(stmt, level + 1))
            if len(node.orelse) == 1 and \
                    isinstance(node.orelse[0], ast27.If) and \
                    (node.col_offset == node.orelse[0].col_offset or
                     node.col_offset + 5 == node.orelse[0].col_offset):
                node = node.orelse[0]
                continue
            elif node.orelse:
                src.append('{indent}else:'.format(indent=self.indent(level)))
                for stmt in node.orelse:
                    src.append(self.generate(stmt, level + 1))
            break
        return '\n'.join(src)

    def visit_import(self, node, level):
        return '{indent}import {names}'.format(
            indent=self.indent(level),
            names=', '.join(self.generate(n, level) for n in node.names),
        )

    def visit_import_from(self, node, level):
        return '{indent}from {level}{module} import {names}'.format(
            indent=self.indent(level),
            level='.' * node.level,
            module=node.module or '',
            names=', '.join(self.generate(n, level) for n in node.names),
        )

    def visit_pass(self, node, level):
        return '{indent}pass'.format(indent=self.indent(level))

    def visit_print(self, node, level):
        if node.dest:
            keyword = 'print>>{dest},'.format(dest=self.generate(node.dest, level))
        else:
            keyword = 'print'
        values = ', '.join(self.generate(n, level) for n in node.values)
        if not node.nl:
            values += ','
        if values:
            return '{indent}{keyword} {values}'.format(
                indent=self.indent(level),
                keyword=keyword,
                values=values,
            )
        else:
            return '{indent}{keyword}'.format(
                indent=self.indent(level),
                keyword=keyword,
            )

    def visit_raise(self, node, level):
        src = []
        if node.type:
            src.append(self.generate(node.type, level))
        if node.inst:
            src.append(self.generate(node.inst, level))
        if node.tback:
            src.append(self.generate(node.tback, level))
        if src:
            return '{indent}raise {expr}'.format(
                indent=self.indent(level),
                expr=', '.join(src),
            )
        return '{indent}raise'.format(indent=self.indent(level))

    def visit_return(self, node, level):
        if node.value:
            return '{indent}return {values}'.format(
                indent=self.indent(level),
                values=self.generate(node.value, level),
            )
        else:
            return '{indent}return'.format(
                indent=self.indent(level),
            )

    def visit_try_except(self, node, level):
        src = ['{indent}try:'.format(indent=self.indent(level))]
        for stmt in node.body:
            src.append(self.generate(stmt, level + 1))
        for handler in node.handlers:
            except_ = '{indent}except'.format(indent=self.indent(level))
            if handler.type:
                except_ += ' ' + self.generate(handler.type, level)
            if handler.name:
                except_ += ' as ' + self.generate(handler.name, level)
            src.append(except_ + ':')
            for stmt in handler.body:
                src.append(self.generate(stmt, level + 1))
        for stmt in node.orelse:
            src.append(self.generate(stmt, level + 1))
        return '\n'.join(src)

    def visit_try_finally(self, node, level):
        src = []
        if len(node.body) == 1 and \
                isinstance(node.body[0], ast27.TryExcept) and \
                node.col_offset == node.body[0].col_offset:
            src.append(self.generate(node.body[0], level))
        else:
            src.append('{indent}try:'.format(indent=self.indent(level)))
            for stmt in node.body:
                src.append(self.generate(stmt, level + 1))
        src.append('{indent}finally:'.format(indent=self.indent(level)))
        for stmt in node.finalbody:
            src.append(self.generate(stmt, level + 1))
        return '\n'.join(src)

    def visit_while(self, node, level):
        src = [
            '{indent}while {test}:'.format(
                indent=self.indent(level),
                test=self.generate(node.test, level),
            )
        ]
        for stmt in node.body:
            src.append(self.generate(stmt, level + 1))
        if node.orelse:
            src.append('{indent}else:'.format(indent=self.indent(level)))
            for stmt in node.orelse:
                src.append(self.generate(stmt, level + 1))
        return '\n'.join(src)

    def visit_with(self, node, level):
        src = []
        context = self.generate(node.context_expr, level)
        var = comment = ''
        if node.optional_vars:
            var = ' as ' + self.generate(node.optional_vars, level)
        if node.type_comment:
            comment = '  $ type: ' + node.type_comment
        src.append('{indent}with {context}{var}:{comment}'.format(
            indent=self.indent(level),
            context=context,
            var=var,
            comment=comment,
        ))
        for stmt in node.body:
            src.append(self.generate(stmt, level + 1))
        return '\n'.join(src)

    def visit_alias(self, node, level):
        if node.asname:
            return '{name} as {asname}'.format(
                name=node.name,
                asname=node.asname,
            )
        else:
            return node.name

    def visit_arguments(self, node, level):
        args = []
        for arg in node.args:
            args.append(self.generate(arg, level))
        i = len(node.defaults) - len(node.args)
        for default in node.defaults:
            args[i] = '{arg}={default}'.format(
                arg=args[i],
                default=self.generate(default, level),
            )
            i += 1

        # allow tail comma
        args = [[a, True] for a in args]
        if node.vararg:
            args.append(['*{vararg}'.format(vararg=node.vararg), False])
        if node.kwarg:
            if args:
                args[-1][1] = True
            args.append(['**{kwarg}'.format(kwarg=node.kwarg), False])

        if node.type_comments:
            arg_list = []
            for (arg, comma), comment in zip(args, node.type_comments):
                if comma:
                    comma = ','
                else:
                    comma = ''
                arg = '{indent}{arg}{comma}'.format(
                    indent=self.indent(level + 1),
                    arg=arg,
                    comma=comma,
                )
                if comment:
                    arg_list.append('{arg}  # type: {comment}'.format(
                        arg=arg,
                        comment=comment,
                    ))
                else:
                    arg_list.append(arg)
            arguments = '\n'.join([''] + arg_list + [''])
        else:
            arguments = ', '.join(a for a, _ in args)
        return arguments

    def visit_comprehension(self, node, level):
        ifs = [' if {test}'.format(test=self.generate(test, level))
               for test in node.ifs]
        return 'for {target} in {iter}{ifs}'.format(
            target=self.generate(node.target, level),
            iter=self.generate(node.iter, level),
            ifs=''.join(ifs),
        )

    def visit_keyword(self, node, level):
        return '{arg}={value}'.format(
            arg=node.arg,
            value=self.generate(node.value, level),
        )

    def visit_attribute(self, node, level):
        return '{value}.{attr}'.format(
            value=self.generate(node.value, level),
            attr=node.attr,
        )

    def visit_bin_op(self, node, level):
        ops = {
            ast27.Add: '+',
            ast27.Sub: '-',
            ast27.Mult: '*',
            ast27.Div: '/',
            ast27.FloorDiv: '//',
            ast27.Mod: '%',
            ast27.Pow: '**',
            ast27.BitAnd: '&',
            ast27.BitOr: '|',
            ast27.BitXor: '^',
            ast27.LShift: '<<',
            ast27.RShift: '>>',
        }
        right_bounds = isinstance(node.op, ast27.Pow)

        op_s = ops[node.op.__class__]
        left_s = self.generate(node.left, level)
        right_s = self.generate(node.right, level)
        node_order = expr_order(node)
        left_order = expr_order(node.left)
        right_order = expr_order(node.right)
        if node_order < left_order:
            left_s = '(%s)' % left_s
        elif node_order == left_order and right_bounds:
            left_s = '(%s)' % left_s
        if node_order < right_order - right_bounds:
            right_s = '(%s)' % right_s
        elif node_order == right_order and not right_bounds:
            right_s = '(%s)' % right_s

        return '{left} {op} {right}'.format(
            left=left_s,
            op=op_s,
            right=right_s,
        )

    def visit_bool_op(self, node, level):
        op = {
            ast27.And: ' and ',
            ast27.Or: ' or ',
        }[node.op.__class__]
        values = [self.generate_paren(node, value, level)
                  for value in node.values]
        return op.join(values)

    def visit_call(self, node, level):
        func = self.generate_paren(node, node.func, level)
        args = [self.generate(arg, level) for arg in node.args]
        args.extend(self.generate(keyword, level) for keyword in node.keywords)
        if node.starargs:
            args.append('*' + self.generate(node.starargs, level))
        if node.kwargs:
            args.append('**' + self.generate(node.kwargs, level))
        return '{func}({args})'.format(
            func=func,
            args=', '.join(args)
        )

    def visit_compare(self, node, level):
        ops = {
            ast27.Eq: '==',
            ast27.Gt: '>',
            ast27.GtE: '>=',
            ast27.In: 'in',
            ast27.Is: 'is',
            ast27.IsNot: 'is not',
            ast27.Lt: '<',
            ast27.LtE: '<=',
            ast27.NotEq: '!=',
            ast27.NotIn: 'not in',
        }
        src = [self.generate_paren(node, node.left, level)]
        assert len(node.ops) == len(node.comparators)
        for op, right in zip(node.ops, node.comparators):
            src.append(ops[op.__class__])
            src.append(self.generate_paren(node, right, level))
        return ' '.join(src)

    def visit_dict(self, node, level):
        assert len(node.keys) == len(node.values)
        num = len(node.keys)
        if num <= 1:
            body = ', '.join('{key}: {value}'.format(
                key=self.generate(key, level),
                value=self.generate(value, level),
            ) for key, value in zip(node.keys, node.values))
            return '{' + body + '}'
        body = ['{']
        for key, value in zip(node.keys, node.values):
            body.append('{indent}{key}: {value},'.format(
                indent=self.indent(level + 1),
                key=self.generate(key, level + 1),
                value=self.generate(value, level + 1),
            ))
        body.append('{indent}}}'.format(indent=self.indent(level)))
        return '\n'.join(body)

    def visit_dict_comp(self, node, level):
        return '{{{key}: {value} {generators}}}'.format(
            key=self.generate(node.key, level),
            value=self.generate(node.value, level),
            generators=' '.join(self.generate(g, level) for g in node.generators),
        )

    def visit_generator_exp(self, node, level):
        return '({elt} {generators})'.format(
            elt=self.generate(node.elt, level),
            generators=' '.join(self.generate(g, level) for g in node.generators),
        )

    def visit_if_exp(self, node, level):
        return '{body} if {test} else {orelse}'.format(
            body=self.generate_paren(node, node.body, level),
            test=self.generate_paren(node, node.test, level),
            orelse=self.generate_paren(node, node.orelse, level),
        )

    def visit_lambda(self, node, level):
        args = self.generate(node.args, level)
        if args:
            args = ' ' + args
        return 'lambda{args}: {body}'.format(
            args=args,
            body=self.generate(node.body, level),
        )

    def visit_list(self, node, level):
        num = len(node.elts)
        if num <= 1 or isinstance(node.ctx, ast27.Store):
            body = ', '.join(self.generate(elt, level) for elt in node.elts)
            return '[' + body + ']'
        body = ['[']
        for elt in node.elts:
            body.append('{indent}{elt},'.format(
                indent=self.indent(level + 1),
                elt=self.generate(elt, level + 1),
            ))
        body.append('{indent}]'.format(indent=self.indent(level)))
        return '\n'.join(body)

    def visit_list_comp(self, node, level):
        return '[{elt} {generators}]'.format(
            elt=self.generate(node.elt, level),
            generators=' '.join(self.generate(g, level) for g in node.generators),
        )

    def visit_name(self, node, level):
        return node.id

    def visit_num(self, node, level):
        return str(node.n)

    def visit_repr(self, node, level):
        return '`{value}`'.format(value=self.generate(node.value, level))

    def visit_set(self, node, level):
        num = len(node.elts)
        assert num >= 1
        if num <= 3:
            body = ', '.join(self.generate(elt, level) for elt in node.elts)
            return '{' + body + '}'
        body = ['{']
        for elt in node.elts:
            body.append('{indent}{elt},'.format(
                indent=self.indent(level + 1),
                elt=self.generate(elt, level + 1),
            ))
        body.append('{indent}}}'.format(indent=self.indent(level)))
        return '\n'.join(body)

    def visit_set_comp(self, node, level):
        return '{{{elt} {generators}}}'.format(
            elt=self.generate(node.elt, level),
            generators=' '.join(self.generate(g, level) for g in node.generators),
        )

    def visit_str(self, node, level):
        if node.has_b:
            return repr(node.s)
        elif isinstance(node.s, bytes):
            return repr(node.s.decode('utf-8'))
        else:
            return 'u' + repr(node.s)

    def visit_subscript(self, node, level):
        return '{value}[{slice}]'.format(
            value=self.generate_paren(node, node.value, level),
            slice=self.generate(node.slice, level),
        )

    def visit_tuple(self, node, level):
        num = len(node.elts)
        with_paren = not isinstance(node.ctx, ast27.Store)
        with_paren = with_paren or self.ctx.count(ast27.Tuple) >= 2
        with_paren = with_paren or self.ctx.count(ast27.List) >= 1
        if self.ctx.parent() is ast27.Return:
            with_paren = False
        if num == 0:
            return '()'
        elif num == 1:
            elt = self.generate(node.elts[0], level) + ','
            if with_paren:
                return '(' + elt + ')'
            else:
                return elt
        elif num <= 3 or not with_paren:
            elts = ', '.join(self.generate(elt, level) for elt in node.elts)
            if with_paren:
                return '(' + elts + ')'
            else:
                return elts
        body = ['(']
        for elt in node.elts:
            body.append('{indent}{elt},'.format(
                indent=self.indent(level + 1),
                elt=self.generate(elt, level + 1),
            ))
        body.append('{indent})'.format(indent=self.indent(level)))
        return '\n'.join(body)

    def visit_unary_op(self, node, level):
        op = {
            ast27.Invert: '~',
            ast27.Not: 'not ',
            ast27.UAdd: '+',
            ast27.USub: '-',
        }[node.op.__class__]
        return '{op}{operand}'.format(
            op=op,
            operand=self.generate_paren(node, node.operand, level)
        )

    def visit_yield(self, node, level):
        return '(yield {value})'.format(value=self.generate(node.value, level))

    def visit_ellipsis(self, node, level):
        return '...'

    def visit_ext_slice(self, node, level):
        return ', '.join(self.generate(s, level) for s in node.dims)

    def visit_index(self, node, level):
        return self.generate(node.value, level)

    def visit_slice(self, node, level):
        lower = upper = step = ''
        if node.lower:
            lower = self.generate(node.lower, level)
        if node.upper:
            upper = self.generate(node.upper, level)
        if node.step:
            step = ':' + self.generate(node.step, level)
            if step == ':None':
                step = ':'
        s = '{lower}:{upper}'.format(lower=lower, upper=upper)
        return s + step
