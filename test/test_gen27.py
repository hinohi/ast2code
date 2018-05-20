# -*- coding: utf-8 -*-
import unittest

from typed_ast import ast27


class TestCodeGen27(unittest.TestCase):

    def setUp(self):
        from gen27 import CodeGen27

        self.gen = CodeGen27()

        def regen(s):
            return self.gen.generate(ast27.parse(s), 0)

        self.regen = regen

    def test_module_docstring(self):
        cases = [
            '',
            '"""doc string"""',
            '''"""
doc string
"""''',
            'b"""a"""',
            'u"""あ"""',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_assert(self):
        cases = [
            'assert True',
            "assert True, 'assert true'",
            """def f():
    assert False, 'テスト'"""
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_assign(self):
        cases = [
            'a = 1',
            'a = b = 1',
            'a = 2  # type: int',
            'a = b = 2  # type: int',
            """def f():
    a = 1""",
            'a, b = l',
            'a, (b, c) = l',
            '[a] = b = l',
            '[a, (b, [c, (d,)])] = l',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_break_continue(self):
        cases = [
            """for i in l:
    break""",
            """for i in l:
    continue""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_aug_assign(self):
        cases = [
            'b += 1',
            'b -= 1',
            'b *= 1',
            'b /= 1',
            'b //= 1',
            'b %= 1',
            'b **= 1',
            'b >>= 1',
            'b <<= 1',
            'b &= 1',
            'b ^= 1',
            'b |= 1',
            """def f():
    a += 1""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_class_def(self):
        cases = [
            """class A:
    pass""",
            """class A(object):
    a = 1""",
            """@deco
class A(B, C):
    @deco_out
    @deco_iin
    class D:
        def f(self):
            pass""",
            '''class A:
    """
    doc string
    """''',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_delete(self):
        cases = [
            'del a',
            'del a, b',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_exec(self):
        cases = [
            "exec '1'",
            "exec 'a = 1' in globals()",
            "exec 'a = 1' in globals(), locals()",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_for(self):
        cases = [
            """for i in l:
    pass""",
            """for i, j in l:
    pass""",
            """for i, (j, k) in l:
    pass""",
            """for i in l:  # type: int
    for j in ll:  # type: long
        pass""",
            """for i in l:
    a += i
else:
    pass""",
            """for i in l:
    for j in ll:
        a += j
    else:
        a += i
else:
    pass""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_function_def(self):
        cases = [
            """def f():
    pass""",
            """def f(a):
    pass""",
            """def f(a, b):
    pass""",
            """def f(a=1):
    pass""",
            """def f(a=1, b=2):
    pass""",
            """def f(a, b, c=1, d=2):
    pass""",
            """def f(*args):
    pass""",
            """def f(a, *args):
    pass""",
            """def f(a=1, *args):
    pass""",
            """def f(a, b, c=1, d=2, *args):
    pass""",
            """def f(**kw):
    pass""",
            """def f(a, **kw):
    pass""",
            """def f(a=1, **kw):
    pass""",
            """def f(a, b, c=1, d=2, **kw):
    pass""",
            """def f(*args, **kw):
    pass""",
            """def f(a, *args, **kw):
    pass""",
            """def f(a=1, *args, **kw):
    pass""",
            """def f(a, b, c=1, d=2, *args, **kw):
    pass""",
            """def f():  # type: () -> int
    return 1""",
            """def f(
    a,  # type: int
    b,
    c,  # type: float
    *d  # type: list
):  # type: None
    pass""",
            """@deco
def f():
    pass""",
            """@outer
@inner(a, b=1)
def f():
    pass""",
            '''def f():
    """abc"""
    a = b = c = 1''',
            '''def f():
    """
    a
      b
    c
    """
    pass''',
            """def f():
    def g():
        def h():
            pass"""
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_global(self):
        cases = [
            """def f():
    global a
    a = 1""",
            """def f():
    global a, b, c
    a = b = c = 1"""
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_if(self):
        cases = [
            """if a:
    pass""",
            """if a:
    a = 1
else:
    pass""",
            """if a:
    a = 1
elif b:
    pass""",
            """if a:
    a = 1
elif b:
    b = 1
else:
    pass""",
            """if a:
    if b:
        if c:
            c = 1
    elif d:
        d = 1
elif e:
    e = 1""",
            """def f():
    if a:
        pass""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_import(self):
        cases = [
            'import a',
            'import a.b',
            'import a as aa',
            'import a.b as aa',
            'import a, b',
            'import a as aa, b',
            'import a as aa, b as bb',
            'from a import b',
            'from a import b, c',
            'from a import b as bb',
            'from a import b as bb, c',
            'from a import b, c',
            'from . import b',
            'from .a import b',
            'from a.b import c',
            'from ....a import b',
            'from ....a import b',
            'from ..a.b.c import d',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_print(self):
        cases = [
            'print',
            'print a',
            'print a,',
            'print a, b',
            'print>>sys.stderr, a',
            'print>>sys.stderr, a,',
            'print>>sys.stderr, a, b',
            """def f():
    print""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_raise(self):
        cases = [
            'raise',
            'raise TypeError',
            'raise Exception(a)',
            'raise Exception, a',
            'raise Exception, a, tb',
            """def f():
    raise ValueError""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_return(self):
        cases = [
            """def f():
    return""",
            """def f():
    return a""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_try(self):
        cases = [
            """try:
    a = 1
except:
    b = 2""",
            """try:
    pass
except ValueError:
    pass""",
            """try:
    pass
except ValueError as e:
    pass""",
            """try:
    pass
except (ValueError, TypeError) as e:
    pass""",
            """try:
    pass
except ValueError:
    pass
except TypeError:
    pass""",
            """try:
    pass
finally:
    pass""",
            """try:
    pass
except Exception:
    pass
finally:
    pass""",
            """try:
    try:
        a = 1
    except:
        b = 2
    finally:
        c = 3
except:
    pass
finally:
    pass""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_while(self):
        cases = [
            """while True:
    pass""",
            """while True:
    break
else:
    pass""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_with(self):
        cases = [
            """with aa:
    pass""",
            """with open('a.txt') as f:
    pass""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

        neq_cases = [
            (
                """with aa as a, bb as b:
    pass""",
                """with aa as a:
    with bb as b:
        pass"""),
        ]
        for origin, exact in neq_cases:
            self.assertEqual(exact, self.regen(origin))

    def test_call(self):
        cases = [
            'f()',
            'f(1)',
            'f(1, 2)',
            'f(1, 2, 3)',
            'f(a=1)',
            'f(a=1, b=2)',
            'f(*l)',
            'f(**k)',
            'f(1, 2, a=1, b=2)',
            'f(1, a=2, *l, **k)',
            '(f + h)()',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_compare(self):
        cases = [
            '1 < 2 > 3 == 4 != 5',
            'a <= b >= c',
            'a is None',
            'a is not None',
            'a in l not in ll',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_dict(self):
        cases = [
            '{}',
            '{1: 10}',
            """{
    1: 10,
    2: 20,
}""",
            """def f():
    a = {
        b: c,
        d: e,
    }""",
            '{i: i * i for i in a}',
            '{i: i * i for i in a if i}',
            '{i: i * i for i in a if i if i < 10}',
            '{i: i * j for i in a if i if i < 10 for j in b if j * 10}',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_generator_exp(self):
        cases = [
            '(i * j for i in l)',
            '(i * j for i in l if i > 10)',
            '(i * j for i in l for j in ll)',
            '(i * j for i in l if i < 2 for j in ll if j if j < 10)',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_lambda(self):
        cases = [
            'lambda: 1',
            'lambda x: x + 1',
            'lambda x, y: x * y',
            'lambda x, y=1: x / y',
            'lambda x, y=1, *z: x / y + sum(z)',
            'lambda x: lambda y: x ^ y',
            '(lambda x: x) + f',
            'f | (lambda x: x)',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_list(self):
        cases = [
            '[]',
            '[1]',
            """[
    1,
    2,
]""",
            '[a, b, c] = l',
            '[i for i in l]',
            '[i * j for i in l if i for j in ll if j]',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_repr(self):
        cases = [
            '`1`',
            '`1 + 2`',
            "`1 + 2` + '3'",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_set(self):
        cases = [
            '{1}',
            '{1, 2, 3}',
            """def f():
    a = {
        1,
        2,
        3,
        4,
    }""",
            '{i ** 2 for i in a}',
            '{i ** 2 for i in a if i < 10}',
            '{i ** 2 for i in a if i < 10 if i > 2}',
            '{i * j for i in a if i < 10 if i > 2 for j in b}',
            '{i * j for i in a if i < 10 if i > 2 for j in b if j | 3}',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_str(self):
        cases = [
            "s = 'a'",
            "s = 'あ'",
            's = "\'"',
            "b = b'a'",
            "b = b'\\x00'",
            "b = u'あ'",
            "b = u'a'",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_subscript(self):
        cases = [
            'l[0]',
            'l[1 + 2]',
            'l[...]',
            'l[:]',
            'l[1:]',
            'l[:-1]',
            'l[1:-1]',
            'l[::]',
            'l[1::]',
            'l[:2:]',
            'l[::3]',
            'l[1:2:]',
            'l[1::3]',
            'l[:2:3]',
            'l[1:2:3]',
            'l[:, :]',
            'l[:, ...]',
            '(a + b)[0]',
            'a[1][2]',
            'a.b[1]',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_tuple(self):
        cases = [
            '()',
            '(1,)',
            '(1, 2, 3)',
            """(
    1,
    2,
    3,
    4,
)""",
            'a, = l',
            'a, b = l',
            'a, b, c, d = l',
            'a = (1, 2)',
            'a, (b, c) = l',
            '[a, (b, c)] = l',
            """def f(a=(1, 2)):
    return 3, 4""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_bin_op(self):
        cases = [
            '1 + 2 - 3 * 4 / 5 % 6',
            '(1 + 2) * (3 // 4)',
            '1 ** 2 ** 3 ** 4',
            '1 ** (2 ** 3) ** 4',
            '1 + 2 ** 3 & 4 | 5 ^ 6',
            '(1 + 2) ** 3 & (4 | 5) ^ 6',
            '1 * 2 << 3 | 5 >> 6 ** 7 << 8',
            '1 * ((2 << 3 | 5) >> 6 ** 7 << 8)',
            'a < 2 + 3',
            '(a < 2) + 3',
            '(a < 2) + 3',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_bool_op(self):
        cases = [
            'a and b',
            'a and b and c',
            'a and b or c',
            'a and (b or c)',
            'a or b or c',
            'a or 1 + 2',
            '(a or 1) + 2',
            'a or b if c else d',
            'a or (b if c else d)',
            'a < b and b < c',
            'a < (b and c) < d',
            'a < b or c < d',
            'a < (b or c) < d',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_unary_op(self):
        cases = [
            '+a',
            '-a',
            'not a',
            '~a',
            '+a + +b',
            '+a * -b',
            '-~-b',
            '-(not b)',
            '~(not +-b)',
            '-a ** b',
            '(-a) ** b',
            'a ** -b',
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_yield(self):
        cases = [
            """def f():
    a = (yield 1)""",
            """def f():
    a = [(yield i) for i in l]""",
        ]
        for c in cases:
            self.assertEqual(c, self.regen(c))

    def test_error(self):
        self.assertRaises(TypeError, self.gen.generate, '', 0)
