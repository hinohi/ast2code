# -*- coding: utf-8 -*-
import sys
import argparse


def main():
    parser = argparse.ArgumentParser('code to ast to code')
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('--outfile', '-o',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('--python', '-p',
                        choices=['27', '3'],
                        default='27')
    args = parser.parse_args()

    if args.python == '27':
        from gen27 import ast27 as ast, CodeGen27 as CodeGen
    else:
        raise NotImplemented
    source = args.infile.read()
    root = ast.parse(source, filename=args.infile.name)
    gen = CodeGen()
    source_gen = gen.generate(root, 0)
    args.outfile.write(source_gen)


if __name__ == '__main__':
    main()
