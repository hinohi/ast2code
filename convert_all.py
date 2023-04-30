# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from gen27 import ast27 as ast, CodeGen27 as CodeGen


def main():
    parser = argparse.ArgumentParser('code to ast to code')
    parser.add_argument('src')
    parser.add_argument('dst')
    args = parser.parse_args()

    gen = CodeGen()
    src = Path(args.src)
    dst = Path(args.dst)
    for path in src.glob('**/*.py'):
        print(path)
        root = ast.parse(path.read_text(), filename=str(path))
        source_gen = gen.generate(root, 0)
        dst_path = Path(*(dst.parts + path.parts[len(src.parts):]))
        dst_path.parent.mkdir(exist_ok=True, parents=True)
        dst_path.write_text(source_gen)


if __name__ == '__main__':
    main()
