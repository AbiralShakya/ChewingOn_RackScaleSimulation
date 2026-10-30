"""Build the learning page: page.html with rsmath.js and data/learn.json inlined.

    python research/learn/export.py     # data/learn.json (needs the other studies' results)
    python research/learn/build.py      # dist/scaling-lab.html and docs/viz/scaling-lab.html

``dist/scaling-lab.html`` is the page body as published (no doctype; the host
adds the document skeleton); ``docs/viz/scaling-lab.html`` is the same page
as a standalone document for the docs site.
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main() -> None:
    page = open(os.path.join(HERE, "page.html")).read()
    js = open(os.path.join(HERE, "rsmath.js")).read()
    data = json.load(open(os.path.join(HERE, "data", "learn.json")))
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    assert page.count("/*RSMATH*/") == 1 and page.count("/*DATA*/") == 1
    body = page.replace("/*RSMATH*/", js).replace("/*DATA*/", blob)
    os.makedirs(os.path.join(HERE, "dist"), exist_ok=True)
    with open(os.path.join(HERE, "dist", "scaling-lab.html"), "w") as f:
        f.write(body)
    head = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
    with open(os.path.join(ROOT, "docs", "viz", "scaling-lab.html"), "w") as f:
        f.write(head + body + "\n</html>\n")
    print(f"dist/scaling-lab.html: {len(body) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
