"""MkDocs hook: links from the docs into the rest of the repository point at GitHub.

The markdown under docs/ is also read on GitHub, where `../crates/...` or `../examples/x.py`
resolve. On the site those files are not pages, so rewrite any relative link that leaves
docs/ (or names a non-page file) to its GitHub URL.
"""

from __future__ import annotations

import os
import posixpath
import re

REPO = "https://github.com/AbiralShakya/ChewingOn_RackScaleSimulation"
LINK = re.compile(r"(\]\()([^)\s]+)(\))")
PAGE_EXT = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".html")


def on_page_markdown(markdown, page, config, files):
    src_dir = posixpath.dirname(page.file.src_uri)
    docs_dir = config["docs_dir"]

    def fix(m):
        url = m.group(2)
        if re.match(r"^[a-z]+:|^#|^/", url):
            return m.group(0)
        path, _, frag = url.partition("#")
        target = posixpath.normpath(posixpath.join(src_dir, path))
        inside = not target.startswith("..")
        if inside and target.endswith(PAGE_EXT) and os.path.exists(os.path.join(docs_dir, target)):
            return m.group(0)
        repo_path = posixpath.normpath(posixpath.join("docs", target))
        kind = "tree" if path.endswith("/") or os.path.isdir(os.path.join(docs_dir, target)) else "blob"
        return f"{m.group(1)}{REPO}/{kind}/main/{repo_path}{'#' + frag if frag else ''}{m.group(3)}"

    return LINK.sub(fix, markdown)
