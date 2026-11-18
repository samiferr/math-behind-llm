#!/usr/bin/env python3
"""Build the PDF and EPUB editions of the book from manuscript/*.md.

Usage:
    python3 scripts/build.py            # cover + epub + pdf
    python3 scripts/build.py epub       # one target: cover | epub | pdf
    python3 scripts/build.py --publish  # also copy outputs into editions/<edition>/

Requirements: pandoc (3.x), XeLaTeX with fontspec/titlesec/tcolorbox,
Google Chrome or Chromium (for the cover), qpdf, and the fonts
TeX Gyre Pagella, Inter, DejaVu Serif/Sans/Sans Mono.
"""
import datetime
import glob
import html
import os
import shutil
import subprocess
import sys
import uuid

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
DIST = os.path.join(ROOT, "dist")

ORDINAL_WORDS = ["Zeroth", "First", "Second", "Third", "Fourth", "Fifth", "Sixth",
                 "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh", "Twelfth"]


def ordinal_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def edition_label(n: int) -> str:
    word = ORDINAL_WORDS[n] if n < len(ORDINAL_WORDS) else ordinal_suffix(n)
    return f"{word} Edition"


def load_meta() -> dict:
    with open(os.path.join(ROOT, "metadata.yaml"), encoding="utf-8") as f:
        m = yaml.safe_load(f)
    m["edition"] = int(m["edition"])
    m["version"] = str(m["version"])
    d = m["date"]
    m["date"] = d if isinstance(d, datetime.date) else datetime.date.fromisoformat(str(d))
    m["edition_label"] = edition_label(m["edition"])
    m["edition_short"] = ordinal_suffix(m["edition"])
    m["stem"] = f"{m['slug']}_{m['edition_short']}-edition_v{m['version']}"
    m["month_year"] = m["date"].strftime("%B %Y")
    # A stable identifier per version, so e-readers treat each release as distinct.
    m["identifier"] = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL,
                                                    f"{m['slug']}/v{m['version']}"))
    return m


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kw)


def chrome() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("Chrome or Chromium is required to render the cover.")


def manuscript_files():
    files = sorted(glob.glob(os.path.join(ROOT, "manuscript", "*.md")))
    if not files:
        sys.exit("No manuscript files found.")
    return [os.path.relpath(f, ROOT) for f in files]


def history_lines(m):
    out = []
    for h in m.get("history", []):
        d = h["date"] if isinstance(h["date"], datetime.date) else datetime.date.fromisoformat(str(h["date"]))
        out.append((f"{edition_label(int(h['edition']))}, version {h['version']}",
                    d.strftime("%B %Y"), str(h.get("note", "")).strip()))
    return out


# ---------------------------------------------------------------- cover
def build_cover(m):
    os.makedirs(BUILD, exist_ok=True)
    with open(os.path.join(ROOT, "cover", "cover.html"), encoding="utf-8") as f:
        tpl = f.read()

    def fill(page_w, page_h):
        s = tpl
        for k, v in {"TITLE": m["title"], "SUBTITLE": m["subtitle"], "AUTHOR": m["author"],
                     "EDITION": m["edition_label"], "PAGE_W": page_w, "PAGE_H": page_h}.items():
            s = s.replace("{{" + k + "}}", html.escape(str(v)))
        return s

    png_html = os.path.join(BUILD, "cover-screen.html")
    pdf_html = os.path.join(BUILD, "cover-print.html")
    with open(png_html, "w", encoding="utf-8") as f:
        f.write(fill("1600px", "2400px"))
    with open(pdf_html, "w", encoding="utf-8") as f:
        f.write(fill("7in", "10in"))
    c = chrome()
    flags = ["--headless", "--no-sandbox", "--disable-gpu", "--hide-scrollbars"]
    run([c, *flags, "--window-size=1600,2400", f"--screenshot={BUILD}/cover.png", "file://" + png_html])
    run([c, *flags, "--no-pdf-header-footer", f"--print-to-pdf={BUILD}/cover.pdf", "file://" + pdf_html])


# ---------------------------------------------------------------- front matter
def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def write_titlepage_tex(m):
    e = tex_escape
    hist = "\n".join(f"\\item {e(a)} ({e(b)}){': ' + e(c) if c else ''}" for a, b, c in history_lines(m))
    tex = rf"""
\begin{{titlepage}}
\centering\sffamily
\vspace*{{1.6in}}
{{\color{{accent}}\large\bfseries\MakeUppercase{{{e(m['edition_label'])}}}\par}}
\vspace{{0.35in}}
{{\fontsize{{30}}{{36}}\selectfont\bfseries {e(m['title'])}\par}}
\vspace{{0.3in}}
{{\Large {e(m['subtitle'])}\par}}
\vfill
{{\Large\bfseries {e(m['author'])}\par}}
\vspace{{0.6in}}
\end{{titlepage}}
\thispagestyle{{empty}}
\vspace*{{\fill}}
{{\small
\textbf{{{e(m['title'])}}}\\
{e(m['subtitle'])}\par\medskip
{e(m['edition_label'])}, version {e(m['version'])}, {e(m['month_year'])}\par\medskip
Copyright \textcopyright\ {m['date'].year} {e(m['author'])}. {e(m['rights'])}\par\medskip
\textbf{{Edition history}}
\begin{{itemize}}\setlength\itemsep{{0pt}}
{hist}
\end{{itemize}}
Figures marked ``illustrative'' use realistic made-up values chosen to teach a pattern;
they are not measurements from any specific model.\par
}}
\clearpage
"""
    path = os.path.join(BUILD, "titlepage.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(tex)
    return os.path.relpath(path, ROOT)


def write_edition_md(m):
    hist = "\n".join(f"- {a} ({b}){': ' + c if c else ''}" for a, b, c in history_lines(m))
    md = f"""## {m['edition_label']} {{.unnumbered .unlisted epub:type=copyright-page}}

::: edition-page
**{m['title']}**
*{m['subtitle']}*

{m['edition_label']}, version {m['version']}, {m['month_year']}

Copyright © {m['date'].year} {m['author']}. {m['rights']}

**Edition history**

{hist}

Figures marked "illustrative" use realistic made-up values chosen to teach a pattern; they are not measurements from any specific model.

Mathematical formulas in this EPUB use MathML. Readers without MathML support may show them as plain text; the PDF edition shows every formula typeset.
:::
"""
    path = os.path.join(BUILD, "00-edition.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return os.path.relpath(path, ROOT)


# ---------------------------------------------------------------- targets
def build_epub(m):
    os.makedirs(DIST, exist_ok=True)
    if not os.path.exists(os.path.join(BUILD, "cover.png")):
        build_cover(m)
    epub_meta = {
        "title": [{"type": "main", "text": m["title"]}, {"type": "subtitle", "text": m["subtitle"]}],
        "creator": [{"role": "author", "text": m["author"]}],
        "date": m["date"].isoformat(),
        "lang": m["lang"],
        "identifier": [{"scheme": "UUID", "text": m["identifier"]}],
        "rights": f"Copyright {m['date'].year} {m['author']}. {m['rights']} {m['edition_label']}, version {m['version']}.",
        "description": m.get("description", ""),
        "subject": m.get("keywords", []),
        "cover-image": "build/cover.png",
        "css": ["styles/epub.css"],
    }
    meta_path = os.path.join(BUILD, "epub-meta.yaml")
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(epub_meta, f, allow_unicode=True, sort_keys=False)
    out = os.path.join(DIST, m["stem"] + ".epub")
    run(["pandoc", write_edition_md(m), *manuscript_files(),
         "--from=markdown", "--to=epub3", "-o", out,
         f"--metadata-file={os.path.relpath(meta_path, ROOT)}",
         "--toc", "--toc-depth=2", "--split-level=2", "--mathml",
         "--resource-path=.", "--epub-title-page=true"])
    return out


def build_pdf(m):
    os.makedirs(DIST, exist_ok=True)
    if not os.path.exists(os.path.join(BUILD, "cover.pdf")):
        build_cover(m)
    body = os.path.join(BUILD, "body.pdf")
    run(["pandoc", *manuscript_files(), "-o", body,
         "--from=markdown", "--pdf-engine=xelatex",
         "--template=templates/book.latex", "--top-level-division=part",
         "--toc", "--toc-depth=1", "--resource-path=.",
         "-H", "styles/book.tex",
         "-M", f"title={m['title']}", "-M", f"subtitle={m['subtitle']}",
         "-M", f"author={m['author']}", "-M", f"lang={m['lang']}",
         "-V", f"custom-titlepage={write_titlepage_tex(m)}",
         "-V", "documentclass=book", "-V", "classoption=oneside,openany",
         "-V", "fontsize=11pt",
         "-V", "geometry:paperwidth=7in,paperheight=10in,inner=0.8in,outer=0.8in,top=0.85in,bottom=0.85in,headsep=0.25in",
         "-V", "colorlinks=true", "-V", "linkcolor=accent", "-V", "urlcolor=accent", "-V", "toccolor=black",
         "-V", "mainfont=TeX Gyre Pagella", "-V", "sansfont=Inter",
         "-V", "monofont=DejaVu Sans Mono", "-V", "mathfont=Latin Modern Math",
         "-V", f"pdfkeywords={', '.join(m.get('keywords', []))}"])
    out = os.path.join(DIST, m["stem"] + ".pdf")
    # Put the full-colour cover in front, keeping the body's bookmarks and metadata.
    run(["qpdf", body, "--pages", os.path.join(BUILD, "cover.pdf"), body, "--", out])
    return out


def publish(m, outputs):
    dest = os.path.join(ROOT, "editions", f"{m['edition_short']}-edition")
    os.makedirs(dest, exist_ok=True)
    for path in outputs:
        shutil.copy(path, dest)
        print("published", os.path.relpath(os.path.join(dest, os.path.basename(path)), ROOT))


def main(argv):
    m = load_meta()
    targets = [a for a in argv if not a.startswith("--")] or ["cover", "epub", "pdf"]
    print(f"Building {m['title']}: {m['edition_label']} (v{m['version']})")
    outputs = []
    if "cover" in targets:
        build_cover(m)
    if "epub" in targets:
        outputs.append(build_epub(m))
    if "pdf" in targets:
        outputs.append(build_pdf(m))
    if "--publish" in argv:
        publish(m, outputs)


if __name__ == "__main__":
    main(sys.argv[1:])
