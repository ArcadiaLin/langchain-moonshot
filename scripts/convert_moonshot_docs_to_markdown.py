from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
from markdownify import markdownify as to_markdown

ROOT = Path("moonshot-docs")
DOCS_ROOT = ROOT / "platform.moonshot.cn" / "docs"


def extract_code_text(pre: Tag) -> str:
    lines = pre.select("span.line")
    if lines:
        code = "\n".join(line.get_text("", strip=False).rstrip("\n") for line in lines)
    else:
        code = pre.get_text("", strip=False)
    code = code.replace("\xa0", " ").strip("\n")
    return f"{code}\n"


def rewrite_href(href: str, current_html: Path) -> str:
    if not href or href.startswith(("mailto:", "tel:", "javascript:")):
        return href

    parsed = urlparse(href)

    if (
        parsed.scheme in {"http", "https"}
        and parsed.netloc in {"platform.moonshot.cn", "platform.kimi.com"}
    ):
        path = parsed.path or ""
        if path.startswith("/docs/"):
            target = DOCS_ROOT / path.removeprefix("/docs/")
            if target.suffix == ".html":
                target = target.with_suffix(".md")
            elif not target.suffix:
                target = target.with_suffix(".md")
            rel = Path(target).relative_to(DOCS_ROOT)
            href = str(Path(rel))
            if parsed.fragment:
                href = f"{href}#{parsed.fragment}"
            return href
        return href

    if parsed.scheme:
        return href

    if href.startswith("#"):
        return href

    if ".html" in href:
        return re.sub(r"\.html(?=($|#))", ".md", href)

    return href


def simplify_heading(tag: Tag) -> None:
    for button in list(tag.find_all("button")):
        button.decompose()

    anchor = tag.find("a", recursive=False)
    if anchor:
        text = anchor.get_text(" ", strip=True)
        anchor.replace_with(NavigableString(text))


def remove_pager(main: Tag) -> None:
    for child in list(main.find_all(recursive=False)):
        if child.name != "div":
            continue
        classes = set(child.get("class", []))
        if {"nx-mb-8", "nx-border-t", "nx-pt-8"}.issubset(classes):
            child.decompose()


def replace_videos(soup: BeautifulSoup, main: Tag, current_html: Path) -> None:
    for video in list(main.find_all("video")):
        src = video.get("src", "").strip()
        if not src:
            video.decompose()
            continue
        src = rewrite_href(src, current_html)
        p = soup.new_tag("p")
        link = soup.new_tag("a", href=src)
        link.string = f"视频资源: {Path(urlparse(src).path).name or src}"
        p.append(link)
        video.replace_with(p)


def replace_code_blocks(soup: BeautifulSoup, main: Tag) -> None:
    for wrapper in list(main.select("div.nextra-code-block")):
        pre = wrapper.find("pre")
        if not pre:
            wrapper.decompose()
            continue

        lang = (
            pre.get("data-language")
            or (pre.code.get("data-language") if pre.code else None)
            or ""
        )
        code = extract_code_text(pre)

        new_pre = soup.new_tag("pre")
        new_code = soup.new_tag("code")
        if lang:
            new_code["class"] = [f"language-{lang}"]
        new_code.string = code
        new_pre.append(new_code)
        wrapper.replace_with(new_pre)

    for pre in list(main.find_all("pre")):
        if pre.find("span", class_="line"):
            lang = (
                pre.get("data-language")
                or (pre.code.get("data-language") if pre.code else None)
                or ""
            )
            code = extract_code_text(pre)
            pre.clear()
            code_tag = soup.new_tag("code")
            if lang:
                code_tag["class"] = [f"language-{lang}"]
            code_tag.string = code
            pre.append(code_tag)


def replace_tablists(soup: BeautifulSoup, main: Tag) -> None:
    for tablist in list(main.select('[role="tablist"]')):
        labels = [
            btn.get_text(" ", strip=True) for btn in tablist.select('[role="tab"]')
        ]
        if not labels:
            tablist.decompose()
            continue
        p = soup.new_tag("p")
        p.string = f"代码示例语言: {' / '.join(labels)}"
        tablist.replace_with(p)

    for panel in list(main.select('[role="tabpanel"]')):
        if (
            panel.name == "span"
            and panel.get("aria-hidden") == "true"
            and not panel.get_text(strip=True)
        ):
            panel.decompose()
        else:
            panel.unwrap()


def simplify_block_links(soup: BeautifulSoup, main: Tag) -> None:
    for link in list(main.find_all("a", href=True)):
        if link.find_parent(re.compile(r"^h[1-6]$")):
            continue

        classes = set(link.get("class", []))
        is_blockish = bool(
            link.find(["div", "p", "section", "h1", "h2", "h3", "h4", "h5", "h6"])
        )
        is_blockish = is_blockish or bool(classes.intersection({"block", "flex"}))
        if not is_blockish:
            continue

        parts: list[str] = []
        for text in link.stripped_strings:
            cleaned = re.sub(r"\s+", " ", text).strip()
            if cleaned and (not parts or parts[-1] != cleaned):
                parts.append(cleaned)

        flat_text = " - ".join(parts)
        if not flat_text:
            link.decompose()
            continue

        new_link = soup.new_tag("a", href=link["href"])
        new_link.string = flat_text
        wrapper = soup.new_tag("p")
        wrapper.append(new_link)
        link.replace_with(wrapper)


def clean_main(page_soup: BeautifulSoup, main: Tag, current_html: Path) -> str:
    for selector in (
        ".nextra-breadcrumb",
        "button[aria-label='复制本节链接']",
        "button[title='Toggle word wrap']",
        "button[title='Copy code']",
        "svg",
        "script",
        "style",
        "noscript",
    ):
        for node in list(main.select(selector)):
            node.decompose()

    remove_pager(main)
    replace_tablists(page_soup, main)
    replace_code_blocks(page_soup, main)
    replace_videos(page_soup, main, current_html)

    for heading in main.find_all(re.compile(r"^h[1-6]$")):
        simplify_heading(heading)

    for button in list(main.find_all("button")):
        button.decompose()

    for link in main.find_all("a", href=True):
        link["href"] = rewrite_href(link["href"], current_html)
        text = link.get_text(" ", strip=True)
        if text.endswith("(opens in a new tab)"):
            link.string = text.replace(" (opens in a new tab)", "")

    simplify_block_links(page_soup, main)

    markdown = to_markdown(
        str(main),
        heading_style="ATX",
        bullets="-",
        strip=["button", "span"],
    )
    markdown = markdown.replace("\r\n", "\n")
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    markdown = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\.html(#.*?)?\)",
        r"[\1](\2.md\3)",
        markdown,
    )
    markdown = markdown.replace(" (opens in a new tab)", "")
    return markdown.strip() + "\n"


def convert_file(html_path: Path) -> Path:
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        raise ValueError(f"No main content found in {html_path}")

    markdown = clean_main(soup, main, html_path)
    output_path = html_path.with_suffix(".md")
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def build_index(markdown_files: list[Path]) -> None:
    lines = [
        "# Moonshot Docs Markdown Mirror",
        "",
        "- 入口文档: [overview](platform.moonshot.cn/docs/overview.md)",
        f"- Markdown 页面数: {len(markdown_files)}",
        "",
        "## 目录",
        "",
    ]
    for path in sorted(markdown_files):
        rel = path.relative_to(ROOT)
        label = str(rel).removeprefix("platform.moonshot.cn/docs/").removesuffix(".md")
        lines.append(f"- [{label}]({rel.as_posix()})")
    (ROOT / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    html_files = sorted(DOCS_ROOT.rglob("*.html"))
    markdown_files = [convert_file(path) for path in html_files]
    build_index(markdown_files)
    print(f"Converted {len(markdown_files)} HTML files to Markdown.")


if __name__ == "__main__":
    main()
