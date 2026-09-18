"""Tavily-only sourcing: record every Tavily result and enforce the report's reference list against it."""

import re
import threading
from dataclasses import dataclass

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

MAX_REFERENCES = 5


def _norm(url: str) -> str:
    return url.strip().rstrip("/.,;)").lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")


@dataclass
class Source:
    url: str
    title: str
    snippet: str
    score: float
    query: str


class SourceRegistry:
    """Every URL Tavily returned during one research run. The only allowed citations."""

    def __init__(self):
        self._lock = threading.Lock()
        self._by_url: dict[str, Source] = {}
        self.queries: list[str] = []

    def add(self, query: str, results: list[dict]) -> None:
        with self._lock:
            self.queries.append(query)
            for r in results:
                url = r.get("url")
                if not url:
                    continue
                key = _norm(url)
                score = float(r.get("score") or 0)
                if key not in self._by_url or score > self._by_url[key].score:
                    self._by_url[key] = Source(url, r.get("title") or url, (r.get("content") or "")[:300], score, query)

    def ranked(self) -> list[Source]:
        with self._lock:
            return sorted(self._by_url.values(), key=lambda s: s.score, reverse=True)

    def get(self, url: str) -> Source | None:
        return self._by_url.get(_norm(url))

    def __len__(self) -> int:
        return len(self._by_url)


def make_tavily_tools(registry: SourceRegistry, max_results: int, search_depth: str, api_wrapper=None):
    """Return (tavily_search, list_tavily_sources) tools bound to `registry`."""
    kwargs = {"api_wrapper": api_wrapper} if api_wrapper else {}
    client = TavilySearch(max_results=max_results, search_depth=search_depth, **kwargs)

    @tool
    def tavily_search(query: str) -> dict:
        """Search the web with the Tavily API. This is the ONLY permitted source of information."""
        result = client.invoke({"query": query})
        if isinstance(result, dict):
            registry.add(query, result.get("results", []))
            # Keep only what the model needs; drops images/raw noise.
            return {
                "query": query,
                "results": [
                    {"title": r.get("title"), "url": r.get("url"), "content": r.get("content"), "score": r.get("score")}
                    for r in result.get("results", [])
                ],
            }
        return result

    @tool
    def list_tavily_sources() -> str:
        """List every source returned by Tavily in this run, ranked by relevance.
        References in the report MUST be chosen from this list only."""
        sources = registry.ranked()
        if not sources:
            return "No Tavily sources recorded."
        return "\n".join(f"- {s.title} | {s.url} | relevance {s.score:.2f}" for s in sources)

    return tavily_search, list_tavily_sources


# ---------------------------------------------------------------- post-run enforcement

_REF_HEADING = re.compile(r"^(#{1,3})\s*(references|sources)\b.*$", re.I | re.M)
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_BARE_URL = re.compile(r"https?://[^\s)>\]]+")


def enforce_references(report: str, registry: SourceRegistry, limit: int = MAX_REFERENCES) -> tuple[str, dict]:
    """Keep only Tavily-returned references (max `limit`), renumber inline [n] citations to match.

    Returns (new_report, stats).
    """
    m = _REF_HEADING.search(report)
    body, refs_block = (report[: m.start()], report[m.end():]) if m else (report, "")
    heading = m.group(1) if m else "##"

    # Stop the references block at the next heading of same or higher level.
    tail = ""
    nxt = re.search(rf"^#{{1,{len(heading)}}}\s", refs_block, re.M)
    if nxt:
        refs_block, tail = refs_block[: nxt.start()], refs_block[nxt.start():]

    # Parse existing references in order: (old_number, url, title)
    parsed = []
    for i, line in enumerate(l for l in refs_block.splitlines() if l.strip()):
        num_m = re.match(r"\s*(?:\[(\d+)\]|(\d+)[.)])", line)
        old_num = int(num_m.group(1) or num_m.group(2)) if num_m else i + 1
        link = _MD_LINK.search(line)
        url = link.group(2) if link else (_BARE_URL.search(line).group(0) if _BARE_URL.search(line) else None)
        if url:
            parsed.append((old_num, url, link.group(1) if link else None))

    kept, dropped, seen = [], [], set()
    for old_num, url, title in parsed:
        src = registry.get(url)
        if src is None or _norm(url) in seen:
            dropped.append(url)
            continue
        seen.add(_norm(url))
        kept.append((old_num, src))

    # Too few valid references: top up with the highest-ranked uncited Tavily sources.
    for src in registry.ranked():
        if len(kept) >= limit:
            break
        if _norm(src.url) not in seen:
            seen.add(_norm(src.url))
            kept.append((None, src))
    kept = kept[:limit]

    mapping = {old: new for new, (old, _) in enumerate(kept, 1) if old is not None}

    def _renumber(match):
        nums = [n.strip() for n in match.group(1).split(",")]
        new = sorted({mapping[int(n)] for n in nums if n.isdigit() and int(n) in mapping})
        return "[" + ", ".join(map(str, new)) + "]" if new else ""

    body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()", _renumber, body)
    body = re.sub(r" +([.,;])", r"\1", body)

    refs_md = "\n".join(f"{n}. [{s.title}]({s.url})" for n, (_, s) in enumerate(kept, 1))
    new_report = f"{body.rstrip()}\n\n{heading} References\n\n{refs_md}\n" + (f"\n{tail.lstrip()}" if tail.strip() else "")

    return new_report, {"kept": len(kept), "dropped": dropped, "sources_found": len(registry), "searches": len(registry.queries)}
