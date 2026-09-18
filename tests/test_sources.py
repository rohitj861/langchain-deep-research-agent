"""Tavily-only references: non-Tavily URLs dropped, max 5 kept, inline citations renumbered."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_agent.sources import SourceRegistry, enforce_references


def make_registry(n=7):
    reg = SourceRegistry()
    reg.add("q1", [{"url": f"https://site{i}.com/page", "title": f"Site {i}", "content": "x", "score": 1 - i / 10} for i in range(1, n + 1)])
    return reg


def test_drops_non_tavily_and_caps_at_five():
    reg = make_registry()
    report = """# T

Claim A [1]. Claim B [2]. Fake claim [3]. Claim C [4, 5]. Claim D [6]. Claim E [7].

## References
1. [Site 1](https://site1.com/page)
2. [Site 2](https://www.site2.com/page/)
3. [Made up](https://not-from-tavily.com/x)
4. [Site 4](https://site4.com/page)
5. [Site 5](https://site5.com/page)
6. [Site 6](https://site6.com/page)
7. [Site 7](https://site7.com/page)
"""
    out, stats = enforce_references(report, reg)
    refs = out.split("## References")[1]
    assert stats["kept"] == 5
    assert stats["dropped"] == ["https://not-from-tavily.com/x"]
    assert "not-from-tavily" not in out
    assert refs.count("](https://") == 5
    # old 4,5 -> new 3,4 ; old 6 -> 5 ; old 3 (fake) and 7 (over cap) removed
    assert "Claim A [1]." in out and "Claim B [2]." in out
    assert "Fake claim." in out
    assert "Claim C [3, 4]." in out and "Claim D [5]." in out
    assert "Claim E." in out


def test_tops_up_from_tavily_when_report_has_too_few():
    reg = make_registry()
    report = "# T\n\nOnly one [1].\n\n## References\n1. [Site 3](https://site3.com/page)\n"
    out, stats = enforce_references(report, reg)
    assert stats["kept"] == 5
    refs = out.split("## References")[1]
    assert refs.index("site3.com") < refs.index("site1.com")  # original ref stays first
