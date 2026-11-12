"""Code checks on one answer, with no judge.  Each returns what it found wrong."""

import json
import re
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

# An offer to do more at the end of an answer ("If you want, I can..."),
# from the October gpt-5-mini comparison
OFFER = re.compile(
    r"(?i)\bif you(?:'d| would)? (?:want|like|prefer|wish)\b|\bwould you like\b"
    r"|\bdo you want\b|\bwant me to\b|\blet me know if\b|\bhappy to\b|\bshall i\b"
    r"|\bI can (?:also )?(?:help|look|narrow|find|walk|show|suggest|recommend"
    r"|provide|search|check|give|share|explain|turn|tailor|compare|pull|list|break)"
)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
RULE = re.compile(r"[-*_]{3,}")
# [text](url), with the url ending at the first space or closing parenthesis
MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]*)[^)]*\)")
URL = re.compile(r"https?://[^\s)\]>\"']+")


def _normalize_quotes(text: str) -> str:
    return (text or "").replace("\u2019", "'").replace("\u2018", "'")


def _normalize_url(url: str) -> str:
    """Compare urls without case in the scheme and host or a trailing slash."""
    parts = urlsplit(url.strip())
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            parts.query,
            parts.fragment,
        )
    )


def _tool_outputs(context: Iterable[str]) -> list[dict]:
    """Return the tool outputs in an answer's context that are JSON objects."""
    outputs = []
    for item in context or []:
        try:
            parsed = json.loads(item)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            outputs.append(parsed)
    return outputs


def last_paragraph(answer: str) -> str:
    """Return the last paragraph, ignoring HTML comments and horizontal rules."""
    text = HTML_COMMENT.sub("", _normalize_quotes(answer))
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    paragraphs = [p for p in paragraphs if not RULE.fullmatch(p)]
    return paragraphs[-1] if paragraphs else ""


def ends_with_offer(answer: str) -> list[str]:
    """Return the answer's last paragraph if it offers to do more."""
    ending = last_paragraph(answer)
    return [ending] if OFFER.search(ending) else []


def unlinked_resources(answer: str, context: Iterable[str]) -> list[str]:
    """Search result titles the answer mentions without a link to their own url."""
    answer = _normalize_quotes(answer)
    links = {
        (text.strip().lower(), _normalize_url(url))
        for text, url in MARKDOWN_LINK.findall(answer)
    }
    unlinked = []
    for output in _tool_outputs(context):
        for result in output.get("results") or []:
            if not isinstance(result, dict) or not result.get("title"):
                continue
            title = _normalize_quotes(result["title"]).strip()
            if title.lower() not in answer.lower():
                continue
            url = _normalize_url(result.get("url") or "")
            linked = any(
                title.lower() in text and url and link_url == url
                for text, link_url in links
            )
            if not linked and title not in unlinked:
                unlinked.append(title)
    return unlinked


def bare_urls(answer: str) -> list[str]:
    """Urls given as plain text instead of as a linked title."""
    found = []
    for text, url in MARKDOWN_LINK.findall(answer or ""):
        if URL.fullmatch(text.strip()):
            found.append(url)
    outside_links = MARKDOWN_LINK.sub(" ", answer or "")
    found.extend(URL.findall(outside_links))
    return found


# Each check, run on an LLMTestCase
CHECKS = {
    "follow_up_offer": lambda case: ends_with_offer(case.actual_output),
    "unlinked_resource": lambda case: unlinked_resources(
        case.actual_output, case.retrieval_context
    ),
    "bare_url": lambda case: bare_urls(case.actual_output),
}
CHECKS_BY_BOT = {
    "recommendation": ("follow_up_offer", "unlinked_resource", "bare_url"),
    "syllabus": ("follow_up_offer",),
    "video_gpt": ("follow_up_offer",),
}


def run_checks(bot_name: str, test_case) -> dict[str, list[str]]:
    """Run a bot's code checks on one answer; an empty list means it passed."""
    return {name: CHECKS[name](test_case) for name in CHECKS_BY_BOT.get(bot_name, ())}
