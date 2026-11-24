"""Code checks on one answer, with no judge.  Each returns what it found wrong."""

import json
import re
from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit

# An offer to do more at the end of an answer ("If you want, I can..."),
# from the October gpt-5-mini comparison.  "If you want" alone isn't one: a
# recommendation says "Best fit if you want X", so it needs an offer with it.
IF_YOU_WANT = r"\bif you(?:'d| would)? (?:want|like|prefer|wish)\b"
OFFER = re.compile(
    rf"(?i){IF_YOU_WANT}[^.!?\n]*?"
    r"\b(?:I(?:'ll| will| can| could)|let me|tell me|just ask|feel free|ask me)\b"
    rf"|\bI(?:'ll| will| can| could)\b[^.!?\n]*?{IF_YOU_WANT}"
    r"|\bwould you like\b"
    r"|\bdo you want\b|\bwant me to\b|\blet me know\b|\bhappy to\b|\bshall i\b"
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
    try:
        parts = urlsplit(url.strip())
    except ValueError:  # a model's malformed link, such as "https://[x"
        return url.strip()
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
    # Several results can share a title, each with its own url
    urls: dict[str, set[str]] = {}
    for output in _tool_outputs(context):
        for result in output.get("results") or []:
            if isinstance(result, dict) and result.get("title"):
                title = _normalize_quotes(result["title"]).strip()
                url = _normalize_url(result.get("url") or "")
                urls.setdefault(title, set()).update({url} - {""})
    unlinked = []
    unseen = answer.lower()
    # Longest first, so a title inside a longer one isn't taken for a mention
    for title in sorted(urls, key=len, reverse=True):
        if title.lower() not in unseen:
            continue
        unseen = unseen.replace(title.lower(), " ")
        if not any(title.lower() in text and url in urls[title] for text, url in links):
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


def tool_use(metadata: dict) -> list[str]:
    """Return expected tools the bot didn't call, or any it called if none were."""
    expected = metadata.get("expected_tools")
    if expected is None:
        return []
    called = metadata.get("tool_names") or []
    if not expected:
        return [f"called {name}" for name in dict.fromkeys(called)]
    return [f"didn't call {name}" for name in expected if name not in called]


def follow_up_offer(case) -> list[str]:
    """Return the answer's closing offer, if it ends with one.

    A recommendation reply that asked the user instead of searching is exempt:
    the prompt tells it to end with its questions.
    """
    metadata = case.metadata or {}
    if metadata.get("bot_name") == "recommendation" and not metadata.get("tool_names"):
        return []
    return ends_with_offer(case.actual_output)


# Each check, run on an LLMTestCase
CHECKS = {
    "tool_use": lambda case: tool_use(case.metadata or {}),
    "follow_up_offer": follow_up_offer,
    "unlinked_resource": lambda case: unlinked_resources(
        case.actual_output, case.retrieval_context
    ),
    "bare_url": lambda case: bare_urls(case.actual_output),
}
CHECKS_BY_BOT = {
    "recommendation": ("follow_up_offer", "unlinked_resource", "bare_url", "tool_use"),
    "syllabus": ("follow_up_offer",),
    "syllabus_canvas": ("follow_up_offer",),
    "video_gpt": ("follow_up_offer",),
}


def run_checks(bot_name: str, test_case) -> dict[str, list[str]]:
    """Run a bot's code checks on one answer; an empty list means it passed."""
    return {name: CHECKS[name](test_case) for name in CHECKS_BY_BOT.get(bot_name, ())}
