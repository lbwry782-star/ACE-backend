"""
Builder2 video headline via o3-pro (separate from video planning).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
from openai import OpenAI

from engine.ace_usage_memory import get_used_headlines, remember_headline

logger = logging.getLogger(__name__)

_MAX_HEADLINE_WORDS = 7

_HE_FILLER_WORDS = frozenset(
    {
        "את",
        "של",
        "על",
        "עם",
        "אל",
        "מן",
        "מה",
        "אם",
        "כי",
        "גם",
        "רק",
        "עוד",
        "כבר",
        "מאוד",
        "הרבה",
        "יותר",
        "פחות",
        "כמו",
        "כדי",
        "אצל",
        "בין",
        "אחרי",
        "לפני",
        "תוך",
        "כש",
        "כאשר",
        "שיש",
        "שאין",
        "שלא",
    }
)
_EN_FILLER_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "with",
        "from",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "your",
        "my",
        "our",
        "their",
        "that",
        "this",
        "just",
        "also",
        "even",
        "very",
        "really",
        "so",
        "too",
        "when",
        "while",
        "who",
        "whom",
        "which",
        "into",
        "upon",
    }
)


def _headline_word_count(text: str) -> int:
    return len([w for w in (text or "").strip().split() if w])


def _assemble_headline_full(product_name: str, remainder: str) -> str:
    return " ".join(f"{product_name} {remainder}".split())


def _normalize_headline_word(word: str) -> str:
    return (word or "").strip().strip(".,!?;:\"'()[]").lower()


def _find_replacement_object_indices(
    remainder: str,
    *,
    replacement_object: str,
    object_a: str,
    object_b: str,
) -> set[int]:
    """Indices of remainder words that carry the substituted object (must be preserved)."""
    words = [w for w in remainder.split() if w]
    protected: set[int] = set()
    candidates: list[str] = []
    for raw in (replacement_object, object_a, object_b):
        c = (raw or "").strip()
        if c and c not in candidates:
            candidates.append(c)

    for cand in candidates:
        cand_words = [w for w in cand.split() if w]
        if not cand_words:
            continue
        if len(cand_words) == 1:
            cand_norm = _normalize_headline_word(cand_words[0])
            for i, word in enumerate(words):
                word_norm = _normalize_headline_word(word)
                if word_norm == cand_norm or cand_norm in word_norm or word_norm in cand_norm:
                    protected.add(i)
        else:
            cand_norm = " ".join(_normalize_headline_word(w) for w in cand_words)
            for start in range(len(words) - len(cand_words) + 1):
                window = " ".join(_normalize_headline_word(w) for w in words[start : start + len(cand_words)])
                if window == cand_norm:
                    protected.update(range(start, start + len(cand_words)))
    return protected


def _filler_words_for_language(language: str) -> frozenset[str]:
    return _EN_FILLER_WORDS if (language or "").strip().lower() == "en" else _HE_FILLER_WORDS


def _shorten_remainder_to_word_budget(
    remainder: str,
    *,
    max_words: int,
    protected_indices: set[int],
    original_expression: str,
    language: str,
) -> str | None:
    """Deterministically shorten remainder; preserve substituted object word(s)."""
    words = [w for w in remainder.split() if w]
    if max_words < 1:
        return None
    if len(words) <= max_words:
        return " ".join(words)

    fillers = _filler_words_for_language(language)
    original_norm = {
        _normalize_headline_word(w) for w in (original_expression or "").split() if w
    }

    def removal_rank(index: int, word: str) -> tuple[int, int, int]:
        if index in protected_indices:
            return (99, 0, index)
        word_norm = _normalize_headline_word(word)
        if word_norm in fillers:
            tier = 0
        elif original_norm and word_norm not in original_norm:
            tier = 1
        else:
            tier = 2
        if protected_indices:
            dist = min(abs(index - p) for p in protected_indices)
        else:
            dist = min(index, len(words) - 1 - index)
        return (tier, dist, index)

    current = list(words)
    current_protected = set(protected_indices)

    while len(current) > max_words:
        removable = [i for i in range(len(current)) if i not in current_protected]
        if not removable:
            break
        remove_at = min(removable, key=lambda i: removal_rank(i, current[i]))
        current.pop(remove_at)
        current_protected = {
            (i if i < remove_at else i - 1)
            for i in current_protected
            if i != remove_at
        }

    if len(current) <= max_words and current:
        return " ".join(current)

    if not protected_indices:
        return " ".join(words[:max_words]) if len(words) >= max_words else None

    best: str | None = None
    best_score = -1
    for start in range(len(words) - max_words + 1):
        window_indices = set(range(start, start + max_words))
        if not protected_indices & window_indices:
            continue
        window_words = words[start : start + max_words]
        overlap = (
            sum(1 for w in window_words if _normalize_headline_word(w) in original_norm)
            if original_norm
            else max_words
        )
        score = overlap * 100 - start
        if score > best_score:
            best_score = score
            best = " ".join(window_words)
    return best


def _fit_headline_to_word_limit(
    *,
    product_name: str,
    remainder: str,
    data: dict[str, Any],
    object_a: str,
    object_b: str,
    language: str,
) -> tuple[str, str] | None:
    """Return (shortened_remainder, headline_full) within 7 words, or None."""
    pn = " ".join(product_name.split())
    rem = " ".join(remainder.split())
    hfull = _assemble_headline_full(pn, rem)
    if _headline_word_count(hfull) <= _MAX_HEADLINE_WORDS:
        return rem, hfull

    logger.info("VIDEO_HEADLINE_TOO_LONG_SHORTEN_START")
    old_full = hfull
    remainder_budget = _MAX_HEADLINE_WORDS - _headline_word_count(pn)
    if remainder_budget < 1:
        logger.info("VIDEO_HEADLINE_TOO_LONG_SHORTEN_FAIL")
        return None

    protected = _find_replacement_object_indices(
        rem,
        replacement_object=(data.get("headlineReplacementObject") or "").strip(),
        object_a=object_a,
        object_b=object_b,
    )
    shortened = _shorten_remainder_to_word_budget(
        rem,
        max_words=remainder_budget,
        protected_indices=protected,
        original_expression=(data.get("headlineOriginalExpression") or "").strip(),
        language=language,
    )
    if not shortened:
        logger.info("VIDEO_HEADLINE_TOO_LONG_SHORTEN_FAIL")
        return None

    new_full = _assemble_headline_full(pn, shortened)
    if _headline_word_count(new_full) > _MAX_HEADLINE_WORDS:
        logger.info("VIDEO_HEADLINE_TOO_LONG_SHORTEN_FAIL")
        return None

    logger.info(
        "VIDEO_HEADLINE_TOO_LONG_SHORTEN_OK old=%s new=%s",
        old_full[:300],
        new_full[:300],
    )
    return shortened, new_full


def _video_headline_rhyming_substitution_block() -> str:
    """Same creative rule set as Builder1 headline (remainder-only output)."""
    return (
        "HEADLINE (rhyming object substitution — mandatory for headlineText remainder):\n"
        "1. First find an existing familiar expression, idiom, proverb, or well-known phrase that expresses the advertisingPromise.\n"
        "2. The original expression must NOT already contain the name or core word of Object A or Object B.\n"
        "3. Choose exactly one word inside that expression.\n"
        "4. Replace that one word with the name of Object A or Object B (natural headline-language form).\n"
        "5. The replacement must be based on strong phonetic similarity, not rhyme alone.\n"
        "6. The inserted Object A/Object B name must preserve most of the sound structure of the replaced word.\n"
        "7. Prefer replacements where the object name differs by only one small sound, syllable, or letter cluster.\n"
        "8. Simple end-rhyme is not enough.\n"
        "9. If the audience cannot immediately hear the original expression behind the twist, reject it.\n"
        "10. Invalid weak substitution example: original word חשבון → replacement object סבון — only end-rhyme; too much of the original word is lost; the original expression is not immediately recognizable.\n"
        "11. If only a weak rhyme exists, choose another expression/object substitution instead. Do not force a weak substitution.\n"
        "12. The result must still feel like a recognizable twist on the original expression.\n"
        "13. The final headline remainder must visibly differ from the original expression.\n"
        "14. The viewer must immediately notice the substituted object word.\n"
        "15. Do not use a substitution if the final phrase reads exactly like the original expression.\n"
        "16. Do not replace a word with an object name that is already hidden inside the original expression or naturally contained across adjacent letters.\n"
        "17. The replacement must create a visible, readable twist, not only an internal spelling explanation.\n"
        "18. Invalid unchanged-substitution example: original_expression השלם גדול מסך חלקיו, replaced_word סך, replacement_object מסך, final_headline_remainder השלם גדול מסך חלקיו — the final phrase reads exactly like the original expression; no visible twist; forbidden.\n"
        "19. The final substituted headline must express the advertisingPromise.\n"
        "20. Prefer the strongest case: the substitution itself should be the expression of the advertisingPromise.\n"
        "21. The viewer should feel that replacing the original word with Object A or Object B is exactly what creates the advertising meaning.\n"
        "22. It is not enough that the original expression expresses the promise, or that the final phrase sounds clever.\n"
        "23. The best headline has all three: (a) the original expression is recognizable, (b) the object-word substitution is visible and phonetically strong, (c) the substitution itself makes the advertisingPromise understandable.\n"
        "24. If the substitution is only a pun but does not carry the advertisingPromise, reject it and choose another expression/substitution.\n"
        "25. Do not add extra words before, inside, or after the twisted expression.\n"
        "26. headlineProductName must exactly match productNameResolved (backend-fixed). headlineText is the twisted expression remainder only — do not repeat the product name in headlineText.\n"
        "27. headlineFull must be exactly headlineProductName, one ASCII space, then headlineText. ≤7 words total on headlineFull.\n"
        "28. headlineText must express the advertisingPromise through the video interaction — not by restating the promise literally.\n"
        "29. Do NOT pick an expression that already contains the object word before substitution.\n"
        "30. The replacement object may be Object A or Object B.\n"
    )


def _parse_json_object(raw: str) -> dict[str, Any]:
    t = (raw or "").strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1]) if len(lines) > 2 else t
    t = t.strip()
    if t.lower().startswith("```json"):
        t = t[7:].lstrip()
    t = t.strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object")
    obj = json.loads(t[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("model_output_not_object")
    return obj


def _log_video_headline_rhyme_diagnostics(data: dict[str, Any], *, final_remainder: str) -> None:
    logger.info("VIDEO_HEADLINE_RHYME final_headline_remainder=%s", final_remainder[:200])
    for key, log_key in (
        ("headlineOriginalExpression", "original_expression"),
        ("headlineReplacedWord", "replaced_word"),
        ("headlineReplacementObject", "replacement_object"),
        ("headlineRhymeReason", "rhyme_reason"),
    ):
        value = (data.get(key) or "").strip()
        if value:
            logger.info("VIDEO_HEADLINE_RHYME %s=%s", log_key, value[:300])


class VideoHeadlineError(RuntimeError):
    pass


def generate_video_headline_o3(
    *,
    plan: dict[str, Any],
    product_description: str = "",
) -> dict[str, str]:
    resolved = (plan.get("productNameResolved") or "").strip()
    if not resolved:
        raise VideoHeadlineError("missing_product_name_resolved")

    language = (plan.get("language") or "he").strip().lower()
    advertising_promise = (plan.get("advertisingPromise") or "").strip()
    object_a = (plan.get("objectA") or "").strip()
    object_b = (plan.get("objectB") or "").strip()
    interaction_summary = (plan.get("interactionSummary") or "").strip()
    interaction_script = (plan.get("interactionScript") or "").strip()

    used_headlines = get_used_headlines("builder2")
    logger.info("VIDEO_HEADLINE_SEPARATE_CALL_START")
    logger.info(
        "VIDEO_HEADLINE_MEMORY_INJECTED count=%s recent=%r",
        len(used_headlines),
        used_headlines[-10:],
    )
    logger.info("VIDEO_HEADLINE_RULE=rhyming_object_substitution")

    rhyme_block = _video_headline_rhyming_substitution_block()
    system = (
        "Return exactly one JSON object, no markdown, no prose. Required keys only:\n"
        '{"headlineProductName":"...","headlineText":"...","headlineFull":"..."}\n'
        "Optional diagnostic keys (may be omitted): headlineOriginalExpression, headlineReplacedWord, "
        "headlineReplacementObject, headlineRhymeReason.\n"
        "Rules:\n"
        "- Write in the same language as the request language (he or en).\n"
        "- The advertising promise is already resolved inside the video interaction. The headline must not create or restate it literally.\n"
        "- Use objectA, objectB, interactionSummary, and interactionScript as context for how the twisted expression connects to the video.\n"
        f"{rhyme_block}"
        "- headlineProductName must exactly match the given productNameResolved string.\n"
        "- The model controls headlineText only; headlineProductName is fixed by backend.\n"
        "- Do not include product description inside headlineProductName.\n"
        "- Do not create a slogan that merely explains the product.\n"
        "- Do NOT write explanatory slogans or direct benefit statements of the advertising promise.\n"
        "- Phrases like 'מסך מול מסך' are too descriptive and lack meaning.\n"
        "- The headline is not a literal shot description.\n"
        "- Headline memory excludes product names; compare only the slogan/title phrase part.\n"
        "- Do not reuse any previous headlineText from memory.\n"
        "- Do not reuse the same familiar expression; choose a fresh phrasing.\n"
    )
    user = (
        f"language: {language}\n"
        f"productNameResolved: {resolved}\n"
        f"productDescription: {(product_description or '').strip()}\n"
        f"advertisingPromise: {advertising_promise}\n"
        f"objectA: {object_a}\n"
        f"objectB: {object_b}\n"
        f"interactionSummary: {interaction_summary}\n"
        f"interactionScript: {interaction_script}\n"
        f"headlineTextMemoryToAvoidACE: {', '.join(used_headlines)}\n"
    )

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise VideoHeadlineError("openai_unconfigured")
    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(120.0),
        max_retries=0,
    )
    combined = f"{system.strip()}\n\n{user.strip()}"
    response = client.responses.create(
        model="o3-pro",
        input=combined,
        reasoning={"effort": "low"},
    )
    out_text = getattr(response, "output_text", None) or ""
    data = _parse_json_object(out_text)

    hpn = resolved
    htt = (data.get("headlineText") or "").strip()
    if not hpn or not htt:
        raise VideoHeadlineError("headline_empty_field")

    fitted = _fit_headline_to_word_limit(
        product_name=hpn,
        remainder=htt,
        data=data,
        object_a=object_a,
        object_b=object_b,
        language=language,
    )
    if not fitted:
        raise VideoHeadlineError("headline_too_long")
    htt, hfull = fitted

    headline_without_product_name = htt
    hfull_norm = " ".join((hfull or "").split())
    hpn_norm = " ".join((hpn or "").split())
    if hpn_norm and hfull_norm.startswith(f"{hpn_norm} "):
        headline_without_product_name = hfull_norm[len(hpn_norm) + 1 :].strip()
    if (headline_without_product_name or "").strip():
        remember_headline("builder2", headline_without_product_name)

    logger.info("VIDEO_HEADLINE_SEPARATE_CALL_OK headlineText=%s", htt[:300])
    logger.info("VIDEO_HEADLINE_FULL_ASSEMBLED headlineFull=%s", hfull[:300])
    _log_video_headline_rhyme_diagnostics(data, final_remainder=headline_without_product_name)

    return {
        "headlineProductName": hpn,
        "headlineText": htt,
        "headlineFull": hfull,
    }
