"""
Builder2 musicDirection contract — Winner Development output and media validation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from engine.builder2_tournament_contracts import Builder2TournamentError, require_non_empty_str


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _as_bool(value: Any) -> Optional[bool]:
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return None


def normalize_music_direction(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise Builder2TournamentError("builder2_winner_schema_invalid:musicDirection")
    prompt = _clean(raw.get("prompt"))
    instrumental = _as_bool(raw.get("instrumentalOnly"))
    immediate = _as_bool(raw.get("immediateStart"))
    out: Dict[str, Any] = {}
    if prompt:
        out["prompt"] = prompt
    if instrumental is not None:
        out["instrumentalOnly"] = instrumental
    if immediate is not None:
        out["immediateStart"] = immediate
    if not out:
        return None
    return out


def validate_music_direction_shape(raw: Any, *, field_prefix: str = "musicDirection") -> Optional[Dict[str, Any]]:
    """Optional winner-plan validation — does not require musicDirection to exist."""
    if raw is None:
        return None
    normalized = normalize_music_direction(raw)
    if normalized is None:
        raise Builder2TournamentError(f"builder2_winner_schema_invalid:{field_prefix}")
    if normalized.get("prompt"):
        require_non_empty_str(normalized.get("prompt"), field=f"{field_prefix}.prompt")
    instrumental = normalized.get("instrumentalOnly")
    if instrumental is not None and instrumental is not True:
        raise Builder2TournamentError(f"builder2_winner_schema_invalid:{field_prefix}.instrumentalOnly")
    immediate = normalized.get("immediateStart")
    if immediate is not None and immediate is not True:
        raise Builder2TournamentError(f"builder2_winner_schema_invalid:{field_prefix}.immediateStart")
    return normalized


def extract_music_direction(plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = plan.get("musicDirection")
    if raw is None:
        return None
    return validate_music_direction_shape(raw)


def validate_music_direction_for_lyria_media(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Strict validation when BUILDER2_LYRIA_ENABLED and media pipeline requires soundtrack."""
    raw = plan.get("musicDirection")
    if not isinstance(raw, dict):
        raise Builder2TournamentError("builder2_media_missing_music_direction")
    prompt = _clean(raw.get("prompt"))
    if not prompt:
        raise Builder2TournamentError("builder2_media_missing_music_direction_prompt")
    if _as_bool(raw.get("instrumentalOnly")) is not True:
        raise Builder2TournamentError("builder2_media_music_direction_instrumental_required")
    if _as_bool(raw.get("immediateStart")) is not True:
        raise Builder2TournamentError("builder2_media_music_direction_immediate_start_required")
    return {
        "prompt": prompt,
        "instrumentalOnly": True,
        "immediateStart": True,
    }


_LYRIA_PRODUCTION_CONSTRAINTS = (
    "Production constraints (mandatory): Instrumental only. No vocals. No lyrics. No spoken words. "
    "Begin the musical character immediately from the first beat. No long intro. "
    "No silence at the start."
)

_LYRIA_SHORT_AD_GUARDRAIL = (
    "Short-ad soundtrack guardrail (mandatory): This soundtrack is for a very short advertisement. "
    "Full arrangement from the first beat: the essential rhythmic, harmonic, low-frequency, "
    "melodic, and complementary layers appropriate to the creative direction must already be "
    "established within the opening 1–2 seconds. Do not use a gradual build-up that saves important "
    "instruments or arrangement richness for later in the generated track. "
    "The opening portion must already sound complete, layered, and professionally produced. "
    "Gentle or restrained music may still be harmonically and rhythmically complete — not thin or sparse "
    "unless the creative direction genuinely requires it."
)


def build_lyria_request_prompt(music_direction: Dict[str, Any]) -> Tuple[str, str]:
    """Return (creative_prompt, combined_prompt_for_api)."""
    creative = _clean(music_direction.get("prompt"))
    combined = f"{creative}\n\n{_LYRIA_PRODUCTION_CONSTRAINTS}\n\n{_LYRIA_SHORT_AD_GUARDRAIL}".strip()
    return creative, combined
