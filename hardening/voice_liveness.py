"""
§5.1 — Voice-liveness / deepfake-detection integration point. INTERFACE + STUB ONLY.

Kavach does not, and should not, ship its own voice-cloning detector: that is a
specialised audio-ML problem and a fabricated in-house detector would be worse
than none (it would emit confident ``False``/``True`` verdicts with no basis,
which the rules would then act on). What Kavach *does* own is the contract:

* how a third-party provider is called,
* what it must return,
* and — most importantly — what happens when it is unavailable, times out,
  errors, or returns low confidence: the answer is **unknown**, represented as
  ``None`` in ``KavachTransaction.caller_verified_biometric``, never coerced
  to a boolean (brief §5.1) and never treated as evidence either way (§5.5).

Wiring
------
The bank's app / IVR layer captures a short audio sample from the active call
(with consent, under the bank's own privacy terms — out of Kavach's scope). The
adapter layer calls ``resolve_liveness(provider, sample)`` and puts the result
into ``caller_verified_biometric`` before ``normalize()`` completes. Downstream,
``V1RuleClassifier`` and the feature contract already treat ``None`` as
"missing" (its own indicator slot), ``False`` as an escalating signal, and
``True`` as a non-escalating one.

Nothing here performs audio analysis. ``NullLivenessProvider`` returns UNKNOWN
for everything and is the default.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LivenessVerdict(str, Enum):
    LIVE_HUMAN = "live_human"  # provider is confident the voice is a live human speaker
    SYNTHETIC = "synthetic"  # provider is confident the voice is synthetic / replayed / cloned
    UNKNOWN = "unknown"  # unavailable, timed out, low confidence, unsupported, consent absent


@dataclass(frozen=True)
class AudioSample:
    """Opaque handle to a consented audio sample. Kavach never stores the bytes."""

    reference: str  # provider-side or bank-side id of the captured sample
    duration_ms: int
    codec: str = "opus"
    sample_rate_hz: int = 16000


@dataclass(frozen=True)
class LivenessResult:
    verdict: LivenessVerdict
    confidence: Optional[float]  # provider-reported, 0–1, None if not supplied
    provider: str
    latency_ms: Optional[int] = None
    error: Optional[str] = None  # populated when verdict is UNKNOWN because of a failure

    def to_tristate(self, min_confidence: float = 0.9) -> Optional[bool]:
        """Map to ``caller_verified_biometric``. UNKNOWN or low confidence → ``None``. Never a guess."""
        if self.verdict is LivenessVerdict.UNKNOWN:
            return None
        if self.confidence is not None and self.confidence < min_confidence:
            return None
        return self.verdict is LivenessVerdict.LIVE_HUMAN


class VoiceLivenessProvider(ABC):
    """Contract a third-party detector integration must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def check(self, sample: AudioSample, timeout_ms: int = 800) -> LivenessResult:
        """Return a verdict. MUST NOT raise: any failure is a ``LivenessResult`` with UNKNOWN + ``error``.

        ``timeout_ms`` is short on purpose — this runs inside the transaction's
        authorisation window. A slow provider degrades to UNKNOWN, not to a
        blocked payment and not to a silently assumed verdict.
        """


class NullLivenessProvider(VoiceLivenessProvider):
    """Default: no provider configured. Everything is UNKNOWN."""

    name = "none"

    def check(self, sample: AudioSample, timeout_ms: int = 800) -> LivenessResult:
        return LivenessResult(LivenessVerdict.UNKNOWN, None, self.name, error="no liveness provider configured")


class StubLivenessProvider(VoiceLivenessProvider):
    """Test double. Returns a fixed verdict; lets the rest of the pipeline be exercised.

    NOT a detector. Do not deploy.
    """

    name = "stub"

    def __init__(self, verdict: LivenessVerdict, confidence: Optional[float] = 0.99, fail: bool = False):
        self._verdict, self._conf, self._fail = verdict, confidence, fail

    def check(self, sample: AudioSample, timeout_ms: int = 800) -> LivenessResult:
        if self._fail:
            return LivenessResult(LivenessVerdict.UNKNOWN, None, self.name, error="simulated provider failure")
        return LivenessResult(self._verdict, self._conf, self.name, latency_ms=50)


def resolve_liveness(
    provider: Optional[VoiceLivenessProvider],
    sample: Optional[AudioSample],
    is_active_phone_call: bool,
    min_confidence: float = 0.9,
) -> Optional[bool]:
    """The one function adapters call. Encodes every 'degrade to unknown' rule in one place.

    * no active call            → ``None`` (a liveness verdict without a call is meaningless; schema rejects it)
    * no provider / no sample   → ``None``
    * provider raises           → ``None`` (contract says it must not, but we don't trust that)
    * UNKNOWN / low confidence  → ``None``
    """
    if not is_active_phone_call or provider is None or sample is None:
        return None
    try:
        result = provider.check(sample)
    except Exception:  # noqa: BLE001 — a misbehaving provider must never break ingestion
        return None
    return result.to_tristate(min_confidence)
