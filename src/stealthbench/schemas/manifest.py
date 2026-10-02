"""Canonical manifest identity and sample traceability.

These functions let a stored sample be traced back to the exact manifest and prompt
that produced it. They validate before hashing, so an invalid manifest fails at this
boundary rather than becoming a durable artifact with a plausible-looking hash.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import Field

from stealthbench.schemas.campaign import CampaignManifest
from stealthbench.schemas.hashing import content_digest
from stealthbench.schemas.results import Message, ModelRequest, ResultModel

#: Version tag mixed into every hash. Bump when the hashing contract itself changes,
#: so an old and a new digest can never be confused.
MANIFEST_HASH_VERSION: Final[str] = "manifest-hash-v1"


class PromptRef(ResultModel):
    """A dispatchable prompt artifact, addressed by content.

    Holds only what may be sent to a provider. There is no field here for a gold
    answer, so a prompt hash can never be computed over evaluator-only data.
    """

    messages: tuple[Message, ...]
    max_output_tokens: int = Field(gt=0)
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stream: bool = False
    #: Stop sequences are part of what gets dispatched. Omitting them here would let
    #: two materially different requests hash to the same prompt identity.
    stop: tuple[str, ...] = ()

    @classmethod
    def from_request(cls, request: ModelRequest) -> PromptRef:
        """The addressable form of a request.

        Round-trips every dispatchable field. ``to_request()`` of the result is equal
        to ``request``, which is what keeps a prompt hash meaningful.
        """
        return cls(
            messages=request.messages,
            max_output_tokens=request.max_output_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            seed=request.seed,
            stream=request.stream,
            stop=request.stop,
        )

    def to_request(self) -> ModelRequest:
        """The request this prompt dispatches as."""
        return ModelRequest(
            messages=self.messages,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
            stream=self.stream,
            stop=self.stop,
        )


def _validated(manifest: CampaignManifest | dict[str, Any]) -> CampaignManifest:
    if isinstance(manifest, CampaignManifest):
        return manifest
    return CampaignManifest.model_validate(manifest)


def _validated_prompt(prompt: PromptRef | dict[str, Any]) -> PromptRef:
    if isinstance(prompt, PromptRef):
        return prompt
    return PromptRef.model_validate(prompt)


def manifest_hash(manifest: CampaignManifest | dict[str, Any]) -> str:
    """Canonical sha256 of a validated manifest.

    Raises if the manifest is invalid. Hashing an unvalidated document would let a
    manifest carrying a credential value acquire a durable, seemingly valid
    identity, which is exactly what the G01 gate forbids.
    """
    model = _validated(manifest)
    return content_digest({"hash_version": MANIFEST_HASH_VERSION, "manifest": model.to_json_dict()})


def prompt_hash(prompt: PromptRef | dict[str, Any]) -> str:
    """Canonical sha256 of the request that will actually be dispatched."""
    model = _validated_prompt(prompt)
    return content_digest(
        {
            "hash_version": MANIFEST_HASH_VERSION,
            "prompt": model.to_request().model_dump(mode="json"),
        }
    )


def trace_key(
    manifest: CampaignManifest | dict[str, Any], prompt: PromptRef | dict[str, Any]
) -> str:
    """Bind one manifest to one prompt.

    Every stored sample carries this, so "which configuration produced this score"
    becomes a lookup rather than a reconstruction.
    """
    return content_digest(
        {
            "hash_version": MANIFEST_HASH_VERSION,
            "manifest_hash": manifest_hash(manifest),
            "prompt_hash": prompt_hash(prompt),
        }
    )
