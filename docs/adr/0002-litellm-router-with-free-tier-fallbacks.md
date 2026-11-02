# ADR 0002: LiteLLM Router with Groq -> Gemini fallbacks

**Decision:** one `LLMGateway` protocol; production adapter wraps `litellm.Router` (SDK mode) with ordered
fallbacks per tier (`smart`, `fast`); proxy mode is supported via `TRIPMATE_LITELLM_PROXY_URL`.
**Why:** free tiers rate limit aggressively; a second provider turns hard failures into a quality-neutral fallback.
**Structured output** does not rely on any provider's JSON mode: schema in prompt, lenient parse, strict Pydantic
validation, one repair turn with the validation error. **Consequences:** behaves the same on any model; costs at
most one extra call on bad output; Router owns transport retries so we never retry twice.
