"""What each model costs, and how to price one API call.

Prices are US dollars per million tokens, taken from the providers' public
pages. Update them when a provider changes a price or ships a new model. A
model that is missing from this table is priced at UNKNOWN_PRICE, which is
higher than any real model, so an out-of-date table makes the budget stricter
rather than looser.
"""

MILLION = 1_000_000

# (input price, output price) in US dollars per million tokens.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-6-astra": (10.0, 50.0),
    "gpt-5.6-sol": (4.0, 20.0),
    "gpt-5.6-terra": (2.0, 12.0),
    "gpt-5.6-luna": (0.20, 1.20),
}

# Used for any model not listed above. Deliberately expensive.
UNKNOWN_PRICE = (20.0, 100.0)


def price_of(model: str) -> tuple[float, float]:
    """Return the input and output price per million tokens for a model."""
    for name, price in PRICES.items():
        if model.startswith(name):
            return price
    return UNKNOWN_PRICE


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of one completed call."""
    price_in, price_out = price_of(model)
    return (input_tokens * price_in + output_tokens * price_out) / MILLION


def worst_case_cost(model: str, prompt_chars: int, max_output_tokens: int) -> float:
    """Return the most a call could cost before it is sent.

    Token counts are estimated from character counts at three characters per
    token, which is lower than the usual four and therefore over-estimates the
    number of tokens. Output is priced at the full cap even though most replies
    are shorter. Both choices make the estimate an upper bound, so the budget
    check never lets a call through that could overshoot.
    """
    price_in, price_out = price_of(model)
    estimated_input = prompt_chars / 3
    return (estimated_input * price_in + max_output_tokens * price_out) / MILLION
