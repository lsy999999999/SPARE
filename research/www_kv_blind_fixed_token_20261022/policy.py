"""KV-blind selector with a fixed, one-chunk token-saving threshold."""
from research.www_kv_blind_deferral_20261021 import policy as blind

FIXED_TOKEN_SAVING = 256


def select(rows, budget, chunk, free, total, block_size, concurrency,
           max_model_len, threshold=1, recent=None, options=None):
    assert chunk == FIXED_TOKEN_SAVING
    assert (options or {}).get("mode") == "kv_blind_fixed_token"
    # The blind selector's threshold is chunk minus the supplied arrival rate.
    # Zero fixes it at one chunk without exposing KV or arrival state.
    result = blind.select(rows, budget, chunk, free, total, block_size,
                          concurrency, max_model_len, threshold=threshold,
                          recent={"arrival_rate": 0.0},
                          options={"mode": "kv_blind"})
    assert result["threshold"] == FIXED_TOKEN_SAVING
    return result
