"""Optional adapter negotiation; absence preserves the original configuration."""


def negotiate(adapter, config, requested):
    method = getattr(adapter, "optimize", None)
    if method is None:
        return config, {
            name: {
                "status": "unavailable",
                "reason": "adapter has no optimization interface",
            }
            for name in requested
        }
    return method(config, requested)
