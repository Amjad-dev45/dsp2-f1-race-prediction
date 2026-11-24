"""Teams Workflows webhook messages (Adaptive Cards), shared by the DAGs."""


def adaptive_card(title: str, facts: list[tuple[str, str]], color: str) -> dict:
    """A Teams message with a coloured title and a list of facts.

    color: an Adaptive Card text colour, e.g. "Attention" (red), "Warning"
    (orange), "Good" (green). Teams Workflows (Power Automate) webhooks
    expect this format; MessageCard only worked with the retired connectors.
    """
    card = {
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {
                "type": "TextBlock",
                "text": title,
                "weight": "Bolder",
                "size": "Large",
                "color": color,
                "wrap": True,
            },
            {
                "type": "FactSet",
                "facts": [{"title": t, "value": v} for t, v in facts],
            },
        ],
    }
    return {
        "type": "message",
        "attachments": [
            {"contentType": "application/vnd.microsoft.card.adaptive", "content": card}
        ],
    }


def send(webhook_url: str, message: dict, log_prefix: str) -> None:
    """Post a message; without a webhook it is only logged. A rejected call
    raises, so the task fails instead of the alert silently disappearing."""
    import requests

    if not webhook_url:
        print(f"{log_prefix} no webhook set - would send: {message}")
        return
    resp = requests.post(webhook_url, json=message, timeout=10)
    resp.raise_for_status()
    print(f"{log_prefix} Teams alert sent (status {resp.status_code})")
