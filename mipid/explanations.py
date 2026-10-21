
def attack_category(text):
    """Return a heuristic attack category and matched terms."""

    text = str(text).lower()

    checks = [
        (
            "Instruction override",
            [
                "ignore all previous",
                "disregard",
                "override",
                "पिछले सभी निर्देश",
                "पुराने निर्देश",
                "instrucciones anteriores",
                "rules hatao",
                "rules ignore",
            ],
        ),
        (
            "Prompt extraction",
            [
                "system prompt",
                "hidden prompt",
                "hidden instructions",
                "छिपे हुए निर्देश",
                "prompt del sistema",
            ],
        ),
        (
            "Data exfiltration",
            [
                "api key",
                "secrets",
                "private data",
                "confidential",
                "send all",
                "निजी डेटा",
                "datos privados",
            ],
        ),
        (
            "Role manipulation / jailbreak",
            [
                "unrestricted",
                "without restrictions",
                "no safety",
                "administrator",
                "act as root",
                "sin restricciones",
                "बिना किसी पाबंदी",
            ],
        ),
        (
            "Tool manipulation",
            [
                "execute the",
                "use the tool",
                "transfer the private file",
                "tool command",
            ],
        ),
    ]

    for category, terms in checks:
        matched = [term for term in terms if term in text]
        if matched:
            return category, matched

    return "No obvious pattern matched", []
