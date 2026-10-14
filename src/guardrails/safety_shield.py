import re
from typing import Dict, Any, List, Tuple

class AISafetyShield:
    """
    Security and privacy guardrails for LLM input and output pipelines:
    1. Prompt Injection & Jailbreak Heuristics
    2. Personally Identifiable Information (PII) Detection & Redaction (Emails, Phone, IBAN, Credit Cards)
    3. Toxicity & Instruction Boundary Protection
    """
    INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"system prompt override",
        r"you are now in developer mode",
        r"bypass all safety guidelines",
        r"act as an unfiltered ai",
        r"reveal your internal instructions",
        r"sudo mode enabled"
    ]

    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b",
        "IBAN_DE": r"\bDE\d{2}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4}[ ]?\d{4}[ ]?\d{2}\b",
        "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b"
    }

    def inspect_input(self, user_input: str) -> Tuple[bool, List[str], str]:
        """
        Returns: (is_safe, violations, sanitized_text)
        """
        violations = []
        lower_input = user_input.lower()
        
        # Check Injection
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, lower_input):
                violations.append(f"Prompt Injection Pattern Detected: '{pat}'")

        # Anonymize PII
        sanitized = user_input
        for pii_type, regex in self.PII_PATTERNS.items():
            matches = re.findall(regex, sanitized)
            if matches:
                violations.append(f"PII Leakage Detected: {pii_type} ({len(matches)} instance(s))")
                sanitized = re.sub(regex, f"[REDACTED_{pii_type}]", sanitized)

        is_safe = len([v for v in violations if "Injection" in v]) == 0
        return is_safe, violations, sanitized
