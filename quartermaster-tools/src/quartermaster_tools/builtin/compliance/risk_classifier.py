"""
risk_classifier: Classify AI system risk level per EU AI Act Annex III.
"""

from __future__ import annotations

from quartermaster_tools.decorator import tool

_HIGH_RISK_DOMAINS = frozenset(
    {
        "healthcare",
        "law_enforcement",
        "education",
        "employment",
        "critical_infrastructure",
        "border_control",
        "justice",
        "democratic_process",
    }
)

_DOMAIN_CATEGORIES: dict[str, str] = {
    "healthcare": "Annex III, Area 5: Access to essential services -- healthcare",
    "law_enforcement": "Annex III, Area 6: Law enforcement",
    "education": "Annex III, Area 3: Education and vocational training",
    "employment": "Annex III, Area 4: Employment and workers management",
    "critical_infrastructure": "Annex III, Area 2: Critical infrastructure",
    "border_control": "Annex III, Area 7: Migration, asylum and border control",
    "justice": "Annex III, Area 8: Administration of justice and democratic processes",
    "democratic_process": "Annex III, Area 8: Administration of justice and democratic processes",
    "other": "Not classified under Annex III high-risk areas",
}

_OBLIGATIONS: dict[str, list[str]] = {
    "UNACCEPTABLE": [
        "System is PROHIBITED under Article 5 of the EU AI Act.",
        "Must not be placed on the market, put into service, or used.",
    ],
    "HIGH": [
        "Risk management system (Art. 9)",
        "Data and data governance (Art. 10)",
        "Technical documentation (Art. 11)",
        "Record-keeping and logging (Art. 12)",
        "Transparency and information to deployers (Art. 13)",
        "Human oversight measures (Art. 14)",
        "Accuracy, robustness and cybersecurity (Art. 15)",
        "Quality management system (Art. 17)",
        "Conformity assessment before market placement (Art. 43)",
        "EU declaration of conformity (Art. 47)",
        "CE marking (Art. 48)",
        "Registration in EU database (Art. 49)",
        "Post-market monitoring (Art. 61)",
        "Serious incident reporting (Art. 62)",
    ],
    "LIMITED": [
        "Transparency obligations (Art. 52)",
        "Users must be informed they are interacting with an AI system",
        "Emotion recognition / biometric categorisation systems must inform subjects",
        "AI-generated content must be labelled as such",
    ],
    "MINIMAL": [
        "Voluntary codes of conduct encouraged (Art. 69)",
        "No mandatory requirements beyond general product safety laws",
    ],
}


@tool()
def risk_classifier(
    system_description: str,
    domain: str,
    uses_biometrics: bool = False,
    uses_subliminal_techniques: bool = False,
    targets_vulnerable_groups: bool = False,
) -> dict:
    """Classify AI system risk level per EU AI Act.

    Evaluates an AI system description against EU AI Act criteria
    to determine its risk classification (Unacceptable, High,
    Limited, or Minimal) and lists applicable obligations.

    Args:
        system_description: Description of the AI system to classify.
        domain: Application domain of the AI system.
        uses_biometrics: Whether the system uses biometric identification.
        uses_subliminal_techniques: Whether the system uses subliminal techniques to manipulate behaviour.
        targets_vulnerable_groups: Whether the system targets vulnerable groups for manipulation.
    """
    if not system_description:
        raise ValueError("Parameter 'system_description' is required")
    if not domain:
        raise ValueError("Parameter 'domain' is required")

    # Determine risk level
    risk_level: str
    reasoning: str

    if uses_subliminal_techniques:
        risk_level = "UNACCEPTABLE"
        reasoning = (
            "System uses subliminal techniques beyond a person's consciousness "
            "to materially distort behaviour, which is prohibited under Art. 5(1)(a)."
        )
    elif uses_biometrics and domain == "law_enforcement":
        risk_level = "UNACCEPTABLE"
        reasoning = (
            "Real-time remote biometric identification in publicly accessible "
            "spaces for law enforcement is prohibited under Art. 5(1)(d)."
        )
    elif targets_vulnerable_groups:
        risk_level = "UNACCEPTABLE"
        reasoning = (
            "System exploits vulnerabilities of specific groups due to age, "
            "disability, or social/economic situation to distort behaviour, "
            "prohibited under Art. 5(1)(b)."
        )
    elif domain in _HIGH_RISK_DOMAINS:
        risk_level = "HIGH"
        reasoning = (
            f"System operates in the '{domain}' domain which is classified "
            f"as high-risk under Annex III of the EU AI Act."
        )
    elif uses_biometrics:
        risk_level = "HIGH"
        reasoning = (
            "System uses biometric identification which falls under "
            "Annex III, Area 1: Biometric identification and categorisation."
        )
    else:
        # Check if it has transparency obligations (default to LIMITED for non-other)
        if domain != "other":
            risk_level = "LIMITED"
            reasoning = (
                f"System in domain '{domain}' has transparency obligations "
                "but does not meet high-risk criteria."
            )
        else:
            risk_level = "MINIMAL"
            reasoning = (
                "System does not fall under any high-risk category and has no specific risk flags."
            )

    category = _DOMAIN_CATEGORIES.get(domain, _DOMAIN_CATEGORIES["other"])
    obligations = _OBLIGATIONS.get(risk_level, [])

    return {
        "risk_level": risk_level,
        "category": category,
        "obligations": obligations,
        "reasoning": reasoning,
    }
