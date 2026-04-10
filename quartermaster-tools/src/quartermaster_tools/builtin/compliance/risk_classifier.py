"""
RiskClassifierTool: Classify AI system risk level per EU AI Act Annex III.
"""

from __future__ import annotations

from typing import Any

from quartermaster_tools.base import AbstractTool
from quartermaster_tools.types import ToolDescriptor, ToolParameter, ToolResult

_HIGH_RISK_DOMAINS = frozenset({
    "healthcare",
    "law_enforcement",
    "education",
    "employment",
    "critical_infrastructure",
    "border_control",
    "justice",
    "democratic_process",
})

_DOMAIN_CATEGORIES: dict[str, str] = {
    "healthcare": "Annex III, Area 5: Access to essential services – healthcare",
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


class RiskClassifierTool(AbstractTool):
    """Classify AI system risk level per EU AI Act."""

    def name(self) -> str:
        return "risk_classifier"

    def version(self) -> str:
        return "1.0.0"

    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="system_description",
                description="Description of the AI system to classify.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="domain",
                description="Application domain of the AI system.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="uses_biometrics",
                description="Whether the system uses biometric identification.",
                type="boolean",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="uses_subliminal_techniques",
                description="Whether the system uses subliminal techniques to manipulate behaviour.",
                type="boolean",
                required=False,
                default=False,
            ),
            ToolParameter(
                name="targets_vulnerable_groups",
                description="Whether the system targets vulnerable groups for manipulation.",
                type="boolean",
                required=False,
                default=False,
            ),
        ]

    def info(self) -> ToolDescriptor:
        return ToolDescriptor(
            name=self.name(),
            short_description="Classify AI system risk level per EU AI Act.",
            long_description=(
                "Evaluates an AI system description against EU AI Act criteria "
                "to determine its risk classification (Unacceptable, High, "
                "Limited, or Minimal) and lists applicable obligations."
            ),
            version=self.version(),
            parameters=self.parameters(),
            is_local=True,
        )

    def run(self, **kwargs: Any) -> ToolResult:
        system_description: str = kwargs.get("system_description", "")
        if not system_description:
            return ToolResult(
                success=False,
                error="Parameter 'system_description' is required",
            )

        domain: str = kwargs.get("domain", "")
        if not domain:
            return ToolResult(success=False, error="Parameter 'domain' is required")

        uses_biometrics: bool = kwargs.get("uses_biometrics", False)
        uses_subliminal: bool = kwargs.get("uses_subliminal_techniques", False)
        targets_vulnerable: bool = kwargs.get("targets_vulnerable_groups", False)

        # Determine risk level
        risk_level: str
        reasoning: str

        if uses_subliminal:
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
        elif targets_vulnerable:
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
                    "System does not fall under any high-risk category and "
                    "has no specific risk flags."
                )

        category = _DOMAIN_CATEGORIES.get(domain, _DOMAIN_CATEGORIES["other"])
        obligations = _OBLIGATIONS.get(risk_level, [])

        return ToolResult(
            success=True,
            data={
                "risk_level": risk_level,
                "category": category,
                "obligations": obligations,
                "reasoning": reasoning,
            },
        )
