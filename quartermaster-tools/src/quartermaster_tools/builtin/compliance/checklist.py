"""
compliance_checklist: Generate EU AI Act compliance checklists.
"""

from __future__ import annotations

from quartermaster_tools.decorator import tool

_CHECKLISTS: dict[str, list[dict[str, str]]] = {
    "UNACCEPTABLE": [
        {
            "article": "Art. 5",
            "requirement": "Verify system does not fall under prohibited AI practices",
            "status": "pending",
        },
        {
            "article": "Art. 5(1)(a)",
            "requirement": "Confirm no subliminal manipulation techniques are used",
            "status": "pending",
        },
        {
            "article": "Art. 5(1)(b)",
            "requirement": "Confirm no exploitation of vulnerable groups",
            "status": "pending",
        },
        {
            "article": "Art. 5(1)(c)",
            "requirement": "Confirm no social scoring by public authorities",
            "status": "pending",
        },
        {
            "article": "Art. 5(1)(d)",
            "requirement": "Confirm no prohibited real-time remote biometric identification",
            "status": "pending",
        },
        {
            "article": "Art. 5",
            "requirement": "System must be withdrawn from the market if prohibited",
            "status": "pending",
        },
    ],
    "HIGH": [
        {
            "article": "Art. 9",
            "requirement": "Establish and maintain a risk management system throughout the AI system lifecycle",
            "status": "pending",
        },
        {
            "article": "Art. 10",
            "requirement": "Ensure training, validation and testing data meets quality criteria and governance requirements",
            "status": "pending",
        },
        {
            "article": "Art. 11",
            "requirement": "Prepare and maintain technical documentation demonstrating compliance",
            "status": "pending",
        },
        {
            "article": "Art. 12",
            "requirement": "Implement automatic recording of events (logging) for traceability",
            "status": "pending",
        },
        {
            "article": "Art. 13",
            "requirement": "Design system to be sufficiently transparent for deployers to interpret output",
            "status": "pending",
        },
        {
            "article": "Art. 14",
            "requirement": "Enable effective human oversight during the AI system's use",
            "status": "pending",
        },
        {
            "article": "Art. 15",
            "requirement": "Achieve appropriate levels of accuracy, robustness and cybersecurity",
            "status": "pending",
        },
        {
            "article": "Art. 17",
            "requirement": "Implement a quality management system with documented policies and procedures",
            "status": "pending",
        },
        {
            "article": "Art. 43",
            "requirement": "Complete conformity assessment before placing system on the market",
            "status": "pending",
        },
        {
            "article": "Art. 47",
            "requirement": "Draw up EU declaration of conformity",
            "status": "pending",
        },
        {
            "article": "Art. 48",
            "requirement": "Affix CE marking to the AI system",
            "status": "pending",
        },
        {
            "article": "Art. 49",
            "requirement": "Register system in the EU database for high-risk AI systems",
            "status": "pending",
        },
        {
            "article": "Art. 61",
            "requirement": "Establish post-market monitoring system",
            "status": "pending",
        },
        {
            "article": "Art. 62",
            "requirement": "Report serious incidents to market surveillance authorities",
            "status": "pending",
        },
    ],
    "LIMITED": [
        {
            "article": "Art. 52(1)",
            "requirement": "Inform users they are interacting with an AI system",
            "status": "pending",
        },
        {
            "article": "Art. 52(2)",
            "requirement": "Inform subjects of emotion recognition or biometric categorisation systems",
            "status": "pending",
        },
        {
            "article": "Art. 52(3)",
            "requirement": "Label AI-generated or manipulated content (deepfakes) as such",
            "status": "pending",
        },
        {
            "article": "Art. 52",
            "requirement": "Provide clear and distinguishable disclosure of AI involvement",
            "status": "pending",
        },
    ],
    "MINIMAL": [
        {
            "article": "Art. 69",
            "requirement": "Consider adopting voluntary codes of conduct",
            "status": "pending",
        },
        {
            "article": "General",
            "requirement": "Comply with existing EU product safety legislation where applicable",
            "status": "pending",
        },
        {
            "article": "General",
            "requirement": "Consider implementing transparency measures voluntarily",
            "status": "pending",
        },
    ],
}


@tool()
def compliance_checklist(risk_level: str, system_type: str = "") -> dict:
    """Generate EU AI Act compliance checklist.

    Generates a comprehensive compliance checklist based on
    EU AI Act requirements for the specified risk level.
    Each checklist item references the relevant article.

    Args:
        risk_level: Risk level: UNACCEPTABLE, HIGH, LIMITED, or MINIMAL.
        system_type: Optional system type for context.
    """
    risk_level = risk_level.upper() if risk_level else ""
    if not risk_level:
        raise ValueError("Parameter 'risk_level' is required")

    valid_levels = {"UNACCEPTABLE", "HIGH", "LIMITED", "MINIMAL"}
    if risk_level not in valid_levels:
        raise ValueError(
            f"Invalid risk_level: {risk_level!r}. Must be one of {sorted(valid_levels)}."
        )

    checklist = [item.copy() for item in _CHECKLISTS[risk_level]]

    return {
        "risk_level": risk_level,
        "checklist": checklist,
        "total_items": len(checklist),
    }
