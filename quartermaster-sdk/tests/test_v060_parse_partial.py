"""v0.6.0 — ``qm.parse_partial(text, schema)`` progressive-degradation parser.

Tests are grouped by which strategy they exercise: full-JSON-validated,
JSON-extracted-per-field, line-scan, give-up.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from quartermaster_sdk import PartialResult, parse_partial


class CustomerEnrichment(BaseModel):
    company_name: str = ""
    country: str = ""
    website: str = ""
    emails: list[str] = Field(default_factory=list)
    phones: list[str] = Field(default_factory=list)
    industry: str = ""
    summary: str = ""
    sources: list[str] = Field(default_factory=list)


# ── Strategy 1: full_json_validated ──────────────────────────────────


class TestFullJsonValidated:
    def test_clean_json_body(self) -> None:
        text = """{
          "company_name": "Gorenje",
          "country": "Slovenia",
          "website": "https://gorenje.com",
          "emails": ["info@gorenje.si"],
          "phones": ["01 320 92 92"],
          "industry": "Appliances",
          "summary": "Slovenian appliances maker.",
          "sources": ["https://gorenje.com"]
        }"""
        result = parse_partial(text, CustomerEnrichment)
        assert result.strategy == "full_json_validated"
        assert result.data["company_name"] == "Gorenje"
        assert result.data["emails"] == ["info@gorenje.si"]
        assert result.missing_fields == []

    def test_preamble_before_json(self) -> None:
        """The helper walks for the LAST parseable object — a preamble
        doesn't defeat the happy path."""
        text = (
            "Here's the profile you asked for:\n"
            '{"company_name": "X", "country": "SI", "website": "", "emails": [], '
            '"phones": [], "industry": "", "summary": "", "sources": []}'
        )
        result = parse_partial(text, CustomerEnrichment)
        assert result.strategy == "full_json_validated"
        assert result.data["company_name"] == "X"

    def test_empty_list_defaults_respected(self) -> None:
        """Pydantic default_factory for emails / phones / sources means
        ``[]`` is valid where the JSON omits them AND the model has
        defaults. ``CustomerEnrichment`` has string defaults ``""`` so
        omission is allowed."""
        text = '{"company_name": "X"}'
        result = parse_partial(text, CustomerEnrichment)
        assert result.strategy == "full_json_validated"
        assert result.data["company_name"] == "X"
        assert result.data["emails"] == []


# ── Strategy 2: json_extracted ───────────────────────────────────────


class RequiredNameModel(BaseModel):
    """Has a required field — forces the happy path to fail when name
    is missing, so we fall through to json_extracted."""

    name: str
    country: str = ""
    emails: list[str] = Field(default_factory=list)


class TestJsonExtracted:
    def test_missing_required_drops_to_partial(self) -> None:
        """The JSON is valid but missing the required ``name`` field.
        Strategy 1 rejects; strategy 2 salvages ``country`` and ``emails``."""
        text = '{"country": "SI", "emails": ["x@y.z"]}'
        result = parse_partial(text, RequiredNameModel)
        assert result.strategy == "json_extracted"
        assert result.data == {"country": "SI", "emails": ["x@y.z"]}
        assert result.missing_fields == ["name"]


# ── Strategy 3: line_scan ────────────────────────────────────────────


class TestLineScan:
    def test_research_note_style(self) -> None:
        """The pattern matches what Gemma writes in the research note:
        plain ``Key: value`` lines, no JSON at all."""
        text = (
            "Company: MindMade\n"
            "Country: Slovenia\n"
            "Website: https://mindmade.ai/\n"
            'Emails: ["info@mindmade.ai"]\n'
            "Industry: Computer activities\n"
            "Summary: Slovenian software company.\n"
            "Sources:\n"
            "  - https://mindmade.ai/\n"
            "  - https://companywall.si/podjetje/mindmade-doo/MMCckxvq\n"
        )
        result = parse_partial(text, CustomerEnrichment)
        assert result.strategy == "line_scan"
        assert result.data["country"] == "Slovenia"
        assert result.data["website"] == "https://mindmade.ai/"
        assert result.data["emails"] == ["info@mindmade.ai"]

    def test_star_bolded_keys(self) -> None:
        """Markdown-bolded keys (``**Company**: x``) still match."""
        text = "**Company**: MindMade\n**Country**: Slovenia"
        # Field is `company_name`, and the text has "Company" — the
        # line-scan normalisation doesn't try to synonym-match. This
        # assertion documents that constraint: rename your field to
        # match LLM output if you want line_scan to pick it up.
        result = parse_partial(text, CustomerEnrichment)
        assert result.data.get("country") == "Slovenia"

    def test_not_found_placeholder_treated_as_absent(self) -> None:
        """When the LLM writes ``Country: not found`` we DON'T land the
        literal ``"not found"`` string in data — the field goes to
        missing instead."""
        text = "Country: not found\nIndustry: SaaS"
        result = parse_partial(text, CustomerEnrichment)
        assert "country" not in result.data
        assert "country" in result.missing_fields
        assert result.data.get("industry") == "SaaS"

    def test_equals_separator(self) -> None:
        text = "country = SI\nindustry = retail"
        result = parse_partial(text, CustomerEnrichment)
        assert result.data["country"] == "SI"
        assert result.data["industry"] == "retail"

    def test_coerces_list_values(self) -> None:
        text = 'emails: ["a@b.c", "d@e.f"]'
        result = parse_partial(text, CustomerEnrichment)
        assert result.data["emails"] == ["a@b.c", "d@e.f"]

    def test_coerces_bool_numeric_values(self) -> None:
        """Line scan is provider-agnostic — bool / int / float still parse."""

        class Flags(BaseModel):
            active: bool = False
            count: int = 0

        text = "active: true\ncount: 42"
        result = parse_partial(text, Flags)
        assert result.data == {"active": True, "count": 42}


# ── Strategy 4: none ─────────────────────────────────────────────────


class TestGiveUp:
    def test_empty_input(self) -> None:
        result = parse_partial("", CustomerEnrichment)
        assert result.strategy == "none"
        assert result.data == {}
        assert "company_name" in result.missing_fields

    def test_junk_with_no_matches(self) -> None:
        text = "Lorem ipsum dolor sit amet. This paragraph has no keys we recognise."
        result = parse_partial(text, CustomerEnrichment)
        assert result.strategy == "none"
        assert result.data == {}

    def test_raw_output_preserved_on_give_up(self) -> None:
        text = "uneparsable"
        result = parse_partial(text, CustomerEnrichment)
        assert result.raw_output == "uneparsable"


# ── Schema shapes: JSON-schema dicts, dataclasses ────────────────────


class TestSchemaShapes:
    def test_dict_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "year": {"type": "integer"},
            },
        }
        text = "Name: Acme Inc\nYear: 2024"
        result = parse_partial(text, schema)
        assert result.data == {"name": "Acme Inc", "year": 2024}

    def test_dataclass_via_annotations(self) -> None:
        from dataclasses import dataclass

        @dataclass
        class AnnotatedThing:
            title: str = ""
            count: int = 0

        text = '{"title": "x", "count": 3}'
        result = parse_partial(text, AnnotatedThing)
        # Not a Pydantic model → strategy 1 skipped; strategy 2 extracts.
        assert result.strategy == "json_extracted"
        assert result.data == {"title": "x", "count": 3}


# ── Result dataclass ─────────────────────────────────────────────────


class TestPartialResult:
    def test_round_trip_field_access(self) -> None:
        r = PartialResult(
            data={"x": 1},
            missing_fields=["y"],
            raw_output="hello",
            strategy="line_scan",
        )
        assert r.data == {"x": 1}
        assert r.missing_fields == ["y"]
        assert r.raw_output == "hello"
        assert r.strategy == "line_scan"

    def test_exported_from_top_level(self) -> None:
        import quartermaster_sdk

        assert "parse_partial" in quartermaster_sdk.__all__
        assert "PartialResult" in quartermaster_sdk.__all__
