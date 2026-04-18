# Concept Extraction Request

You are extracting candidate concepts from a source document for a knowledge base workflow.

## Title
{{title}}

## Maximum concepts
{{max_concepts}}

## Summary
{{summary}}

## Document
{{body}}

## Task
Identify the most important concepts introduced, defined, or materially discussed in the source.
Prefer durable concepts over incidental details.
Include aliases when the source uses alternate names, abbreviations, or common shorthand.
Use `definition_hint` for a short gloss grounded in the source.
Use `source_anchors` to point back into the source with stable heading anchors when possible and short supporting quotes.

Return only valid JSON in this shape:
{
  "concepts": [
    {
      "name": "Concept name",
      "aliases": ["Alias"],
      "definition_hint": "Short definition",
      "source_anchors": [
        {
          "heading_anchor": "optional-heading-id",
          "quote": "Short supporting quote"
        }
      ]
    }
  ]
}
