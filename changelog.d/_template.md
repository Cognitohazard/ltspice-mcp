{#- The section towncrier writes into CHANGELOG.md for a release, below the
    start marker. The "## [version] - date" line above it comes from
    title_format in pyproject.toml. Every type that has fragments becomes a
    "### Type" heading, in the order pyproject.toml lists the types, over one
    bullet per fragment; a type whose showcontent is false renders nothing.
    towncrier renders this with trim_blocks on, so the newline after a block
    tag is dropped, and the blank line before each heading is the one in the
    template. -#}
{%- set shown = namespace(any=false) -%}
{%- for section in sections -%}
{%- for category, definition in definitions.items() if category in sections[section] and definition.showcontent -%}
{%- set shown.any = true %}

### {{ definition.name }}

{% for text in sections[section][category] -%}
- {{ text }}
{% endfor -%}
{%- endfor -%}
{%- endfor -%}
{%- if not shown.any %}

No significant changes.
{% endif -%}
{{ "\n" }}
