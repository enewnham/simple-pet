from pathlib import Path
from xml.sax.saxutils import escape

from jinja2 import Environment, FileSystemLoader, StrictUndefined

environment = Environment(
    loader=FileSystemLoader(Path(__file__).with_name("templates")),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)


def xmltext(value: object) -> str:
    return escape(str(value))


environment.filters["xmltext"] = xmltext


def render(name: str, **values: object) -> str:
    template = environment.get_template(name)
    return template.render(**values).strip()
