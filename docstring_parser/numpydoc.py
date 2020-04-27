"""Numpydoc-style docstring parsing.

See Also
--------
https://numpydoc.readthedocs.io/en/latest/format.html
"""

import inspect
import re
import typing as T
import itertools
from dataclasses import dataclass

from .common import (
    Docstring,
    DocstringMeta,
    DocstringParam,
    DocstringRaises,
    DocstringReturns,
    DocstringDeprecated,
)


def _pairwise(iterable: T.Iterable, end=None) -> T.Iterable:
    a, b = itertools.tee(iterable)
    next(b, None)
    return itertools.zip_longest(a, b, fillvalue=end)


def _clean_str(string: str) -> T.Optional[str]:
    string = string.strip()
    if len(string) > 0:
        return string


KV_REGEX = re.compile(r'^[^\s].*$', flags=re.M)

PARAM_KEY_REGEX = re.compile(r'^(?P<name>.*?)(?:\s*:\s*(?P<type>.*?))?$')

PARAM_OPTIONAL_REGEX = re.compile(r'(?P<type>.*?)(?:, optional|\(optional\))$')

# numpydoc format has no formal grammar for this,
# but we can make some educated guesses...
PARAM_DEFAULT_REGEX = re.compile(
    r'[Dd]efault(?: is | = |: |s to |)\s*(?P<value>[\w\-\.]+)'
)

RETURN_KEY_REGEX = re.compile(r'^(?:(?P<name>.*?)\s*:\s*)?(?P<type>.*?)$')


@dataclass
class Section:
    title: str
    key: str

    @property
    def title_pattern(self) -> str:
        return r"^({})\s*?\n{}\s*$".format(self.title, '-' * len(self.title))

    def parse(self, text: str) -> T.Iterable[DocstringMeta]:
        yield DocstringMeta([self.key], description=_clean_str(text))


class _KVSection(Section):
    def _parse_item(self, key: str, value: str) -> DocstringMeta:
        pass

    def parse(self, text: str) -> T.Iterable[DocstringMeta]:
        for match, next_match in _pairwise(KV_REGEX.finditer(text)):
            start = match.end()
            end = next_match.start() if next_match is not None else None
            value = text[start:end]
            yield self._parse_item(key=match.group(),
                                   value=inspect.cleandoc(value))


class SphinxSection(Section):
    @property
    def title_pattern(self) -> str:
        return r"^\.\.\s*({})\s*::".format(self.title)


class ParamSection(_KVSection):
    def _parse_item(self, key: str, value: str) -> DocstringParam:
        m = PARAM_KEY_REGEX.match(key)
        arg_name = type_name = is_optional = None
        if m is not None:
            arg_name, type_name = m.group('name'), m.group('type')
            if type_name is not None:
                optional_match = PARAM_OPTIONAL_REGEX.match(type_name)
                if optional_match is not None:
                    type_name = optional_match.group('type')
                    is_optional = True
                else:
                    is_optional = False

        default = None
        if len(value) > 0:
            default_match = PARAM_DEFAULT_REGEX.search(value)
            if default_match is not None:
                default = default_match.group('value')

        return DocstringParam(
            args=[self.key, arg_name],
            description=_clean_str(value),
            arg_name=arg_name,
            type_name=type_name,
            is_optional=is_optional,
            default=default,
        )


class RaisesSection(_KVSection):
    def _parse_item(self, key: str, value: str) -> DocstringRaises:
        return DocstringRaises(
            args=[self.key, key],
            description=_clean_str(value),
            type_name=key if len(key) > 0 else None,
        )


class ReturnsSection(_KVSection):
    is_generator = False

    def _parse_item(self, key: str, value: str) -> DocstringReturns:
        m = RETURN_KEY_REGEX.match(key)
        if m is not None:
            return_name, type_name = m.group('name'), m.group('type')
        else:
            return_name = type_name = None

        return DocstringReturns(
            args=[self.key],
            description=_clean_str(value),
            type_name=type_name,
            is_generator=self.is_generator,
            return_name=return_name,
        )


class YieldsSection(ReturnsSection):
    is_generator = True


class DeprecationSection(SphinxSection):
    def parse(self, text: str) -> T.Iterable[DocstringDeprecated]:
        version, desc, *_ = text.split(sep='\n', maxsplit=1) + [None, None]

        if desc is not None:
            desc = _clean_str(inspect.cleandoc(desc))

        yield DocstringDeprecated(
            args=[self.key],
            description=desc,
            version=_clean_str(version),
        )


DEFAULT_SECTIONS = {s.title: s for s in [
    ParamSection("Parameters", "param"),
    ParamSection("Params", "param"),
    ParamSection("Arguments", "param"),
    ParamSection("Args", "param"),
    ParamSection("Other Parameters", "other_param"),
    ParamSection("Other Params", "other_param"),
    ParamSection("Other Arguments", "other_param"),
    ParamSection("Other Args", "other_param"),
    ParamSection("Receives", "receives"),
    ParamSection("Receive", "receives"),
    RaisesSection("Raises", "raises"),
    RaisesSection("Raise", "raises"),
    RaisesSection("Warns", "warns"),
    RaisesSection("Warn", "warns"),
    ParamSection("Attributes", "attribute"),
    ParamSection("Attribute", "attribute"),
    ReturnsSection("Returns", "returns"),
    ReturnsSection("Return", "returns"),
    YieldsSection("Yields", "yields"),
    YieldsSection("Yield", "yields"),
    Section("Examples", "examples"),
    Section("Example", "examples"),
    Section("Warnings", "warnings"),
    Section("Warning", "warnings"),
    Section("See Also", "see_also"),
    Section("Related", "see_also"),
    Section("Notes", "notes"),
    Section("Note", "notes"),
    Section("References", "references"),
    Section("Reference", "references"),
    DeprecationSection("deprecated", "deprecation"),
]}


class NumpyParser:
    def __init__(
        self, sections: T.Optional[T.Dict[str, Section]] = None
    ):
        """Setup sections.

        :param sections: Recognized sections or None to defaults.
        """
        sections = sections or DEFAULT_SECTIONS
        self.sections = sections.copy()
        self._setup()

    def _setup(self):
        self.titles_re = re.compile(
            r"|".join(s.title_pattern for s in self.sections.values()),
            flags=re.M
        )

    def add_section(self, section: Section):
        """Add or replace a section.

        :param section: The new section.
        """

        self.sections[section.title] = section
        self._setup()

    def parse(self, text: str) -> Docstring:
        """Parse the numpy-style docstring into its components.

        :returns: parsed docstring
        """
        ret = Docstring()
        if not text:
            return ret

        # Clean according to PEP-0257
        text = inspect.cleandoc(text)

        # Find first title and split on its position
        match = self.titles_re.search(text)
        if match:
            desc_chunk = text[:match.start()]
            meta_chunk = text[match.start():]
        else:
            desc_chunk = text
            meta_chunk = ""

        # Break description into short and long parts
        parts = desc_chunk.split("\n", 1)
        ret.short_description = parts[0] or None
        if len(parts) > 1:
            long_desc_chunk = parts[1] or ""
            ret.blank_after_short_description = long_desc_chunk.startswith(
                "\n"
            )
            ret.blank_after_long_description = long_desc_chunk.endswith("\n\n")
            ret.long_description = long_desc_chunk.strip() or None

        for match, nextmatch in _pairwise(self.titles_re.finditer(meta_chunk)):
            title = next(g for g in match.groups() if g is not None)
            factory = self.sections[title]

            # section chunk starts after the header,
            # ends at the start of the next header
            start = match.end()
            end = nextmatch.start() if nextmatch is not None else None
            ret.meta.extend(factory.parse(meta_chunk[start:end]))

        return ret


def parse(text: str) -> Docstring:
    """Parse the numpy-style docstring into its components.

    :returns: parsed docstring
    """
    return NumpyParser().parse(text)