
import re

CLEAN_TEXT_PATTERN = re.compile(r"[\r\n]")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return CLEAN_TEXT_PATTERN.sub("", text)
