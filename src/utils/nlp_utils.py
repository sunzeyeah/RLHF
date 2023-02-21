
import re

CLEAN_TEXT_PATTERN = re.compile(r"[\r\n]")


def clean_text(text):
    return CLEAN_TEXT_PATTERN.sub("", text)
