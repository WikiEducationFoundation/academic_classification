import mwparserfromhell.nodes as mw_nodes
from mwparserfromhell.wikicode import Wikicode
import re


def select_words_from_wikicode(wikicode, first_n=None):
    first_words = []
    for node in wikicode.nodes:
        text = _extract_text_from_node(node)
        words = _extract_words_from_text(text)
        first_words.extend(words)
        if first_n is not None and len(first_words) > first_n:
            break
    if first_n is None:
        first_n = len(first_words)
    return first_words[:first_n]


def _extract_words_from_text(text):
    return [w for w in re.split(r'[/.,?\- \n\t\[\]\{\}]', str(text))
            if w != '']


def _extract_text_from_node(node):
    n_type = type(node)
    if n_type == mw_nodes.ExternalLink:
        text = str(node.title)
    elif n_type == mw_nodes.Heading:
        text = str(node.title)
    elif n_type == mw_nodes.Tag:
        text = str(node.contents)
    elif n_type == mw_nodes.Text:
        text = node.value
    elif n_type == mw_nodes.Wikilink:
        if node.text is not None:
            text = str(node.text)
        else:
            text = node.title
    else:
        text = ''
    return text.strip()
