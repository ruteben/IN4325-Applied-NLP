"""This file contains all functions to extract the basic content related features"""
from nltk.corpus import wordnet as wn


# Note: wordnet corpus should be downloaded. In order to download it run: nltk.download()


def post_img(p):
    """Get the image of a post

    :param p: post
    :type p: dict
    :return: image of p
    :rtype: Image
    """
    return None


def post_img_ocr(img_p):
    """Get the text of the image of a post

    :param img_p: image of post
    :type img_p: Image
    :return: image text
    :rtype: str
    """
    return None


def post_title(p):
    """Get post title/text

    :param p: post
    :type p: dict
    :return: title/text of the post
    :rtype: str
    """
    return p["postText"]


def article_title(p):
    """Get article title
    :param p: post
    :type p: dict
    :return: title of article of post
    :rtype: str
    """
    return p["targetTitle"]


def article_description(p):
    """Get article description

    :param p: post
    :type p: dict
    :return: description of article of post
    :rtype: str
    """
    return p["targetDescription"]


def article_keywords(p):
    """Get article keywords

    :param p: post
    :type p: dict
    :return: keywords of article of post
    :rtype: list[str]
    """
    return p["targetKeywords"]


def article_paragraphs(p):
    """Get article paragraphs

    :param p: post
    :type p: dict
    :return: paragraphs of article of post
    :rtype: list[str]
    """
    return p["targetParagraphs"]


def article_captions(p) -> None:
    """Get article captions

    :param p: post
    :type p: dict
    :return: captions of article of post
    :rtype: list[str]
    """
    return p["targetCaptions"]


def len_characters(content):
    """Get number of characters in content
    If the content is a list, it returns the average number of characters per element

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: number of characters in content, or -1 if no available content
    :rtype: float
    """
    if not content:
        return -1

    if type(content) is list:
        cumulative = 0
        for element in content:
            cumulative += len(element)
        return cumulative / len(content)

    return float(len(content))


# Assumption: Return the total number of words (not number of unique words)
def len_words(content):
    """Get number of words in content
    If the content is a list, it returns the average number of words per element

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: number of words in content, or -1 if no available content
    :rtype: float
    """
    if not content:
        return -1

    if type(content) is list:
        cumulative = 0
        for element in content:
            cumulative += len(element.split())
        return cumulative / len(content)

    return float(len(content.split()))


# Assumption: We remove punctuation, numbers and convert all letters to lowercase for easy comparison between words
def words(content):
    """Get the set of words in the content. Note: Removes interpunction, numbers and capital letters in the process

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: set of words in content
    :rtype: set[str]
    """

    if type(content) is list:
        separator = " "
        content = separator.join(content)

    return set(content.split())


# Assumption: WordNet is the same in pydictionary and NLTK. As we use NLTK instead of pydictionary
def lang_dict_formal(words):
    """Get the set of formal words from a set of words

    :param words: set of words
    :type words: set[str]
    :return: set of formal words
    :rtype: set[str]
    """
    formal_set = set([])

    for word in words:
        if len(wn.synsets(word)) != 0:
            formal_set.add(word)

    return formal_set


def lang_dict_informal(words):
    """Get the set of informal words from a set of words

    :param words: set of words
    :type words: set[str]
    :return: set of informal words
    :rtype: set[str]
    """
    informal_set = set([])

    for word in words:
        if len(wn.synsets(word)) == 0:
            informal_set.add(word)

    return informal_set
