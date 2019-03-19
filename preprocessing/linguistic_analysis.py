"""This file contains all functions to perform the linguistic analysis, see section 3.2.2"""


def num_of_characters(x):
    """Measure the number of characters in x

    :param x: post_title, post_img_ocr, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type x: str or list[str]
    :return: number of characters in x, or -1 if x contains no characters
    :rtype: int
    """
    print("asdf")
    return -1


def diff_num_of_characters(cont_x, cont_y):
    """Measures the difference between the number of characters in two content elements

    :param cont_x:
    :type cont_x:
    :param cont_y:
    :type cont_y:
    :return: difference between the number of characters in cont_x and cont_y
    :rtype: int
    """
    return 0


def num_of_characters_ratio(cont_x, cont_y):
    """Measure the ratio between the number of characters of two content elements

    :param cont_x:
    :type cont_x:
    :param cont_y:
    :type cont_y:
    :return: ratio between the number of characters of two content elements, -1 if either is None
    :rtype: float
    """
    return -1


def num_of_words(x):
    """Measure the number of words in x

    :param x: post title, text in post image, article title, article description, article keywords, article captions or article paragraphs
    :type x: str or list[str]
    :return: number of words in x, or -1 if x contains no words
    :rtype: int
    """
    return -1


def diff_num_of_words(cont_x, cont_y):
    """Measure the difference between the number of words in two content elements

    :param cont_x:
    :type cont_x:
    :param cont_y:
    :type cont_y:
    :return: difference between the number of words in two content elements
    :rtype: int
    """
    return 0


def num_of_words_ratio(cont_x, cont_y):
    """Measure the ratio between the number of words of two content elements

    :param cont_x:
    :type cont_x:
    :param cont_y:
    :type cont_y:
    :return: ratio between the number of words of two content elements, -1 if either is None
    :rtype: float
    """
    return -1


def num_of_common_words(keywords, cont_x):
    """Measure the number of words the article keywords and content element have in common

    :param keywords: keywords of article
    :type keywords: str
    :param cont_x:
    :type cont_x:
    :return: number of words keywords and cont_x have in common
    :rtype: int
    """
    return 0


def number_of_formal_words(x):
    """Measure the number of formal words in a content element

    :param x:
    :type x:
    :return: number of formal words in content element x
    :rtype: int
    """
    return 0


def number_of_informal_words(x):
    """Measure the number of informal words in a content element

    :param x:
    :type x:
    :return: number of informal words in content element x
    :rtype: int
    """
    return 0


def percent_of_formal_words(x):
    """Measure the percentage of formal words of a content element

    :param x:
    :type x:
    :return: ratio between formal and informal words of a content element
    :rtype: float
    """
    return -1


def percent_of_informal_words(x):
    """Measure the percentage of informal words of a content element

    :param x:
    :type x:
    :return: ratio between informal and formal words of a content element
    :rtype: float
    """
    return -1
