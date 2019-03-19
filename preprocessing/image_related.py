"""This file contains all functions to extract the image-related features, see section 3.2.1"""


def image_presence(x):
    """Check the presence of an image in post x

    :param x: the post
    :type x: dict
    :return: 1 if image is present, otherwise 0
    :rtype: int
    """
    return int(x["postMedia"] != [])


def text_in_image(x):
    """Check the presence of text in the image of post x

    :param x: the post
    :type x: dict
    :return: 1 if text is present, otherwise 0
    :rtype: int
    """
    return 0
