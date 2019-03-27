"""This file contains all functions to extract the basic content related features"""
import io
import json
import re
from PyDictionary import PyDictionary

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

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: number of characters in content, or -1 if no available content
    :rtype: int
    """
    if content is None:
        return -1

    if type(x) is list:
        seperator = ""
        x = seperator.join(x)
    
    return len(x)


def len_words(content):
    """Get number of words in content

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: number of words in content, or -1 if no available content
    :rtype: int
    """
    return len(words(content))

#Assumption: We remove interpunction, numbers and convert all letters to lowercase for easy comparison between words
def words(content):
    """Get the set of words in the content. Note: Removes interpunction, numbers and capital letters in the process

    :param content: post_title, article_title, article_description, article_keywords, article_paragraphs or article_captions
    :type content: str or list[str]
    :return: set of words in content
    :rtype: set[str]
    """

    if type(content) is list:
        seperator = " "
        content = seperator.join(content)
    
    regex = re.sub("[^a-zA-Z\s]", "", content).lower()

    return regex.split()


def lang_dict_formal(words):
    """Get the set of formal words from a set of words

    :param words: set of words
    :type words: set[str]
    :return: set of formal words
    :rtype: set[str]
    """
    formal_set = set([])
    return formal_set


def lang_dict_informal(words):
    """Get the set of informal words from a set of words

    :param words: set of words
    :type words: set[str]
    :return: set of informal words
    :rtype: set[str]
    """
    informal_set = set([])
    return informal_set


def process(post):
    # Existing fields:
    #   postMedia(optional): relative link to image used in post
    #   postText: text used in post
    #   id: post id
    #   targetCaptions: list of captions of images in target article
    #   targetParagraphs: list of paragraphs in target article
    #   targetTitle: title of target article
    #   postTimestamp: timestamp of post, formatted as "%a %b %d %H:%M%:%S %z %Y"
    #   targetKeywords: keywords in target article
    #   targetDescription: description of target article

    # add all other fields to post by calling all the functions
    # in image_related.py, linguistic_analysis.py and abuser_detection.py
    with open("../data/preprocessed.jsonl", 'a') as output_file:
        # write post to file
        # output_file.write(json.dumps(post))
        print(words(post_title(post)))


if __name__ == '__main__':
    counter = 0   
    with io.open("../data/instances.jsonl", 'r') as input_file:
        for line in input_file:
            if counter == 10:
                break
            post = json.loads(line)
            process(post)
            counter += 1

