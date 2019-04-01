"""This file contains the preprocessing logic"""
import io
import json
import csv
import re
from preprocessing import basics
from preprocessing import image_related
from preprocessing import linguistic_analysis
from preprocessing import abuser_detection


def filter(post):
    """Filter out the hexadecimal non-breaking space

    :param post: post
    :type post: dict
    :return: post with non-breaking spaces replaced by regular spaces
    :rtype: dict
    """
    filtered_post = post
    pattern = r"\xa0"
    filtered_post["postText"] = re.sub(pattern, " ", post["postText"][0])
    filtered_captions = []
    for caption in post["targetCaptions"]:
        filtered_captions.append(re.sub(pattern, " ", caption))
    filtered_post["targetCaptions"] = filtered_captions
    filtered_paragraphs = []
    for paragraph in post["targetParagraphs"]:
        filtered_paragraphs.append(re.sub(pattern, " ", paragraph))
    filtered_post["targetParagraphs"] = filtered_paragraphs
    filtered_post["targetTitle"] = re.sub(pattern, " ", post["targetTitle"])
    filtered_post["targetKeywords"] = re.sub(pattern, " ", post["targetKeywords"])
    filtered_post["targetDescription"] = re.sub(pattern, " ", post["targetDescription"])

    return filtered_post


def process(post, linguistic=True, image=True, abuser=True):
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
    filtered_post = filter(post)
    features = {}

    # calculate base features
    features["img"] = basics.post_img(filtered_post)

    # calculate image related features
    if image:
        # TODO add more
        features["imagePresent"] = image_related.image_presence(filtered_post)

    # calculate linguistic features
    if linguistic:
        # TODO add more
        features["charsPostTest"] = linguistic_analysis.num_of_characters(post["Text"])

    # calculate abuser detection related features
    if abuser:
        # TODO add more
        features["creation_hour"] = ""
    return features


if __name__ == '__main__':
    posts = []
    counter = 0
    with io.open("../data/instances.jsonl", 'r') as input_file:
        for line in input_file:
            post = json.loads(line)
            posts.append(process(post, abuser=False))
            counter += 1
            if counter % 1000 == 0:
                print(counter)

    with open("../data/preprocessed.csv", 'w', encoding='utf-8') as output_file:
        fieldnames = ['postTimestamp', 'postText', 'targetKeywords', 'id', 'targetCaptions', 'targetTitle',
                      'targetParagraphs', 'postMedia', 'targetDescription']  # TODO add actual fieldnames
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';', dialect='excel')
        writer.writeheader()
        for post in posts:
            writer.writerow(post)
