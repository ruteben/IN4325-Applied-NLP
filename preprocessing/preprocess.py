"""This file contains the preprocessing logic"""
import io
import json
import csv
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.corpus import wordnet
from preprocessing import image_related as ir
from preprocessing import linguistic_analysis as la
from preprocessing import abuser_detection as ad


def filter_post(post):
    """Filter out hexadecimal non-breaking spaces, non-alphabetical characters and leading or trailing spaces

    :param post: post
    :type post: dict
    :return: post with non-breaking spaces, non-alphabetical characters replaced by regular spaces
    :rtype: dict
    """
    filtered_post = post.copy()
    pattern = r"[^a-zA-Z\s]+|\\u[a-z0-9]{4}|\\xa0"

    post_text = post["postText"]
    if type(post_text) is not list:
        post_text = [post_text]
    filtered_post["postText"] = stem_content(re.sub(pattern, " ", post_text[0]))
    filtered_post["postTextUnstemmed"] = re.sub(pattern, " ", post_text[0]).strip().lower()

    filtered_captions = []
    filtered_captions_unstemmed = []
    for caption in post["targetCaptions"]:
        filtered_caption = stem_content(re.sub(pattern, " ", caption))
        if filtered_caption:
            filtered_captions.append(filtered_caption)
        filtered_caption = re.sub(pattern, " ", caption).strip().lower()
        if filtered_caption:
            filtered_captions_unstemmed.append(filtered_caption)
    filtered_post["targetCaptions"] = filtered_captions
    filtered_post["targetCaptionsUnstemmed"] = filtered_captions_unstemmed

    filtered_paragraphs = []
    filtered_paragraphs_unstemmed = []
    for paragraph in post["targetParagraphs"]:
        filtered_paragraph = stem_content(re.sub(pattern, " ", paragraph))
        if filtered_paragraph:
            filtered_paragraphs.append(filtered_paragraph)
        filtered_paragraph = re.sub(pattern, " ", paragraph).strip().lower()
        if filtered_paragraph:
            filtered_paragraphs_unstemmed.append(filtered_paragraph)
    filtered_post["targetParagraphs"] = filtered_paragraphs
    filtered_post["targetParagraphsUnstemmed"] = filtered_paragraphs_unstemmed

    filtered_post["targetTitle"] = stem_content(re.sub(pattern, " ", post["targetTitle"]))
    filtered_post["targetTitleUnstemmed"] = re.sub(pattern, " ", post["targetTitle"]).strip().lower()

    filtered_post["targetKeywords"] = stem_content(re.sub(pattern, " ", post["targetKeywords"]))
    filtered_post["targetKeywordsUnstemmed"] = re.sub(pattern, " ", post["targetKeywords"]).strip().lower()

    filtered_post["targetDescription"] = stem_content(re.sub(pattern, " ", post["targetDescription"]))
    filtered_post["targetDescriptionUnstemmed"] = re.sub(pattern, " ", post["targetDescription"]).strip().lower()

    return filtered_post


def stem_content(content):
    ps = PorterStemmer()
    stemmed = []
    if type(content) is list:
        for word in content:
            stemmed.append(ps.stem(word).strip())
    if type(content) is str:
        words = content.split(" ")
        for word in words:
            stemmed.append(ps.stem(word).strip())

    return " ".join(stemmed).strip()


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
    fp = filter_post(post)
    f = {"id": post["id"]}

    # calculate image related features
    if image:
        f["imagePresent"] = ir.image_presence(fp)
        f["imageTextPresent"] = ir.text_in_image(fp)
        # TODO add features related to image text

    # calculate linguistic features
    if linguistic:
        content_list = ["postText", "targetCaptions", "targetParagraphs", "targetTitle", "targetKeywords", "targetDescription"]
        for i in range(len(content_list)):
            # properly capitalize the string
            content_string1 = content_list[i][0].capitalize() + content_list[i][1:]
            # number of characters in content
            f["chars" + content_string1] = la.num_of_characters(fp[content_list[i]])
            # number of words in content
            f["words" + content_string1] = la.num_of_words(fp[content_list[i]])
            # number of words in common with keywords
            if content_list[i] != "targetKeywords":
                f["wordsCommon" + content_string1] =\
                    la.num_of_common_words(fp["targetKeywords"], fp[content_list[i]])
            # number of formal words in content
            f["wordsFormal" + content_string1] =\
                la.number_of_formal_words(fp[content_list[i] + "Unstemmed"])
            # number of informal words in content
            f["wordsInformal" + content_string1] =\
                la.number_of_informal_words(fp[content_list[i] + "Unstemmed"])
            # percentage of formal words in content
            f["wordsFormalPercent" + content_string1] =\
                la.percent_of_formal_words(fp[content_list[i] + "Unstemmed"])
            # percentage of informal words in content
            f["wordsInformalPercent" + content_string1] =\
                la.percent_of_informal_words(fp[content_list[i] + "Unstemmed"])
            # ASSUMPTION: no need to determine ratio(a,b) and ratio(b,a)
            for j in range(i+1, len(content_list)):
                # properly capitalize the string
                content_string2 = content_list[j][0].capitalize() + content_list[j][1:]
                # difference between number of characters in two pieces of content
                f["charsDiff" + content_string1 + content_string2] =\
                    la.diff_num_of_characters(fp[content_list[i]], fp[content_list[j]])
                # ratio of number of characters in two pieces of content
                f["charsRatio" + content_string1 + content_string2] =\
                    la.num_of_characters_ratio(fp[content_list[i]], fp[content_list[j]])
                # difference between number of words in two pieces of content
                f["wordsDiff" + content_string1 + content_string2] =\
                    la.diff_num_of_words(fp[content_list[i]], fp[content_list[j]])
                # ratio of number of words in two pieces of content
                f["wordsRatio" + content_string1 + content_string2] =\
                    la.num_of_words_ratio(fp[content_list[i]], fp[content_list[j]])

    # calculate abuser detection related features
    if abuser:
        # TODO add more
        f["creation_hour"] = ""
    return f


if __name__ == '__main__':
    post_features_list = []
    counter = 0
    with io.open("../data/instances.jsonl", 'r') as input_file:
        for line in input_file:
            post = json.loads(line)
            post_features_list.append(process(post, image=False, abuser=False))
            counter += 1
            if counter % 100 == 0:
                print(counter)

    with open("../data/preprocessed.csv", 'w', encoding='utf-8') as output_file:
        # get header, make sure id is first
        fieldnames = list(post_features_list[0].keys())  # TODO add actual fieldnames
        fieldnames.remove("id")
        fieldnames.sort()
        fieldnames = ["id"] + fieldnames
        # create dictionary writer
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, delimiter=';', dialect='excel')
        # write header to file first
        writer.writeheader()
        for post_features in post_features_list:
            # write each post to file
            writer.writerow(post_features)
