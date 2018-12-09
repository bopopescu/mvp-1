import spacy
import pandas as pd
from spacy import displacy
from collections import Counter
import re
import io
import os
import time
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
from itertools import chain
#import matplotlib.pyplot as plt
import json
from gensim.summarization.summarizer import summarize
import heapq

def getText(TextFile):
    f = open(TextFile,'r')
    message = f.read()
    f.close()
    return message

def getSentences(article):
    sentences = [x for x in article.sents]
    return(sentences)

def getData_fromcsv(filename):
    df = pd.read_csv(filename)
    return df

def showWordCount(wordlist):
    return Counter(wordlist)

def getCompanyData(df, companyName):
    df = df[df['company'].str.contains(companyName)]
    return df.sort_values(['published-date'])

def ExtractKeyPhrases(Text):
    #keyphrase extraction works with file, thus writin text to temp file and pass it to process
    TextFile = "./temp.txt"
    f = open(TextFile,'w')
    f.write(Text)
    f.close()
    # initialize keyphrase extraction model, here TopicRank
    extractor = pke.unsupervised.TopicRank(input_file=TextFile)
    # load the content of the document, here document is expected to be in raw
    # format (i.e. a simple text file) and preprocessing is carried out using nltk
    extractor.read_document(format='raw')
    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives
    extractor.candidate_selection()
    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 50 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=50, stemming=False)
    return ([i[0] for i in keyphrases]) #returning only the keyphrases without their score

def ExtractSummary(content,n):
    summary = ""
    try:
        summary = summarize(content, word_count=n)
    except:
        summary = content
    return summary

def SentimentAnalysisNLTK(Text):
    sia = SIA()
    SentimentScore = sia.polarity_scores(Text.lower())
    return (SentimentScore)


def summaryofAllArticles(df, n):
    summary_list = []
    for content in df['content']:
        summary = ExtractSummary(content, n)
        summary_list = summary_list.append(summary)
    return summary_list


def keyPointsOfAllArticles(df):
    allPhrases = []
    for content in df['content']:
        allPhrases.append(ExtractKeyPhrases(content))
    return list(chain.from_iterable(allPhrases))

def find_key_themes(companyList,df):
    for item in companyList:
        df_company = getCompanyData(df,item)
        keyPoints = keyPointsOfAllArticles(df_company)
        wordCount = showWordCount(keyPoints)
        with open("./"+item+"_KeyPhrases.csv",'w') as f:
            for k,v in  wordCount.most_common():
                f.write( "{}, {}\n".format(k,v) )


def insightJSON(phrase, company):
    filename = "./" + company + "_Sentiments.csv"
    df = pd.read_csv(filename, error_bad_lines=False)
    df = df[df['Sentence'].str.contains(phrase)]
    df = df[['Date', 'Sentence', 'Compound']]
    insightJSON = pd.DataFrame.to_json(df, orient='records')
    print(json.dumps(insightJSON, indent=4, sort_keys=True))


def sentimentPlot(phrase, company):
    filename = "./" + company + "_Sentiments.csv"
    df = pd.read_csv(filename, error_bad_lines=False)
    df = df[df['Sentence'].str.contains(phrase)]
    x_data = df['Date']
    y_data = df['Compound']
    plt.plot(x_data, y_data)
    plt.savefig('./plots/'+company+"_impactScore_"+phrase+".png")
    #plt.show()



def get_sentences_with_key_word(sentenceList, keyword):
    return [sentence.text for sentence in sentenceList if keyword in sentence.text]


def get_impact_with_keyword(phrase, company):
    filename = "/home/kasun/SpeechRec/" + company + "_Sentiments_News.csv"
    df = pd.read_csv(filename, error_bad_lines=False)
    df = df[df['Sentence'].str.contains(phrase)]
    df_positive = df[df['Compound']>0]
    df_negative = df[df['Compound']<0]
    return [df_positive[-5:],df_negative[-5:]]



