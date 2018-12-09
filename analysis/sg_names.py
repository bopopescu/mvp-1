from corefunctions import core
import pandas as pd

def get_company_list():
    df_company_list = pd.read_csv("./sg_company_list.csv")
    return df_company_list

def sg_research_analysis():
    filename= "/home/kasun/Documents/mvp/data/sginvestors.csv"
    df = core.getData_fromcsv(filename) #all research analysis on singapore stock names

    df_company_list = get_company_list()
    core.find_key_themes(df_company_list['company'],df)
    core.find_sentiment_scores(df_company_list['company'],df)

def get_key_phrases(companyName):
    df_keyPhrases = pd.read_csv("./"+companyName+"_keyPrases.csv")
    return df_keyPhrases

def get_analysis(companyName):
    keyPrases = get_key_phrases(companyName)
    for keyPhrase in keyPrases['keyPhrase']:
        core.sentimentPlot(keyPrase,companyName)


def show_analysis(key_phrase,company):
    df_list = core.get_impact_with_keyword(key_phrase,company)
    print("Postives : \n")
    for sent in df_list[0]['Sentence']:
        print(sent)
    print("\n")
    print("Negatives : \n")
    for sent in df_list[1]['Sentence']:
        print(sent)

show_analysis("private wealth","dbs") # issue a qeury with key phrase and company name