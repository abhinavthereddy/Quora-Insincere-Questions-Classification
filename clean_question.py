import pandas as pd
import re
import nltk
import nltk.data
import json
pd.options.mode.chained_assignment = None
import contractions
import csv
from html.parser import HTMLParser
import ftfy
import string
from bs4 import BeautifulSoup

class Clean_Question ():

    def __init__(self):
        pass
       
    def remove_non_ascii (self, sent):
        sent = sent.encode("ascii", errors="ignore").decode()   # Removes non-ASCII symbols
        return sent

    def beautiful_soup (self, df):    # Takes entire data
        df = df.apply(lambda x: BeautifulSoup(x, "lxml").text)
        return df

    def basic_clean (self, sent):            
        # Removes weird characters, extra spaces, hashtags, HTTP, and @xyz taggers, and contraction 		
        sent = ftfy.fix_text(sent)        
        return sent

    def contraction (self, sent):
        return contractions.fix(sent)

    def html_parser (self, sent):        
        sent = re.sub(r'http\S+', 'URL', sent)
        sent = re.sub(r'pic\S+', 'URL', sent)
        sent = re.sub(r'www\S+', 'URL', sent)
        return sent

    def remove_emojis (self, sent):
        RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
        sent = RE_EMOJI.sub(r'', sent)          
        return sent 

    def lower_case (self, sent):
        return sent.lower()
    
    def remove_punctuation (self, sent):
        for x in sent.lower(): 
                if x in string.punctuation: 
                    sent = sent.replace(x, " ") 
        return sent

    def clean_all(self, sent):
        # Cleans an single sentence.
        sent = self.remove_punctuation(sent)
        sent = self.remove_non_ascii(sent)
        sent = self.html_parser(sent)
        sent = self.basic_clean(sent)        
        sent = self.contraction(sent)
        sent = self.remove_emojis (sent)
        sent = self.lower_case (sent)
        return sent

    def clean_df (self, df):
        # Cleans an entire dataframe.
        df = self.beautiful_soup(df)
        df = df.apply(lambda x:  self.clean_all(x)  ) 
        return df
