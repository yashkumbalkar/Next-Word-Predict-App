import re
import string
from bs4 import BeautifulSoup
import demoji

def preprocess_text(text):
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()

        # Removing Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        #p = inflect.engine() #101
        demoji.download_codes()
        # Remove emojis
        text = demoji.replace(text, "")

        # Remove new line character
        text = text.replace('\n','')

        # Remove mentions of "Human 1" and "Human 2"
        text = re.sub(r'\b(?:Human 1|Human 2)\b:?', " ", text)

        # Remove special characters, keeping only alphabetic and spaces
        text = re.sub(r'[^a-zA-Z\s]', r' ', text)

        # Replace specific unicode spaces with standard spaces and trim
        text = text.replace(u'\xa0', u' ').replace('\u200a', ' ').strip()

        # Removing white spaces
        text = re.sub(r'\s+',r' ',text)

        # Lower case every word
        text = text.lower()

        return text


def preprocess_lines(lines):
        preprocessed = [preprocess_text(line) for line in lines]
        preprocessed = [line for line in preprocessed if line != '']
        
        return preprocessed




