 ##
 ## Entity Extraction using Pyreparser an open source library

from pyresparser import ResumeParser
import json
import spacy
import nltk


path_to_pdf = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Input_data/Navnath Harihar_Data Scientist.pdf'

data = ResumeParser(path_to_pdf).get_extracted_data()   #entity extraction saving the results in a dict
#print(data)



with open("sample.json", "w") as outfile:               #json conversion
    json.dump(data, outfile)



