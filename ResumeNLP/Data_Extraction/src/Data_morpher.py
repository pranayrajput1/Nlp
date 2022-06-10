## Entity Extraction using Pyreparser an open source library

from pyresparser import ResumeParser
import json
import spacy
import nltk
import os

# Resume Entity extraction


INPUT_DIRECTORY = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Input_data'
OUTPUT_DIRECTORY = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Extracted_data/'
extension = (".pdf", ".docx")

def resume_extraction(inp_dirc,ext_direc):
    try:

        for root, dirs, files in os.walk(inp_dirc):
            for dir in dirs:
                filepath = os.path.join(inp_dirc, dir)
                for files in os.listdir(filepath):
                    if files.endswith(extension):
                        data = ResumeParser(os.path.join(filepath, files)).get_extracted_data()        #Getting the extracted data in a dict using pyreparser
                        out_file = open(os.path.join(ext_direc, files + ".json"), "w")
                        json.dump(data, out_file, indent=4, sort_keys=False)
                        out_file.close()
    except Exception as e:
        print(e)


resume_extraction(INPUT_DIRECTORY,OUTPUT_DIRECTORY)