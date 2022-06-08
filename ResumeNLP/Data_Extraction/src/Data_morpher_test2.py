import json
import re
import os
import shutil


# CONVERTING PDF FILES TO TXT FORMAT
def main():

    from pathlib import Path

    from pdfminer.pdfparser import PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.pdfdevice import PDFDevice
    from pdfminer.layout import LAParams, LTTextBox, LTTextLine
    from pdfminer.converter import PDFPageAggregator

    path_to_pdf = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Input_data/'
    path2 = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Extracted_data/'

    try:
        for path in Path(path_to_pdf).glob("*.pdf"):
            with path.open("rb") as file:
                parser = PDFParser(file)
                document = PDFDocument(parser, "")
                if not document.is_extractable:
                    continue

                manager = PDFResourceManager()
                params = LAParams()

                device = PDFPageAggregator(manager, laparams=params)
                interpreter = PDFPageInterpreter(manager, device)

                text = ""

                for page in PDFPage.create_pages(document):
                    interpreter.process_page(page)
                    for obj in device.get_result():
                        if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
                            text += obj.get_text()
            with open("/home/aayush/PycharmProjects/ResumeNLP/Data/Extracted_data/{}.txt".format(path.stem), "w") as file:
                file.write(text)
        return 0

    except Exception as e:
        print(e)



#JSON ENTITY EXTRCATION

# try:
#     import spacy
#     import json
# except Exception as e:
#     print(e)
#
#
# class EntityGenerator(object):
#     _slots__ = ['text']
#
#     def __init__(self, text=None):
#         self.text = text
#
#     def get(self):
#         """
#         Return a Json
#         """
#         nlp = spacy.load("en_core_web_sm")
#         doc = nlp(self.text)
#         text = [ent.text for ent in doc.ents]
#         entity = [ent.label_ for ent in doc.ents]
#
#         from collections import Counter
#
#         data = Counter(zip(entity))
#         unique_entity = list(data.keys())
#         unique_entity = [x[0] for x in unique_entity]
#
#         d = {}
#         for val in unique_entity:
#             d[val] = []
#
#         for key, val in dict(zip(text, entity)).items():
#             if val in unique_entity:
#                 d[val].append(key)
#         return d
#
#
# filename = r'/home/aayush/PycharmProjects/ResumeNLP/Data/Extracted_data/Navnath_Harihar_Data_Scientist.txt'
#
# helper = EntityGenerator(text=filename)
# response = helper.get()
# print(json.dumps(response , indent=3))







#CLEANING DIRECTORIES

# def clean_dir(Extraction_DIRECTORY, INPUT_DIRECTORY):
#     for file_dir in os.listdir(Extraction_DIRECTORY):
#         shutil.rmtree(Extraction_DIRECTORY + file_dir)
#
#     for f in os.listdir(INPUT_DIRECTORY):
#         os.remove(os.path.join(INPUT_DIRECTORY, f))
#     return None



if __name__ == "__main__":
    import sys
    sys.exit(main())