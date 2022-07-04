from pyresparser import ResumeParser
import json
import os
import pandas as pd

extension = (".pdf", ".docx")

# Extraction of JSON from Resumes PDF/DOC-wise
def resume_extraction(input_file):
    dict = ResumeParser(input_file).get_extracted_data()
    dict_to_df = pd.DataFrame.from_dict(dict, orient='index')
    dict_to_df = dict_to_df.transpose()
    # dict_to_df.to_csv("single_resume_extraction_result.csv", index=False)
    return dict_to_df

# Creating domain-wise directories in output directories
def generate_outputDir(inp_dirc, out_dirc):
    for root, dirs, files in os.walk(inp_dirc):
        for dir in dirs:
            output_filepath = os.path.join(out_dirc, dir)
            isExist = os.path.exists(output_filepath)
            if not isExist:
                os.makedirs(output_filepath)

# Extraction of resumes information batch-wise
def batch_resume_extraction(inp_dirc,out_dirc):
    try:
        generate_outputDir(inp_dirc,out_dirc) # Creating domain-wise directories in output directories
        for root, dirs, files in os.walk(inp_dirc):
            nested_dict = {}
            dirc_log = open("Directory.log", "w")
            for dir in dirs:
                lst = []
                input_filepath = os.path.join(inp_dirc, dir)
                output_filepath = os.path.join(out_dirc, dir)
                dirc_log.write(str(input_filepath) + os.linesep)
                file_log = open("Files.log", "w")
                for resume_file in os.listdir(input_filepath):
                    if resume_file.endswith(extension):
                        file_log.write(str(resume_file) + os.linesep)
                        data = ResumeParser(os.path.join(input_filepath, resume_file)).get_extracted_data() #Getting the extracted data in a dict using pyreparser
                        skill_list = data['skills']
                        out_file = open(os.path.join(output_filepath,resume_file + ".json"), "w")
                        json.dump(data, out_file, indent=4, sort_keys=False)
                        out_file.close()
                        lst.append(skill_list)
                dict = {dir: lst}
                nested_dict.update(dict)
            return nested_dict
    except Exception as e:
        print(e)

# Generating the dataset
def generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY):
    nested_list = batch_resume_extraction(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    skill_df = pd.DataFrame.from_dict(nested_list, orient='index')
    skill_df = skill_df.stack().reset_index()
    skill_df.rename(columns={'level_0': "Department", 0:"Skills"}, inplace=True)
    skill_df = skill_df.drop(['level_1'], axis = 1)
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace('[', '')
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace(']', '')
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace("'", '')
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace(",", '')
    skill_df.to_csv("dataset.csv", index=False)
    return skill_df