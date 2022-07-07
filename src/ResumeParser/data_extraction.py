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
                skills_lst = []
                name_lst = []
                email_lst = []
                phone_lst = []
                file_lst = []
                dir_lst = []
                input_filepath = os.path.join(inp_dirc, dir)
                output_filepath = os.path.join(out_dirc, dir)
                dirc_log.write(str(input_filepath) + os.linesep)
                file_log = open("Files.log", "w")
                for resume_file in os.listdir(input_filepath):
                    if resume_file.endswith(extension):
                        file_log.write(str(resume_file) + os.linesep)
                        data = ResumeParser(os.path.join(input_filepath, resume_file)).get_extracted_data() #Getting the extracted data in a dict using pyreparser
                        skill_list = data['skills']
                        name = data['name']
                        email = data["email"]
                        mobile_number = data["mobile_number"]
                        file_name = resume_file
                        out_file = open(os.path.join(output_filepath, resume_file + ".json"), "w")
                        json.dump(data, out_file, indent=4, sort_keys=False)
                        out_file.close()
                        skills_lst.append(skill_list)
                        name_lst.append(name)
                        email_lst.append(email)
                        phone_lst.append(mobile_number)
                        file_lst.append(file_name)
                        dir_lst.append(dir)
                dict = {"Name": name_lst, "Email_ID": email_lst, "Phone_Number": phone_lst, "File_Name": file_lst, "Deptartment": dir_lst, "Skills": skills_lst}
                nested_dict.update({dir: dict})
            return nested_dict
    except Exception as e:
        print(e)

# Generating the dataset
def generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY):
    nested_list = batch_resume_extraction(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    skill_df = pd.DataFrame.from_dict(nested_list, orient='index')
    copy_skill_df = skill_df.copy()
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace('[', '')
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace(']', '')
    skill_df['Skills'] = skill_df['Skills'].astype(str).str.replace("'", '')
    skill_df.to_csv("union_skill_dataset.csv", index=False)

    result_df = copy_skill_df.explode(['Name', 'Email_ID', 'Phone_Number', 'File_Name', 'Deptartment', 'Skills'])
    result_df[result_df['Skills'].map(lambda d: len(d)) > 0]
    result_df.to_csv("DataSet.csv", index=False)
    return result_df

def visualising_overlapping():
    data_set = pd.read_csv("DataSet.csv",na_filter=True, na_values='[]')
    new_data_set = data_set.copy()
    new_data_set['Skills'] = new_data_set['Skills'].astype(str).str.replace('[', '')
    new_data_set['Skills'] = new_data_set['Skills'].astype(str).str.replace(']', '')
    new_data_set['Skills'] = new_data_set['Skills'].astype(str).str.replace("'", '')
    skills_union = ', '.join(new_data_set.Skills)
    skills_union_list = list(skills_union.split(","))
    final_dict_ = {}
    for i in skills_union_list:
        i = i.lstrip()
        dept_list = []
        for ind in data_set.index:
            df_skill = list(new_data_set['Skills'][ind].split(","))
            for skill in df_skill:
                skill = skill.lstrip()
                if skill == i:
                    dept = new_data_set['Deptartment'][ind]
                    dept_list.append(dept)
                    dept_list = list(set(dept_list))
        dict_ = {i: dept_list}
        final_dict_.update(dict_)
    return final_dict_
