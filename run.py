import pandas as pd

from ResumeParser.data_extraction import generate_dataset
from ResumeParser.model import predict_on_splitData, predict_on_one_testData, predict_on_csv_testData

INPUT_DIRECTORY = r'src/Data/Input_data'
OUTPUT_DIRECTORY = r'src/Data/Output_data'
sample_test = 'machine learning python keras ai django database spacy nltkanalysis agile'
nums = 2

if __name__ == "__main__":
    # generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY) # To generate the training dataset pipeline
    predict_on_splitData() # To generate predictions on split dataset
    test_set = pd.read_csv("old_test_df.csv")
    print(predict_on_one_testData(sample_test)) # To generate predictions on input given
    print(predict_on_csv_testData(test_set)) # To generate predictions on test data




