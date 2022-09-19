from ResumeParser.data_extraction import generate_dataset
from ResumeParser.model import predict_on_splitData, predict_on_testData

INPUT_DIRECTORY = r'src/Data/Input_data'
OUTPUT_DIRECTORY = r'src/Data/Output_data'
sample_test = ['tkinter algorithms machine learning algorithms numpy python tensorflow pycharm matplotlib opencv anaconda business intelligence nltk c sql pandas scikitlearn spacy nlp sql nltk']
nums = 2

if __name__ == "__main__":
    # generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY) # To generate the training dataset pipeline
    # predict_on_splitData() # To generate predictions on split dataset
    print(predict_on_testData(sample_test, nums)) # To generate predictions on input given



