from ResumeParser.data_extraction import generate_dataset

INPUT_DIRECTORY = r'src/Data/Input_data'
OUTPUT_DIRECTORY = r'src/Data/Output_data'

if __name__ == "__main__":
    generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY) # To generate the training dataset pipeline


