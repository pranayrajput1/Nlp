from ResumeParser.data_extraction import generate_dataset, visualising_overlapping
from ResumeParser.model import predict, error_graph

INPUT_DIRECTORY = r'src/Data/Input_data'
OUTPUT_DIRECTORY = r'src/Data/Output_data'

if __name__ == "__main__":
    # generate_dataset(INPUT_DIRECTORY, OUTPUT_DIRECTORY) # To generate the training dataset pipeline
    # predict()
    # error_graph()
    obj = visualising_overlapping()
    print(obj)
    import json
    with open('overlapping.json', 'w') as fp:
        json.dump(obj, fp)


