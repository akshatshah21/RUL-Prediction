import sys
from rul_predictor import RULPredictor
from data_extraction import DataExtraction

if __name__ == '__main__':
    data_extractor = DataExtraction()
    rp = RULPredictor()

    data_extractor.set_md_threshold('../dataset/Learning_set/Bearing1_1/', mode='max')
    rp.w = data_extractor.w

    # data = data_extractor.get_test_data('../dataset/Learning_set/Bearing1_1/', mode='max', save_to_file=True, file_path='../my_data/test_data.npz')
    
    data = data_extractor.get_test_data_from_file(file_path='../my_data/test_data.npz')
    # print(data.shape)

    # data_extractor.plot_test_data(data, load_from_file=False, file_path=None)

    predictions = []
    for y, t in data:
        if y > data_extractor.MD_THRESHOLD :
            rp.reading(y ,t) # Update theta using EM
            rul = rp.predict_RUL()
            print(rul)
            predictions.append(y, t, rul) # Calculate RUL