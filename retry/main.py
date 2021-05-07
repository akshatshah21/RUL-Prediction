import sys
from rul_predictor import RULPredictor
from data_extraction import DataExtraction

if __name__ == '__main__':
    data_extractor = DataExtraction()
    rp = RULPredictor()


    data_extractor.set_md_threshold('./dataset/Learning_set/Bearing1_1/', mode='max')

    data = data_extractor.get_test_data('./dataset/Learning_set/Bearing1_1/', mode='max', save_to_file=False, file_path=None)
    print(data.shape)
    # data = data_extractor.get_test_data_from_file(data, file_path='')

    # data_extractor.plot_test_data(data, load_from_file=False, file_path=None)

    # predictions = []
    # for y, t in data:
    #     rp.reading(y ,t) # Update theta using EM
	# 	predictions.append((y, t, rp.predict_rul()) # Calculate RUL