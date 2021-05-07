from rul_predictor import RULPredictor
from data_extraction import DataExtraction

if __name__ == '__main__':
    data_extractor = DataExtraction()
    rp = RULPredictor()

    # data_extractor.get_dataset()
    # data_extractor.set_md_threshold()
    # data_extractor.get_test_data()


    predictions = []
    '''
    for y, t in data:
        rp.reading(y ,t) # Update theta using EM
		predictions.append((y, t, rp.predict_rul()) # Calculate RUL
    '''