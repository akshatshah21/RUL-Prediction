import sys
import numpy as np
from rul_predictor import RULPredictor
from data_extraction import DataExtraction

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_extractor = DataExtraction()
    rp = RULPredictor()

    data_extractor.set_md_threshold('../dataset/Learning_set/Bearing1_1/', mode='max')
    
    rp.w = data_extractor.w
    MD_MEAN = data_extractor.mean
    MD_THRESHOLD = data_extractor.MD_THRESHOLD


    # Values for B1_1
    # rp.w = 14.798881245900155
    # MD_THRESHOLD = 891602.985136060
    # MD_MEAN = 518767.90291495435
    

    data = data_extractor.get_test_data('../dataset/Test_set/Bearing1_3/', mode='max', save_to_file=True, file_path='my_data/test_data_1_3_wrt_1_1.npz')
    
    # data = data_extractor.get_test_data_from_file(file_path='../my_data/test_data.npz')
    # print(data.shape)

    data_extractor.plot_test_data(data, load_from_file=False, file_path=None)

    print("DATA")
    print(data[:2])

    pred_ruls = []
    pred_times = []
    start = 0
    it = start
    for t, md in data[start:]:
        if md > MD_THRESHOLD:
            y = np.log(np.absolute(md - MD_MEAN))
            rp.reading(y ,t / 10 ** 6) # Update theta using EM
            rul = rp.predict_RUL()
            print("it", it, "time=", t/10 ** 6, "rul", rul)
            # predictions.append([y, t / 10 ** 6, rul]) # Calculate RUL
            pred_times.append(it)
            pred_ruls.append(rul)
        it += 1
        

    plt.plot(pred_times, pred_ruls)
    plt.xlabel("Sample number")
    plt.ylabel("RUL in seconds?")
    plt.grid()
    plt.show()