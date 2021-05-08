import numpy as np
from rul_prediction.rul_predictor import RULPredictor
from rul_prediction.data_extraction import DataExtraction

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_extractor = DataExtraction()
    rp = RULPredictor()

    data_extractor.set_md_threshold('./dataset/Learning_set/Bearing1_1/', mode='max')
    
    rp.w = data_extractor.w
    MD_MEAN = data_extractor.mean
    MD_THRESHOLD = data_extractor.MD_THRESHOLD


    data = data_extractor.get_test_data('./dataset/Learning_set/Bearing1_1/', mode='max', save_to_file=True, file_path='./my_data/test_data_1_1_wrt_1_1.npz')
    
    # data = data_extractor.get_test_data_from_file(file_path='./my_data/test_data_1_1_wrt_1_1.npz')

    data_extractor.plot_test_data(data, load_from_file=False, file_path=None)

    pred_ruls = []
    pred_times = []
    start = 0
    it = start
    for t, md in data[start:]:
        if md > MD_THRESHOLD:
            y = np.log(np.absolute(md - MD_MEAN))
            rp.reading(y ,t / 10 ** 6)  # Update theta using EM
            rul = rp.predict_RUL()
            print(f"Iteration: {it}\tTime: {t / 10 ** 6}\tRUL: {rul}")
            pred_times.append(it)
            pred_ruls.append(rul)
        it += 1

    # Actual RUL
    actual_ruls = [*range(pred_times[-1]-pred_times[0], -1, -1)]
    actual_ruls = list(map(lambda x: x*10, actual_ruls))    # time between two files is 10s

    # RUL plot
    plt.plot(pred_times, actual_ruls, label='Actual')
    plt.plot(pred_times, pred_ruls, label='Predicted')
    plt.xlabel("Sample number")
    plt.ylabel("RUL in seconds")
    plt.legend()
    plt.title("Bearing1_1 RUL Plot")
    plt.grid()
    plt.show()


    # Error Plot
    error = list(np.array(actual_ruls) - np.array(pred_ruls))
    plt.plot(pred_times, error, label='Error')
    plt.xlabel("Sample number")
    plt.ylabel("Tracking error")
    plt.legend()
    plt.title("Bearing1_1 Error plot")
    plt.grid()
    plt.show()