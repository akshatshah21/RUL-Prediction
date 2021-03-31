from rul_predictor import RULPredictor

if __name__ == '__main__':
    # rp = RULPredictor(debug=False)
    # rp.test_data()
    # rp.plot_RUL()

    rp = RULPredictor(debug=False)
    rp.set_md_threshold('Learning_set/Bearing1_1/')
    data = rp.get_test_data('Learning_set/Bearing1_1/')
    rp.test_data(data)