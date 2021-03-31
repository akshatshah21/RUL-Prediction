from rul_predictor import RULPredictor

if __name__ == '__main__':
    rp = RULPredictor(debug=False)
    rp.test_data()
    rp.plot_RUL()