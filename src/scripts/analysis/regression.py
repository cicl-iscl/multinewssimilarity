import matplotlib.pyplot as plt

from pandas import read_csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import SCORE_CSV
from src.logger import log

if __name__ == "__main__":
    log.info(f"Using {SCORE_CSV} for linear regression.")

    df = read_csv(SCORE_CSV, sep=',')
    x, y = df['computed_score'].to_numpy().reshape(-1, 1), df['original_score'].to_numpy()
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=5)
    r = LinearRegression()
    r.fit(train_x, train_y)
    print(r.score(train_x, train_y))
    print(r.coef_)
    y_pred = r.predict(test_x)

    log.info("Mean squared error: %.2f" % mean_squared_error(test_y, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    log.info("Coefficient of determination: %.2f" % r2_score(test_y, y_pred))

    plt.scatter(test_x, test_y, color="black")
    plt.plot(test_x, y_pred, color="blue", linewidth=3)
    plt.show()
