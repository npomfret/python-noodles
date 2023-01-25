import numpy as np
from alpha_vantage.timeseries import TimeSeries
from PriceHistory import PriceHistory
from typing import List, Any
from nptyping import NDArray, Int, Shape, Float, assert_isinstance
import io, json

CONFIG = {
    "key": "YOUR_API_KEY",  # Claim your free API key here: https://www.alphavantage.co/support/#api-key
    "outputsize": "full",
    "key_adjusted_close": "5. adjusted close",
}


def download_price_history(symbol: str):
    path = f'data/alpha_vantage_{symbol}.json'

    try:
        with open(path) as file:
            json_data = json.load(file)
        print(f'Using cached market data from: {path}')
    except:
        ts = TimeSeries(key=(CONFIG["key"]))

        json_data, *_ = ts.get_daily_adjusted(symbol, outputsize=(CONFIG["outputsize"]))

        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(json_data, ensure_ascii=False, indent=2))

    dates: List[str] = list(json_data.keys())
    dates.reverse()

    close_col_name = CONFIG["key_adjusted_close"]
    adjusted_close_prices: List[float] = [float(json_data[date][close_col_name]) for date in json_data.keys()]
    adjusted_close_prices.reverse()

    prices: NDArray[Shape["*"], Float] = np.array(adjusted_close_prices)
    assert_isinstance(prices, NDArray[Shape["*"], Float])

    return PriceHistory(symbol, dates, prices)
