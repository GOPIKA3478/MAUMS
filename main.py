from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import warnings
import io
import base64

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data
df = pd.read_csv("medicine_sales.csv", parse_dates=["date"])
products = df['product'].unique()

@app.route("/", methods=["GET", "POST"])
def dindex():
    selected_product = None
    forecast_df = None
    plot_url = None

    if request.method == "POST":
        selected_product = request.form.get("product_name").strip()

        product_df = df[df['product'] == selected_product]
        if len(product_df) < 10:
            forecast_df = [{"Date": "Error", "Forecasted Quantity": "Not enough data"}]
        else:
            daily_sales = product_df.groupby('date')['quantity'].sum().asfreq('D').fillna(0)

            # Train ARIMA model
            stepwise_model = auto_arima(daily_sales, seasonal=False, suppress_warnings=True)
            model = ARIMA(daily_sales, order=stepwise_model.order)
            model_fit = model.fit()

            # Forecast next 7 days
            n_days = 7
            forecast = model_fit.forecast(steps=n_days)
            forecast_index = pd.date_range(start=daily_sales.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')

            # Create forecast dataframe
            forecast_df = pd.DataFrame({
                "Date": forecast_index.strftime('%Y-%m-%d'),
                "Forecasted Quantity": np.round(forecast).astype(int)
            }).to_dict(orient="records")

            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(daily_sales.index, daily_sales.values, label="Historical Demand")
            plt.plot(forecast_index, forecast, label="7-Day Forecast", color="red")
            plt.title(f"{selected_product} - 7-Day Demand Forecast")
            plt.xlabel("Date")
            plt.ylabel("Quantity")
            plt.legend()
            plt.tight_layout()

            # Convert plot to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            plt.close()

    return render_template("dindex.html",
                           products=products,
                           selected_product=selected_product,
                           forecast_df=forecast_df,
                           plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True,port =5002)
