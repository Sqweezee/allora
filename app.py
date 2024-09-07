from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from granite import GraniteModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create our Flask app
app = Flask(__name__)

# Define the Hugging Face model we will use
model_name = "ibm-granite/granite-timeseries-ttm-v1"

def get_coingecko_url(token):
    base_url = "https://api.coingecko.com/api/v3/coins/"
    token_map = {
        'ETH': 'ethereum',
        'SOL': 'solana',
        'BTC': 'bitcoin',
        'BNB': 'binancecoin',
        'ARB': 'arbitrum'
    }
    
    token = token.upper()
    if token in token_map:
        url = f"{base_url}{token_map[token]}/market_chart?vs_currency=usd&days=30&interval=daily"
        return url
    else:
        raise ValueError("Unsupported token")

@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for given token."""
    try:
        # Load the model and tokenizer
        model = GraniteModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)  # Adjust if different
        model.eval()
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        return Response(json.dumps({"model error": str(e)}), status=500, mimetype='application/json')

    try:
        # Get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        logger.error(f"URL generation error: {e}")
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": "CG-ts4JYFHPiNtfFkn7F88EgR2s" 
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"API request error: {e}")
        return Response(json.dumps({"Failed to retrieve data from the API": str(e)}), 
                        status=500, 
                        mimetype='application/json')

    # Process the data
    try:
        df = pd.DataFrame(data["prices"])
        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], unit='ms')
        df = df[:-1]  # Removing today's price

        if df.empty:
            raise ValueError("No historical data available")

        logger.info(f"Data retrieved: {df.tail(5)}")

        # Prepare data for the model
        prices = df["price"].values.astype(float)
        context = torch.tensor(prices).unsqueeze(0)  # Adding batch dimension
        
        # Make prediction
        with torch.no_grad():
            forecast = model(context)
        
        forecast_mean = forecast.mean().item()  # Adjust if model output is different

        return Response(str(forecast_mean), status=200)
    except Exception as e:
        logger.error(f"Data processing or prediction error: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# Run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
