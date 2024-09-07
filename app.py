from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
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
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        forecasting_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        return Response(json.dumps({"pipeline error": str(e)}), status=500, mimetype='application/json')

    try:
        # Get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        logger.error(f"URL generation error: {e}")
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        # Replace with your API key or manage securely
        "x-cg-demo-api-key": "CG-ts4JYFHPiNtfFkn7F88EgR2s" 
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
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
        context = df["price"].tolist()
        context_str = ' '.join(map(str, context))
        
        # Define the prediction length
        prediction_length = 1
        
        # Make prediction
        forecast = forecasting_pipeline(context_str, max_length=prediction_length)
        forecast_mean = forecast[0]['generated_text']

        return Response(str(forecast_mean), status=200)
    except Exception as e:
        logger.error(f"Data processing or prediction error: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# Run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
