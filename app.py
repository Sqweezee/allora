from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create our Flask app
app = Flask(__name__)

# Define the Hugging Face model we will use
model_name = "Salesforce/moirai-1.0-R-large"

# Load the model and tokenizer once when the app starts
try:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Model initialization error: {e}")
    raise RuntimeError("Model initialization failed.")

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
        # Get the data from Coingecko
        url = get_coingecko_url(token)
    except ValueError as e:
        logger.error(f"URL generation error: {e}")
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": os.getenv("COINGECKO_API_KEY")  # Use environment variable for API key
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error(f"API request error: {e}")
        return Response(json.dumps({"error": "Failed to retrieve data from the API", "details": str(e)}), 
                        status=500, 
                        mimetype='application/json')

    # Process the data
    try:
        df = pd.DataFrame(data.get("prices", []))
        if df.empty or df.shape[1] != 2:
            raise ValueError("Invalid data format received from API")

        df.columns = ["date", "price"]
        df["date"] = pd.to_datetime(df["date"], unit='ms')
        df = df[:-1]  # Removing today's price

        if df.empty:
            raise ValueError("No historical data available")

        logger.info(f"Data retrieved: {df.tail(5)}")

        # Prepare data for the model
        prices = df["price"].values.astype(float)
        prices_str = ' '.join(map(str, prices))  # Convert prices to a space-separated string
        
        # Tokenize input
        inputs = tokenizer(prices_str, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return Response(json.dumps({"generated_text": generated_text}), status=200, mimetype='application/json')
    except Exception as e:
        logger.error(f"Data processing or prediction error: {e}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# Run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
