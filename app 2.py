import streamlit as st
from flask import Flask, request, jsonify
from model import get_products_needed

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    # Get the input data from the request
    input_data = request.get_json()

    # Call the Streamlit app's function with the input data
    output_data = my_streamlit_app([10, 10, 10], 'rice', 8)

    # Return the output data as the response
    return jsonify(output_data)

def my_streamlit_app(input_data):
    # Move the Streamlit app's code into this function
    # Modify the Streamlit app's code to use the input data
    # Return the output data
    return "Vasco"


if __name__ == '__main__':
    # Start the Flask app
    app.run()
