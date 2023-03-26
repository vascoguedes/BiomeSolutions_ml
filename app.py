import streamlit as st
from flask import Flask, request, jsonify
from model import get_products_needed

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    from flask import Flask

    app = Flask(__name__)

    @app.route('/getProducts')
    def serve_foo():
        input_data = request.args

        # split by using ',' 
        items = input_data['clientVals'].split(",")

        new_list = []

        # add items to new list
        for item in items:
            new_list.append(int(item.replace("[", "").replace("]", "")))

        output_data = get_products_needed(new_list, input_data['crop'], int(input_data['area']))

        return jsonify(output_data)

    app.run(port=8888)


# We'll never reach this part of the code the first time this file executes!

# Your normal Streamlit app goes here:
x = st.slider('Pick a number')
st.write('You picked:', x)