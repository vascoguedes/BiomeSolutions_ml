import streamlit as st
import pandas as pd
import numpy as np
from model import get_products_needed



result = get_products_needed([10, 10, 10], 'rice', 8)
st.text(result)