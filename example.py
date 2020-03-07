import numpy as np
import pandas as pd
import streamlit as st

st.markdown("# Ejemplo de app")

#df = pd.read_csv("datasets/toxic_language/test.csv")

st.text("Esto es un texto")

#df.head()
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df

chart_data = pd.DataFrame(
     np.random.randn(20, 3),
     columns=['a', 'b', 'c'])

st.line_chart(chart_data)



map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)



if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)


option = st.sidebar.selectbox(
    'Which number do you like best?',
     df['first column'])

'You selected:', option
