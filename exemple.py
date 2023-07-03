import streamlit as st
import pandas as pd

# Load the dataset
@st.cache  # Cache the dataset for improved perfor mance
def load_data(dir_="PCD_data/"):
    data = pd.read_csv(dir_+'your_dataset.csv')  # Replace with the path to your dataset
    return data

# Main function
def main():
    # Set app title and description
    st.title("Dataset Explorer")
    st.write("Select a dataset to explore its contents.")

    # Load the dataset
    data = load_data()

    # Display dataset selection dropdown
    dataset_name = st.selectbox("Select Dataset", data.keys())

    # Display the selected dataset
    st.write("Selected Dataset:", dataset_name)
    st.write(data[dataset_name])

if __name__ == '__main__':
    main()