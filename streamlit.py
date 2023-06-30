import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define the function for contour plotting
def plot_contour():
    # Generate some data for contour plot
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    # Create the contour plot
    fig, ax = plt.subplots()
    contour = ax.contour(X, Y, Z)

    # Set plot labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Contour Plot')

    # Display the plot
    st.pyplot(fig)

# Streamlit app
def main():
    st.title('Contour Plot with Streamlit')
    st.write('Example of a contour plot using Streamlit and Matplotlib')

    # Plot the contour
    plot_contour()

if __name__ == '__main__':
    main()