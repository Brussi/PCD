import streamlit as st

def main():
    options = ['Option 1', 'Option 2', 'Option 3']

    selected_option = st.selectbox('Select an option:', options)

    if selected_option == 'Option 1':
        selected_radio = st.radio('Choose an item:', ['Item A', 'Item B', 'Item C'])
        st.write('You selected:', selected_option, 'and', selected_radio)
    elif selected_option == 'Option 2':
        selected_radio = st.radio('Choose an item:', ['Item X', 'Item Y', 'Item Z'])
        st.write('You selected:', selected_option, 'and', selected_radio)
    else:
        st.write('You selected:', selected_option)

if __name__ == '__main__':
    main()