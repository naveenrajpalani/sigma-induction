import streamlit as st

def main():
    st.set_page_config(page_title="My App", page_icon=":rocket:")
    
    st.title("My Streamlit App")
    st.write("Welcome to my app deployed on Streamlit Community Cloud!")
    
    # Add your app functionality here
    user_input = st.text_input("Enter something")
    if user_input:
        st.write(f"You entered: {user_input}")

if __name__ == "__main__":
    main()
