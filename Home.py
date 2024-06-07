import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Home",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set theme to hide the decoration bar
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''

# Define function for home page
def home():
    st.markdown("# Exploratory Data Analysis on AirBnB Data")
    st.write(""" 
        A 2-week project from [Corise](https://corise.com/course/intro-to-numpy-and-pandas) using Numpy and Pandas for exploratory data analysis.
    """)
    st.write("""
        As a Data Scientist for the Amsterdam area at Airbnb, the objective is to help visitors make an informed choice of which Airbnb to stay by analyzing the data close to a location they would like to visit. This is an analytical platform with resources like price range and type of rooms for users to choose from. It includes two interactive pages and map visualizations that allows users to explore different aspects of the Airbnb listings such as price, location, and availability.
    """)
    try:
        img = Image.open("Images/Luxury-Airbnb-Apartment-Amsterdam.jpg")
        new_image = img.resize((900, 600))
        st.image(new_image, caption='Luxury Airbnb apartment', use_column_width="always")
    except FileNotFoundError:
        st.error("Image not found. Please check the file path.")

    st.markdown(
        """
       ### Summary of Projects:
       ## Week 1
        - Exploratory data analysis using Numpy
        - Data cleaning and analysis
        - Exploring Amsterdam Airbnb dataset using meters from the chosen location and price range as filters. (Use sidebar sliders to filter data)
        - Include a download option for the filtered dataframe
        - Web app development and deployment on Streamlit
        
        ## Week 2
        - Exploratory Data Analysis Using Pandas
        - Interactive visuals for exploring different filters for Airbnb listings such as Neighborhood, bedrooms, beds, etc.
        - Map visualization
        - Added a multiselect option to filter dataframe
        - Include a theme and multi-page configuration
        - Include a download option for the filtered CSV dataframe
        - Web app development and deployment
        """
    )

    st.sidebar.title("Connect")
    st.sidebar.markdown("[Linktree](https://linktr.ee/ameusifoh)")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ameti-obong-u-395a25111/)")
    st.sidebar.markdown("[Tableau Profile](https://public.tableau.com/app/profile/amyu)")
    st.sidebar.markdown("[CoRise Course](https://uplimit.com/course/intro-to-numpy-and-pandas)")

    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Call the function
if __name__ == '__main__':
    home()
