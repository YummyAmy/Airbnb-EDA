import streamlit as st
from PIL import Image

#set page configuration
st.set_page_config(
    page_title = "Home",
    page_icon = ":house:",
    layout = "wide",
    initial_sidebar_state = "expanded"

)
#set theme
hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''

#define function for each page
def home():
    
    st.markdown("# Exploratory Data Analysis on AirBnB Data")
    st.write(""" 
        A 2-week project from [Corise](https://corise.com/course/intro-to-numpy-and-pandas) using Numpy and Pandas for exploratory data analysis.""")
    st.write("""
        As a Data Scientist for the Amsterdam area at Airbnb, the objective is to help visitors to make an informed choice of which airbnb to stay by analyzing the data close to a location they would like to visit. This is an analytical platform with resources like price range and type of rooms for users to choose from. It includes two interactive pages and map visualizations that allows users to explore different aspects of the airbnb listings such as price, location, and availability""")
    img = Image.open("Images/Luxury-Airbnb-Apartment-Amsterdam.jpg")
    new_image = img.resize((900,600))
    st.image(new_image, caption = 'Luxury Airbnb apartment', use_column_width="always")
    st.markdown(
        """
       ### Summary of projects:
       ## Week 1
        - Exploratory data analysis using Numpy
        - Data cleaning and analysis
        - Exploring Amsterdam AirBnB dataset using metres from the chosen location and price range as filters. (Use sidebar sliders to filter data)
        - Include a download option for the filtered dataframe
        - Web app development and deployment on streamlit
        
        ## Week 2
        - Exploratory Data Analysis Using Pandas
        - Interactive visuals for exploring different filters for AirBnB listings such as Neighborhood, bedrooms, beds etc.
        - Map visualization
        - Added a multiselect option to filter dataframe
        - Include a theme and multi-page configuration
        - Include a download option for the filtered CSV dataframe
        - Web app development and deployment
        """
        )
    
    st.sidebar.title("Connect with me")
    st.sidebar.markdown("[Linktree](https://linktr.ee/ameusifoh)")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ameti-obong-u-395a25111/)")
<<<<<<< HEAD
   #st.sidebar.markdown('<a href="mailto:ameikpe@yahoo.com">E-mail</a>', unsafe_allow_html=True)
=======
   # st.sidebar.markdown('<a href="mailto:ameikpe@yahoo.com">E-mail</a>', unsafe_allow_html=True)
>>>>>>> 53ca55e24ce086a9fbb4487a58a6571cf4d82134
    st.sidebar.markdown("[Tableau Profile](https://public.tableau.com/app/profile/amyu)")
    st.sidebar.markdown("[CoRise Course](https://uplimit.com/course/intro-to-numpy-and-pandas)")
    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

    # call the function
if __name__ == '__main__':
    home()

#def week2():
 #   st.markdown("# Week 2")
  #  st.write("Use the dropdown below to filter data by neighborhood")
   # neighborhoods = np.sort(airbnb_df.neighbourhood_group.unique())
    #selected_neighborhood = st.selectbox("Neighborhood", neighborhoods)
    #filtered_df = airbnb_df[airbnb_df.neighbourhood_group == selected_neighborhood]
   # st.write("Filtered Data:")
  #  st.write(filtered_df)
 #   map_data = filtered_df[['latitude', 'longitude']]
#    map_data = map_data.dropna(how='any')
    #st.map(map_data)
    #st.sidebar.markdown("## Download CSV file")
   # st.sidebar.download_button(
      #  label="Download Data",
     #   data=filtered_df.to_csv(index=False),
    #    file_name="airbnb_filtered.csv",
   #     mime="text/csv",
  #  )
