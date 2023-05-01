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

#define function for each page
def home():
    
    st.markdown("# Welcome to my AirBnB Data Analysis web app")
    st.write(""" 
        A 2-week project from [Corise](https://corise.com/course/python-for-data-science). As a Data Scientist for the Amsterdam area at Airbnb, the objective is to analyze the data, make informed decisions and provide a data analysis platform with a wide range of tools and resources, including an interactive visualizations that allows you to explore different aspects of your listings such as price, location, and availability""")
    img = Image.open("Images/Luxury-Airbnb-Apartment-Amsterdam.jpg")
    new_image = img.resize((900,600))
    st.image(new_image, caption = 'Luxury Airbnb apartment', use_column_width="always")
    st.markdown(
        """
       ### Summary of projects:
       ## Week 1
        - Exploratory data analysis using Numpy
        - Data Cleaning
        - Web app development and deployment on streamlit
        - Exploring Amsterdam AirBnB dataset using metres to location and price range as filters. (Use 2 sidebar sliders to filter data)
        - Used the experimental data editor and download option for the dataframe.
        
        ## Week 2
        - Exploratory Data Analysis Using Pandas
        - Interactive visuals for exploring different filters for AirBnB listings such as Neighborhood, bedrooms, beds etc.
        - Web app development and deployment
        - Map visualization and ability to change map styles
        - Ability to change colour marker on map based on column data.
        - Added background image
        - Changed css styling, theme and page configurations
        - Ability to download filtered CSV file
        """
        )
    
    st.sidebar.title("Connect with me")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ameti-obong-e-395a25111)")
    st.sidebar.markdown('<a href="mailto:ameikpe@yahoo.com">E-mail</a>', unsafe_allow_html=True)
    st.sidebar.markdown("[CoRise Course](https://corise.com/course/python-for-data-science)")

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