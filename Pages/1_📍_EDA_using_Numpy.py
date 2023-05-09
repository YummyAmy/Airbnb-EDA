import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title = "Week 1",
    page_icon="📈"":round_pushpin:",
    layout = "wide"
)

# Display title and text
st.title("Week 1 - Data analysis and visualization using Numpy")
st.markdown("The map below shows all Airbnb listings in Amsterdam, including a chosen location. The Red dots represent listings close to the chosen location and the light blue dot represent the chosen location to visit. Use the sliders to filter the data based on price and meters from chosen location.")

# Read dataframe
dataframe = pd.read_csv(
    "/Users/yummy/Desktop/CoRise/WK1_Airbnb_Amsterdam_listings_proj_solution.csv",
    names=[
        "Airbnb Listing ID",
        "Price",
        "Latitude",
        "Longitude",
        "Meters from chosen location",
        "Location",
    ],
)

# We have a limited budget, therefore we would like to exclude listings with a price above 200 CAD_usd per night
dataframe = dataframe[dataframe["Price"] <= 200]

#Add sliders
price_range = st.sidebar.slider("Price Range", min_value=0.00, max_value=200.00, value=(0.00,200.00), step = 10.00)#, format = "%f")
meters_from_chosen_location = st.sidebar.slider("Meters from chosen location", min_value=0.0, max_value=16500.0, value=(0.0, 16000.0))

#filter dataframe(map) based on sliders values
dataframe = dataframe[
    (dataframe["Price"] >= price_range[0])
    & (dataframe["Price"] <= price_range[1])
    & (dataframe["Meters from chosen location"] >= meters_from_chosen_location[0])
    & (dataframe["Meters from chosen location"] <= meters_from_chosen_location[1])
]

# Display as integer
dataframe["Airbnb Listing ID"] = dataframe["Airbnb Listing ID"].astype(int)

# Round off values
#dataframe['Price'] = dataframe['Price'].str.replace('$ ', '').astype(float)
# Round of values
dataframe["Price"] = "$ " + dataframe["Price"].round(2).astype(str) # <--- CHANGE THE POUND SYMBOL TO CAD

dataframe["Meters from chosen location"] = dataframe["Meters from chosen location"].round(2).astype(int)

# Rename the number to a string
dataframe["Location"] = dataframe["Location"].replace(
    {1.0: "To visit", 0.0: "Airbnb listing"}
)

# Filter the dataframe based on the slider values
#filtered_dataframe = dataframe[
 #   (dataframe["Price"].str.replace("$", "").str.replace(" ", "").astype(float) >= price_range[0]) &
 #   (dataframe["Price"].str.replace("$", "").str.replace(" ", "").astype(float) <= price_range[1]) &
  #  (dataframe["Meters from chosen location"] >= meters_from_chosen_location[0]) &
   # (dataframe["Meters from chosen location"] <= meters_from_chosen_location[1])
#]


### mycodedataframe["Meters from chosen location"] = dataframe["Meters from chosen location"].round(2).astype(str)
#trial and error
#dataframe["Location"] = df1["Location"].replace(
#    {1.0: "To visit", 0.0: "Airbnb listing"}
#)
# Create the plotly express figure map 1
fig = px.scatter_mapbox(
    dataframe,
    lat="Latitude",
    lon="Longitude",
    color="Location",
    color_discrete_sequence=["blue", "red"],
    zoom=11,
    height=500,
    width=800,
    hover_name="Price",
    hover_data=["Meters from chosen location", "Location"],
    labels={"color": "Locations"},
)
fig.update_geos(center=dict(lat=dataframe.iloc[0][2], lon=dataframe.iloc[0][3]))
fig.update_layout(mapbox_style="stamen-terrain")

# Show the figure
st.plotly_chart(fig, use_container_width=True)


# Download dataframe
st.download_button(
    label="Download dataframe",
    data=dataframe.to_csv(index=False),
    file_name="WK1_Airbnb_Amsterdam_listings_proj_solution.csv",
    mime="text/csv",)

# Display dataframe and text
st.write("We have a limited budget, therefore we would like to exclude listings with a price above 200 CAD_usd per night")
st.dataframe(dataframe)
