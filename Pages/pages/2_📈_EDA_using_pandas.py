import pandas as pd
import streamlit as st
import plotly.express as px
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype
)

st.set_page_config(
    page_title = "Week 2",
    page_icon = ":bar_chart",
    layout = "wide"
)

#Display title and text
st.markdown("# Data Analysis and visualization using Pandas")
st.write(
    """This web page app is based on this blog [here](https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/). 
    """
)
st.write("Use the filter on the sidebar to filter the listings by color on the map. You can also use the multiselect box to filter the airbnb listings on the dataframe below the map")
df = pd.read_csv("WK2_Airbnb_Amsterdam_listings_proj_solution.csv", index_col=0)


# Filter dataframe with a UI
df_filtered = df

# Display map with filtered data
selected_column = st.sidebar.selectbox("Select column and hover on map to show data", df.columns)
fig = px.scatter_mapbox(
    df, 
    lat="latitude", 
    lon="longitude", 
    color=selected_column, 
    color_continuous_scale="Viridis", 
    zoom=11,
    height=600,
    width=1000,
    hover_name="price_in_dollar",
    hover_data=["amenities", "price_in_dollar", "five_day_dollar_price"]
)

fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=False)

# Display filtered data in a table
st.write("Filtered Airbnb Listings dataframe:")

#st.dataframe(df_filtered) #using st.write to show a df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    Args:
        df (pd.DataFrame): Original dataframe
    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Click to add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    _min,
                    _max,
                    (_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].str.contains(user_text_input)]

    return df

df = pd.read_csv(
    "WK2_Airbnb_Amsterdam_listings_proj_solution.csv", index_col=0
)
st.dataframe(filter_dataframe(df))

# Download dataframe
st.sidebar.download_button(
    label="Download filtered dataframe",
    data=df.to_csv(index=False),
    file_name="WK2_Airbnb_Amsterdam_listings_proj_solution.csv",
    mime="text/csv",)