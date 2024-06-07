import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Function to load data from GitHub
def load_data_from_github():
    base_url = "https://raw.githubusercontent.com/YummyAmy/ML-and-Exploratory-Data-Analysis-Bootcamp/main/archive%20(3)/"
    file_paths = {
        'circuits': base_url + 'circuits.csv',
        'constructor_results': base_url + 'constructor_results.csv',
        'constructor_standings': base_url + 'constructor_standings.csv',
        'constructors': base_url + 'constructors.csv',
        'driver_standings': base_url + 'driver_standings.csv',
        'races': base_url + 'races.csv',
        'qualifying': base_url + 'qualifying.csv',
        'pit_stops': base_url + 'pit_stops.csv',
        'lap_times': base_url + 'lap_times.csv',
        'drivers': base_url + 'drivers.csv',
        'status': base_url + 'status.csv',
        'sprint_results': base_url + 'sprint_results.csv',
        'seasons': base_url + 'seasons.csv',
        'results': base_url + 'results.csv'
    }

    dataframes = {name: pd.read_csv(url) for name, url in file_paths.items()}
    return dataframes

# Set page configuration
st.set_page_config(
    page_title="Home",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tabs at the top
tab1, tab2, tab3 = st.tabs(["Home", "Decision Modeling with Formula 1 Datasets", "Airbnb Listings Analysis"])

with tab1:
    st.markdown("# Machine Learning and Exploratory Data Analysis Platform")
    st.write("""
        This platform hosts multiple exploratory data analysis projects:
        1. **Decision Modeling on Formula 1 Drivers**
        2. **Airbnb Listings Analysis**
    """)
    st.write("""
        ## F1 Driver Performance Analysis
        Creating a decision model to support the hypothesis on Formula 1 drivers' performances. This section includes metrics such as win rates, race participation, and performance over time. It leverages regression models to predict win rates and provides insightful visualizations.
    """)

    try:
        img_f1 = Image.open("Images for F1/Ferrari_Formula_1_lineup_at_the_Nürburgring.jpg")
        new_image_f1 = img_f1.resize((800, 400))
        st.image(new_image_f1, caption='F1 Drivers', use_column_width="always")
    except FileNotFoundError:
        st.error("F1 image not found. Please check the file path.")

    st.write("""
        ## Airbnb Listings Analysis
        This analysis helps visitors make informed choices about Airbnb listings in Amsterdam by examining various aspects such as price, location, and availability. It includes interactive visualizations and filter options to explore the data.
    """)

    try:
        img_airbnb = Image.open("Images/Luxury-Airbnb-Apartment-Amsterdam.jpg")
        new_image_airbnb = img_airbnb.resize((800, 400))
        st.image(new_image_airbnb, caption='Luxury Airbnb apartment', use_column_width="always")
    except FileNotFoundError:
        st.error("Airbnb image not found. Please check the file path.")

    st.markdown(
        """
        ### Summary of Projects:
        ## F1 Driver Performance Analysis
        - Exploratory Analysis of Formula 1 datasets
        - Correlation Matrices of datasets
        - Evaluating driver performance metrics
        - Regression models to predict win rates
        - Visualizations of driver statistics and performance trends
        - Export functionality for top driver data

        ## Airbnb Listings Analysis
        ### Week 1
        - Exploratory data analysis using Numpy
        - Data cleaning and analysis
        - Exploring Amsterdam Airbnb dataset using filters (location and price range)
        - Download option for filtered data
        
        ### Week 2
        - Exploratory data analysis using Pandas
        - Interactive visuals for Airbnb listing using filters
        - Map visualization
        - Multiselect filter options
        - Theme and multi-page configuration
        - Download option for filtered CSV data
        - Web app development and deployment
        """
    )

    st.sidebar.title("Connect")
    st.sidebar.markdown("[Linktree](https://linktr.ee/ameusifoh)")
    st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ameti-obong-u-395a25111/)")
    st.sidebar.markdown("[Clicked](https://www.clicked.com/browse-experiences)")
    # st.sidebar.markdown('<a href="mailto:ameikpe@yahoo.com">E-mail</a>', unsafe_allow_html=True)
    # st.sidebar.markdown("[Tableau Profile](https://public.tableau.com/app/profile/amyu)")
    st.sidebar.markdown("[CoRise Course](https://uplimit.com/course/intro-to-numpy-and-pandas)")
    # st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# Tab 2: F1 Analysis
with tab2:
    st.header("F1 Analysis")

    # Load data
    dataframes = load_data_from_github()

    # Inspect data
    # st.subheader("Data Inspection")
    # def inspect_data(dataframes):
    #     inspection_results = {}
    #     for name, df in dataframes.items():
    #         inspection_results[name] = {
    #             'shape': df.shape,
    #             'missing_values': df.isnull().sum(),
    #         }
    #     return inspection_results

    # inspection_results = inspect_data(dataframes)
    # st.write("Data Inspection Results:")
    # st.write(inspection_results)

    # Driver nationality distribution
    st.subheader("Exploration of datasets")
    st.subheader("Driver Nationality Distribution")
    drivers = dataframes['drivers']
    nationality_distribution = drivers['nationality'].value_counts()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=nationality_distribution.values, y=nationality_distribution.index)
    plt.title('Nationality Distribution of Drivers')
    plt.xlabel('Number of drivers')
    plt.ylabel('Nationality')
    st.pyplot(plt)

    # Aggregate total points for each driver
    st.subheader("Top 10 Drivers by Total Points")
    driver_points = dataframes['results'].groupby('driverId')['points'].sum().reset_index()
    driver_points_sorted = driver_points.sort_values(by='points', ascending=False)
    top_10_drivers = driver_points_sorted.head(10)
    top_10_drivers_details = top_10_drivers.merge(dataframes['drivers'], on='driverId')
    driver_names = top_10_drivers_details['forename'] + ' ' + top_10_drivers_details['surname']
    driver_points = top_10_drivers_details['points']

    plt.figure(figsize=(6, 3))
    plt.barh(driver_names, driver_points, color='skyblue')
    plt.xlabel('Total Points')
    plt.ylabel('Drivers')
    plt.title('Top 10 Drivers by Total Points')
    plt.gca().invert_yaxis()
    st.pyplot(plt)

    # Driver performance by constructor
    st.subheader("Driver Performance by Constructor")
    results = dataframes['results']
    constructors = dataframes['constructors']
    driver_constructor_points = results.groupby(['driverId', 'constructorId'])['points'].sum().reset_index()
    driver_constructor_points = driver_constructor_points.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    driver_constructor_points = driver_constructor_points.merge(constructors[['constructorId', 'name']], on='constructorId')
    top_10_drivers = driver_constructor_points.groupby('driverId')['points'].sum().reset_index().sort_values(by='points', ascending=False).head(10)
    top_10_driver_constructor_points = driver_constructor_points[driver_constructor_points['driverId'].isin(top_10_drivers['driverId'])]

    plt.figure(figsize=(8, 6))
    sns.barplot(x='surname', y='points', hue='name', data=top_10_driver_constructor_points)
    plt.title('Points Scored by Top 10 Drivers for Each Constructor')
    plt.xlabel('Driver')
    plt.ylabel('Total Points')
    plt.legend(title='Constructor', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt)

    # Points scored by top 10 drivers over seasons
    st.subheader("Points Scored by Top 10 Drivers Over Seasons")
    top_10_driver_ids = top_10_drivers['driverId'].tolist()
    top_10_results = dataframes['results'][dataframes['results']['driverId'].isin(top_10_driver_ids)]
    top_10_results = top_10_results.merge(dataframes['races'][['raceId', 'year']], on='raceId')
    driver_season_points = top_10_results.groupby(['driverId', 'year'])['points'].sum().reset_index()
    driver_season_points = driver_season_points.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    driver_season_pivot = driver_season_points.pivot(index='year', columns='surname', values='points')

    plt.figure(figsize=(8, 5))
    driver_season_pivot.plot(kind='line', marker='o', ax=plt.gca())
    plt.title('Points Scored by Top 10 Drivers Over Seasons')
    plt.xlabel('Season')
    plt.ylabel('Total Points')
    plt.legend(title='Driver')
    plt.grid(True)
    st.pyplot(plt)

    # Win rates of top 5 drivers
    st.subheader("Win Rates of Top 5 Drivers")
    driver_races = results.groupby('driverId')['raceId'].count().reset_index()
    driver_races.rename(columns={'raceId': 'total_races'}, inplace=True)
    driver_wins = results[results['positionOrder'] == 1].groupby('driverId')['raceId'].count().reset_index()
    driver_wins.rename(columns={'raceId': 'total_wins'}, inplace=True)
    driver_performance = driver_races.merge(driver_wins, on='driverId', how='left')
    driver_performance['total_wins'].fillna(0, inplace=True)
    driver_performance['win_rate'] = driver_performance['total_wins'] / driver_performance['total_races']
    top_5_drivers = driver_performance.sort_values(by='total_wins', ascending=False).head(5)
    top_5_drivers = top_5_drivers.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')
    top_5_drivers = top_5_drivers.sort_values(by='win_rate', ascending=True)

    plt.figure(figsize=(5, 4))
    sns.barplot(x='surname', y='win_rate', data=top_5_drivers, palette='viridis')
    plt.title('Win Rates of Top 5 Drivers')
    plt.xlabel('Driver')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)
    plt.tight_layout()
    st.pyplot(plt)

    # Correlation matrices
    st.subheader("Correlation Matrices")
    results_numeric = results.select_dtypes(include=['float64', 'int64'])
    corr = results_numeric.corr()

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, cbar_kws={"shrink": .8})
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j+0.5, i+0.5, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', color='black')
    plt.title('Correlation Matrix for Results')
    st.pyplot(plt)

    # Pairplots
    st.subheader("Pairplots for distribution of Results")
    sns.pairplot(results)
    plt.suptitle("Pairplot for Results", y=1.02)
    st.pyplot(plt)

    # Relationship between pitstops and wins
    st.subheader("Relationship Between Pitstops and Wins")
    pit_stops = dataframes['pit_stops']
    avg_pit_stop_duration = pit_stops.groupby(['raceId', 'driverId'])['milliseconds'].mean().reset_index(name='avg_pit_stop_duration')
    performance = results.merge(avg_pit_stop_duration, on=['raceId', 'driverId'])
    correlation = performance['avg_pit_stop_duration'].corr(performance['positionOrder'])
    st.write(f"Correlation between Average Pit Stop Duration and Race Position Order: {correlation}")

    plt.figure(figsize=(8, 4))
    plt.scatter(performance['avg_pit_stop_duration'], performance['positionOrder'], alpha=0.6)
    plt.title('Correlation Between Average Pit Stop Duration and Race Position Order')
    plt.xlabel('Average Pit Stop Duration (milliseconds)')
    plt.ylabel('Race Position Order')
    z = np.polyfit(performance['avg_pit_stop_duration'], performance['positionOrder'], 1)
    p = np.poly1d(z)
    plt.plot(performance['avg_pit_stop_duration'], p(performance['avg_pit_stop_duration']), "r--")
    plt.grid(True)
    st.pyplot(plt)

    # Linear regression
    st.subheader("Linear Regression")
    driver_performance['dob'] = pd.to_datetime(driver_performance['dob'])
    driver_performance['age'] = 2024 - driver_performance['dob'].dt.year
    features = ['total_races', 'total_wins', 'age']
    target = 'win_rate'
    X = driver_performance[features]
    y = driver_performance[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")
    coefficients = pd.DataFrame(model.coef_, features, columns=['Coefficient'])
    st.write("Model Coefficients:")
    st.write(coefficients)

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title('Predicted vs. Actual Win Rates')
    plt.xlabel('Actual Win Rate')
    plt.ylabel('Predicted Win Rate')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.tight_layout()
    st.pyplot(plt)

    # Decision Tree Model
    st.subheader("Decision Tree Model")
    results['win'] = results['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
    data = results.merge(constructors, on='constructorId')
    features = ['grid', 'laps', 'milliseconds', 'fastestLapSpeed']
    target = 'win'
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
    data[features] = data[features].fillna(data[features].mean())
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    max_depth = 3
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.write(f'Precision: {precision:.2f}')
    st.write(f'Recall: {recall:.2f}')
    st.write('Confusion Matrix:')
    st.write(conf_matrix)

    plt.figure(figsize=(14, 5))
    plot_tree(model, feature_names=features, class_names=['Not Win', 'Win'], filled=True, rounded=True, fontsize=10)
    st.pyplot(plt)

# Tab 3: Airbnb Listings Analysis
with tab3:
    st.header("Airbnb Listings Analysis")

    # Load Week_1.py and Week_2.py as separate sections within the tab
    exec(open("pages/Week_1.py").read())
    exec(open("pages/Week_2.py").read())

# Call the function
if __name__ == '__main__':
    st.set_page_config(page_title="Machine Learning and Exploratory Data Analysis Platform", layout="wide")

# import streamlit as st
# from PIL import Image

# # Set page configuration
# st.set_page_config(
#     page_title="Home",
#     page_icon=":house:",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Set theme to hide the decoration bar
# hide_decoration_bar_style = '''
#     <style>
#         header {visibility: hidden;}
#     </style>
# '''

# # Define function for home page
# def home():
#     st.markdown("# Exploratory Data Analysis on AirBnB Data")
#     st.write(""" 
#         A 2-week project from [Corise](https://corise.com/course/intro-to-numpy-and-pandas) using Numpy and Pandas for exploratory data analysis.
#     """)
#     st.write("""
#         As a Data Scientist for the Amsterdam area at Airbnb, the objective is to help visitors make an informed choice of which Airbnb to stay by analyzing the data close to a location they would like to visit. This is an analytical platform with resources like price range and type of rooms for users to choose from. It includes two interactive pages and map visualizations that allows users to explore different aspects of the Airbnb listings such as price, location, and availability.
#     """)
#     try:
#         img = Image.open("Images/Luxury-Airbnb-Apartment-Amsterdam.jpg")
#         new_image = img.resize((900, 600))
#         st.image(new_image, caption='Luxury Airbnb apartment', use_column_width="always")
#     except FileNotFoundError:
#         st.error("Image not found. Please check the file path.")

#     st.markdown(
#         """
#        ### Summary of Projects:
#        ## Week 1
#         - Exploratory data analysis using Numpy
#         - Data cleaning and analysis
#         - Exploring Amsterdam Airbnb dataset using meters from the chosen location and price range as filters. (Use sidebar sliders to filter data)
#         - Include a download option for the filtered dataframe
#         - Web app development and deployment on Streamlit
        
#         ## Week 2
#         - Exploratory Data Analysis Using Pandas
#         - Interactive visuals for exploring different filters for Airbnb listings such as Neighborhood, bedrooms, beds, etc.
#         - Map visualization
#         - Added a multiselect option to filter dataframe
#         - Include a theme and multi-page configuration
#         - Include a download option for the filtered CSV dataframe
#         - Web app development and deployment
#         """
#     )

#     st.sidebar.title("Connect")
#     st.sidebar.markdown("[Linktree](https://linktr.ee/ameusifoh)")
#     st.sidebar.markdown("[LinkedIn](https://www.linkedin.com/in/ameti-obong-u-395a25111/)")
#     st.sidebar.markdown("[Tableau Profile](https://public.tableau.com/app/profile/amyu)")
#     st.sidebar.markdown("[CoRise Course](https://uplimit.com/course/intro-to-numpy-and-pandas)")

#     st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

# # Call the function
# if __name__ == '__main__':
#     home()
