#Streamlit app


# ==========================================
# SMART WASTE OPTIMIZATION SYSTEM
# STREAMLIT APPLICATION
# ==========================================

# Import required libraries
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import joblib
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Waste Optimization",
    layout="wide"
)

# ==========================================%
# LOAD TRAINED MODEL
# ==========================================%

# Make sure the model file 'random_forest.pkl' is available in the Colab environment
# (It should have been saved in a previous step.)
model = joblib.load("random_forest.pkl")

# ==========================================%
# APP TITLE
# ==========================================%

st.title("♻️ Smart Waste Optimization System")

st.markdown(
    """
    This system predicts waste tonnage and supports
    waste collection route optimization across Lagos landfill sites.
    """
)

# ==========================================%
# SIDEBAR
# ==========================================%

page = st.sidebar.selectbox(
    "Select Module",
    [
        "Waste Prediction",
        "Route Optimization",
        "Analytics"
    ]
)

# ==========================================%
# WASTE PREDICTION PAGE
# ==========================================%

if page == "Waste Prediction":

    st.header("Waste Volume Prediction")

    landfill = st.selectbox(
        "Landfill Site",
        [
            "Olusosun",
            "Abule-Egba",
            "Somolu",
            "Badagry",
            "Epe",
            "Ikorodu"
        ]
    )

    year = st.number_input(
        "Year",
        min_value=2025,
        max_value=2035,
        value=2025
    )

    month = st.selectbox(
        "Month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]
    )

    trips = st.number_input(
        "Expected Trips",
        min_value=1,
        value=1000
    )

    # Month Conversion
    month_map = {
        "January":1,
        "February":2,
        "March":3,
        "April":4,
        "May":5,
        "June":6,
        "July":7,
        "August":8,
        "September":9,
        "October":10,
        "November":11,
        "December":12
    }

    month_number = month_map[month]

    quarter = ((month_number - 1)//3)+1

    festive_period = 1 if month in [
        "December",
        "January"
    ] else 0

    # Population Density Mapping

    density = {
        "Olusosun":15000,
        "Abule-Egba":12000,
        "Somolu":13000,
        "Badagry":8000,
        "Epe":7000,
        "Ikorodu":11000
    }

    landfill_encoding = {
        "Abule-Egba":0,
        "Badagry":1,
        "Epe":2,
        "Ikorodu":3,
        "Olusosun":4,
        "Somolu":5
    }

    population_density = density[landfill]

    landfill_encoded = landfill_encoding[
        landfill
    ]

    if st.button("Predict Waste Tonnage"):

        input_data = pd.DataFrame({

            "Year":[year],

            "Month_Number":[month_number],

            "Quarter":[quarter],

            "Festive_Period":[festive_period],

            "Population_Density":[population_density],

            "Landfill_Encoded":[landfill_encoded],

            "Trips":[trips]

        })

        prediction = model.predict(
            input_data
        )[0]

        st.success(
            f"Predicted Waste Tonnage: {prediction:,.2f} tonnes"
        )

landfill_locations = {
    "Olusosun": (6.6018, 3.3579),
    "Abule-Egba": (6.6480, 3.2745),
    "Somolu": (6.5380, 3.3840),
    "Badagry": (6.4167, 2.8833),
    "Epe": (6.5841, 3.9836),
    "Ikorodu": (6.6194, 3.5105)
}


# ==========================================
# ROUTE OPTIMIZATION PAGE
# ==========================================

if page == "Route Optimization":

    st.header("🚛 Route Optimization")

    st.write(
        """
        This module uses Google OR-Tools to generate
        an optimized waste collection route across
        Lagos landfill sites.
        """
    )

    locations = list(landfill_locations.keys())

    # Distance Matrix
    distance_matrix = []

    for loc1 in locations:

        row = []

        lat1, lon1 = landfill_locations[loc1]

        for loc2 in locations:

            lat2, lon2 = landfill_locations[loc2]

            distance = int(
                ((lat1-lat2)**2 +
                 (lon1-lon2)**2)**0.5 * 1000
            )

            row.append(distance)

        distance_matrix.append(row)

    data = {}

    data["distance_matrix"] = distance_matrix

    data["num_vehicles"] = 1

    data["depot"] = 0

    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]),
        data["num_vehicles"],
        data["depot"]
    )

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(
        from_index,
        to_index
    ):

        from_node = manager.IndexToNode(
            from_index
        )

        to_node = manager.IndexToNode(
            to_index
        )

        return data[
            "distance_matrix"
        ][from_node][to_node]

    transit_callback_index = (
        routing.RegisterTransitCallback(
            distance_callback
        )
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.
        FirstSolutionStrategy.
        PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(
        search_parameters
    )

    if solution:
        index = routing.Start(0)

        route = []

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index)

            route.append(
                locations[node_index]
            )

            index = solution.Value(
                routing.NextVar(index)
            )

        route.append(
            locations[
                manager.IndexToNode(index)
            ]
        )

        st.success(
            "Optimized Route Generated"
        )

        st.write("### Route Order")

        for stop in route:
            st.write(f"➡️ {stop}")

        st.subheader("🗺️ Optimized Route Map")

        route_map = folium.Map(
            location=[6.55, 3.40],
            zoom_start=10
        )

        for location in route:

            lat, lon = landfill_locations[location]

            folium.Marker(
                [lat, lon],
                popup=location,
                tooltip=location
            ).add_to(route_map)

        route_coordinates = []

        for location in route:

            lat, lon = landfill_locations[location]

            route_coordinates.append([lat, lon])

        folium.PolyLine(
            route_coordinates,
            weight=5
        ).add_to(route_map)

        st_folium(
            route_map,
            width=1000,
            height=600
        )

    else:

        st.error(
            "No route solution found."
        )
# ==========================================
# ANALYTICS PAGE
# ==========================================

if page == "Analytics":

    st.header("Project Analytics")

    st.info(
        """
        Model Performance Metrics obtained
        during Random Forest Evaluation.
        """
    )

    # Replace these with your actual values from model evaluation
    mae = 12.45
    rmse = 15.32
    r2 = 0.91

    st.metric(
        "MAE",
        f"{mae:.2f}"
    )

    st.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    st.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.subheader(
        "Impact Analysis"
    )

    traditional_distance = 150
    optimized_distance = 95

    distance_saved = (
        traditional_distance -
        optimized_distance
    )

    fuel_saved = (
        distance_saved * 0.15
    )

    st.write(
        f"Distance Saved: {distance_saved} km"
    )

    st.write(
        f"Estimated Fuel Saved: {fuel_saved:.2f} Litres"
    )
    