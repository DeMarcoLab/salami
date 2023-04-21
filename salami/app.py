import streamlit as st

from salami import analysis as sa
import os
import pandas as pd

import plotly.express as px
st.set_page_config(layout="wide")

def main():

    st.title("Salami Analysis")

    DEFAULT_PATH = path = "/home/patrick/github/data/salami/analysis/2023-04-14-07-05-07PM/data"
    path = st.text_input("Dataset Path", value=DEFAULT_PATH)

    df = pd.read_csv(os.path.join(path, "parameters_metrics.csv"))
    
    df["current"] = df["current"].astype("float") * 1e9
    df["dwell_time"] = df["dwell_time"].astype("float") * 1e6
    df["hfw"] = df["hfw"].astype("float") * 1e6

    # drop path and metric

    cols = st.columns(2)
    params = {
            # "voltage": {"min": 300, "max": 300, "step": 1},
              "current": {"min": 0, "max": 100, "step": 1},
              "dwell_time": {"min": 0, "max": 100, "step": 1},
              "pixelsize": {"min": 0, "max": 100, "step": 1},
              }

    for param in params:
        df[param] = df[param].astype("float")

        vals = float(df[param].min()), float(df[param].max())
        range = cols[0].slider(f"{param} Range", min_value=vals[0], max_value=vals[1], value=vals)

        params[param]["min"] = range[0]
        params[param]["max"] = range[1]

    # filter df   
    for param in params:
        df = df[(df[param] >= params[param]["min"]) & (df[param] <= params[param]["max"])]

    cols[1].write(f"{len(df)} Filtered Data Points")
    
    show_df  = df.drop(columns=["path", "metric", "idx", "basename", "resolution_x", "resolution_y"])
    cols[1].dataframe(show_df, use_container_width=True)

    # convert current, dwell time, hfw  to category
    df["current"] = df["current"].astype("category")
    df["dwell_time"] = df["dwell_time"].astype("category")
    df["hfw"] = df["hfw"].astype("category")


    st.header('Plots')
    cols = st.columns(2)

    # plot 3d scatter plot with current, pixelsize and int_05 color by dwell time
    fig = px.scatter_3d(df, x="current", 
                        y="pixelsize", 
                        z="int_05", 
                        color="dwell_time", 
                        opacity=0.5)
    # set title
    fig.update_layout(title="FRC 0.5")
    cols[0].plotly_chart(fig)


    # plot 3d scatter plot with current, pixelsize and int_0143 color by dwell time
    fig = px.scatter_3d(df, x="current", 
                        y="pixelsize", 
                        z="int_0143", 
                        color="dwell_time", 
                        opacity=0.5)
    fig.update_layout(title="FRC 0.143")

    cols[1].plotly_chart(fig)


    fig = px.line(df, x="pixelsize", y="int_05", 
              color="dwell_time", 
              line_group="current",
              line_dash="current", 
              hover_name="basename")

    # set title
    fig.update_layout(title="FRC 0.5")

    cols[0].plotly_chart(fig)

    fig = px.line(df, x="pixelsize", y="int_0143", 
                color="dwell_time", 
                line_group="current",
                line_dash="current", 
                hover_name="basename")
    # set title
    fig.update_layout(title="FRC 0.143")
    cols[1].plotly_chart(fig)


if __name__ == "__main__":
    main()