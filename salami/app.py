import streamlit as st

from salami import analysis as sa
import os
import pandas as pd

import plotly.express as px
st.set_page_config(layout="wide")

def main():

    st.title("Salami Analysis")

    px = st.selectbox("PX", [1536, 2*1536, 4*1536, 8*1536])
    py = st.selectbox("PY", [1024, 2048, 4096, 8192, ])
    dwell_time = st.selectbox("Dwell Time", [0.5, 1.0, 3.0, 5.0, 8.0])

    time = dwell_time * px * py / 1e6
    st.write(f"Time: {time} s")


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
        rn = cols[0].slider(f"{param} Range", min_value=vals[0], max_value=vals[1], value=vals)

        params[param]["min"] = rn[0]
        params[param]["max"] = rn[1]

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
              facet_col="current",
            #   line_group="current",
            #   line_dash="current", 
              hover_name="basename")

    # set title
    fig.update_layout(title="FRC 0.5")

    cols[0].plotly_chart(fig)

    fig = px.line(df, x="pixelsize", y="int_0143", 
                color="dwell_time", 
                facet_col="current",
                # line_group="current",
                # line_dash="current", 
                hover_name="basename")
    # set title
    fig.update_layout(title="FRC 0.143")
    cols[1].plotly_chart(fig)


    col = st.selectbox("Select Metric", [param for param in params.keys()])
    data = df[[col, "metric"]]

    def _to_floats(data):
        data = data.split("\n")
        data = "".join(data).replace("[", "").replace("]", "").split(" ")

        # drop empty strings
        return [float(x) for x in data if x != ""]
    



    # plot metrics for each unique pixelsize
    import numpy as np

    vals = df[col].unique()
    for val in vals:
        data = df[df[col] == val]["metric"]
        metrics = [_to_floats(d) for d in data]
        mean_metric = np.mean(metrics, axis=0)

        x = np.array(range(len(mean_metric))) / val
        fig = px.line(x=x, y=mean_metric)
        fig.update_layout(title=f"{col} {val}")
        st.plotly_chart(fig)


if __name__ == "__main__":
    main()