import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import (
    List,
    Dict,
    Literal
)

def series_plot(
    df:pd.DataFrame, 
    label_names:Dict[str, str]=None,
    colors:List[str]=px.colors.qualitative.Dark24, 
    col_to_yaxis:Dict[str, Literal["y", "y2"]]=None,
    show_rangeobjects:bool=True,
    **layout_kwargs,
):
    # make traces
    traces = [] # trace list
    for i, col in enumerate(df.columns):
        name = label_names[col] if label_names is not None else col # change label name
        trace = go.Scatter(
            x = df.index,
            y = df[col],
            name = name,
            marker_color = colors[i % len(colors)],
            yaxis = col_to_yaxis[col] if col_to_yaxis else "y"
        )
        traces.append(trace)
    # make layout
    layout = go.Layout(layout_kwargs)
    # make figure
    fig = go.Figure(data=traces, layout=layout)
    # set range objects
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=[
                {
                    "count":i, "label":f"last {i} year", "step":"year",
                } for i in [1, 3, 5, 10]
            ],
            bgcolor="#2F301F"
        )
    )
    return fig