"""Presentation helpers for the benchmark app: the results chart, the plain-language
captions, readable column names, friendly error messages and export file names.

Nothing here computes a result. Every function takes numbers the app has already
produced and only decides how they are drawn, labelled or named.
"""

from __future__ import annotations

import re
from datetime import datetime

import streamlit as st

# One colour per algorithm, the same on every module and tab. The shape and feature
# methods are blues and greys, the dynamics-aware reference is vermilion and fSFA is
# amber; all are from the Okabe-Ito colour-blind-safe set or tints of its blue.
ALGORITHM_COLOURS = {
    "Rocket": "#0072B2",
    "MiniRocket": "#56B4E9",
    "catch22": "#5C6F82",
    "DTW": "#A9B7C6",
    "fSFA": "#E69F00",
    "Dynamics (NN)": "#D55E00",
}
ALGORITHM_ORDER = list(ALGORITHM_COLOURS)

LEGEND = ("Blue and grey bars are the shape and feature methods (Rocket, MiniRocket, catch22, "
          "DTW). Vermilion is the dynamics-aware reference, amber is fSFA. The dashed line is "
          "the chance level and the whiskers are one standard error.")

# Internal column names -> what the tables show. The accuracy column of the per-dataset
# matrix keeps its name, so "Accuracy" is renamed only where it is a mean.
PLAIN_COLUMNS = {
    "Std. error": "Uncertainty (± std. error)",
    "Chance": "Chance level",
    "Vs chance": "Compared with chance",
    "Reference": "Dynamics reference",
    "Best shape/feature": "Best shape/feature method",
    "Gap SE": "Gap uncertainty (± std. error)",
    "Datasets": "Datasets used",
}
MEAN_ACCURACY = "Mean accuracy"

EXPLANATIONS = (
    "**Accuracy** is the share of test series given the correct label, averaged over the "
    "datasets that were run.",
    "**Chance level** is the accuracy of always guessing the more common label in the test "
    "set. A method has to beat it to have learned anything.",
    "**Not clearly above chance** means the mean accuracy minus two standard errors is at or "
    "below the chance level, so the result could be luck. It needs at least two datasets.",
    "**Gap** is the dynamics reference's accuracy minus the best of Rocket, MiniRocket, "
    "catch22 and DTW on the same dataset, averaged over datasets. Positive means looking at "
    "the dynamics helps. fSFA is shown beside it, not included.",
)


def plain(df, mean_accuracy=False):
    """``df`` with its internal column names replaced by plain-English ones."""
    names = dict(PLAIN_COLUMNS)
    if mean_accuracy:
        names["Accuracy"] = MEAN_ACCURACY
    return df.rename(columns=names)


def accuracy_chart(summary):
    """Grouped bars of mean accuracy per algorithm, one panel per condition.

    ``summary`` has Condition, Algorithm, Accuracy, Std. error and Chance columns. The
    y axis is fixed at 0 to 1, the bars take their colour from ALGORITHM_COLOURS, the
    whiskers are one standard error and the dashed line is the chance level.
    """
    import altair as alt

    data = summary[["Condition", "Algorithm", "Accuracy", "Std. error", "Chance"]].copy()
    data["Low"] = (data["Accuracy"] - data["Std. error"].fillna(0)).clip(lower=0)
    data["High"] = (data["Accuracy"] + data["Std. error"].fillna(0)).clip(upper=1)
    x = alt.X("Algorithm:N", sort=ALGORITHM_ORDER, title=None, axis=alt.Axis(labelAngle=-40))
    y_scale = alt.Scale(domain=[0, 1])
    colour = alt.Color("Algorithm:N", legend=None,
                       scale=alt.Scale(domain=ALGORITHM_ORDER,
                                       range=[ALGORITHM_COLOURS[a] for a in ALGORITHM_ORDER]))
    bars = alt.Chart().mark_bar().encode(
        x=x, y=alt.Y("Accuracy:Q", scale=y_scale, title="Mean accuracy"), color=colour,
        tooltip=["Algorithm", alt.Tooltip("Accuracy:Q", format=".1%"),
                 alt.Tooltip("Std. error:Q", format=".1%"), alt.Tooltip("Chance:Q", format=".1%")])
    whiskers = alt.Chart().mark_errorbar(color="black").encode(
        x=x, y=alt.Y("Low:Q", scale=y_scale, title="Mean accuracy"), y2="High:Q")
    chance = alt.Chart().mark_rule(color="black", strokeDash=[6, 4]).encode(
        y=alt.Y("Chance:Q", scale=y_scale))
    return (alt.layer(bars, whiskers, chance, data=data)
            .properties(width=190, height=230)
            .facet(facet=alt.Facet("Condition:N", sort=list(dict.fromkeys(data["Condition"])), title=None),
                    columns=3))


def show_chart(summary):
    """The chart and its one-line colour legend."""
    st.altair_chart(accuracy_chart(summary))
    st.caption(LEGEND)


def show_explanations():
    """What the figures mean, tucked away so the page stays short."""
    with st.expander("How to read these results"):
        for line in EXPLANATIONS:
            st.markdown(f"- {line}")


def show_problem(message, details=None):
    """A short plain message, with the technical text one click away."""
    st.error(message)
    if details:
        with st.expander("Technical details"):
            st.code(str(details))


def export_name(label, what, extension="csv"):
    """``<label>_<what>_<YYYYmmdd-HHMMSS>.<extension>`` with the label made file-safe."""
    safe = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_").lower() or "results"
    return f"{safe}_{what}_{datetime.now():%Y%m%d-%H%M%S}.{extension}"


def csv_download(df, label, what, key):
    """A download button for ``df`` that does not rerun the page (the results stay up)."""
    st.download_button(f"Download {what.replace('_', ' ')} as CSV", df.to_csv(index=False).encode(),
                       file_name=export_name(label, what), mime="text/csv", key=key,
                       on_click="ignore")
