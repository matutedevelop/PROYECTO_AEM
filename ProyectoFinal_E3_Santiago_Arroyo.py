import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/ProyectoFinal_E3_Santiago_Arroyo.slides.json",
    css_file="custom.css",
)

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import altair as alt
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from wigglystuff import PairPlot
    from sklearn.model_selection import train_test_split
    from hastyplot import qplot

    SEED = 67


@app.cell
def _():
    # Paleta aproximada tomada de la imagen
    PALETTE = [
        "#2F6FBF",  # azul
        "#D95F2B",  # naranja quemado
        "#E0A00F",  # ocre / amarillo
        "#1F9D6B",  # verde
        "#EF233C",  # rosa
    ]

    BG = "#F9F8F0"
    PANEL = "#F9F8F0"  # fondo de cada panel
    TEXT = "#F9F8F0"  # fondo de cada panel
    BLK = "#2B2B2B"
    FONT = "JetBrains Mono, Fira Code, Menlo, Consolas, monospace"
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "mine", [PALETTE[0], BG, PALETTE[-1]]
    )

    sns.set_theme(style="white", palette=PALETTE, font="IBM Plex Mono")

    def beige_theme():
        return {
            "config": {
                "background": BG,
                "view": {
                    "fill": PANEL,
                    "stroke": None,  # sin borde en los paneles
                    "continuousWidth": 180,
                    "continuousHeight": 180,
                },
                "range": {
                    "category": PALETTE,
                    "ordinal": PALETTE,
                },
                "axis": {
                    "domain": False,  # sin línea de eje
                    "grid": True,
                    "gridColor": BLK,
                    # "gridDash": [1, 0, 100, 0],
                    "gridOpacity": 0.1,
                    "gridDashOffset": 200,
                    "ticks": False,
                    "labelColor": TEXT,
                    "titleColor": TEXT,
                    "labelFont": FONT,
                    "titleFont": FONT,
                    "labelFontSize": 11,
                    "titleFontSize": 12,
                    "titleFontWeight": "normal",
                },
                "title": {
                    "font": FONT,
                    "color": TEXT,
                    "fontSize": 14,
                    "fontWeight": "normal",
                    "anchor": "start",
                },
                "legend": {
                    "labelFont": FONT,
                    "titleFont": FONT,
                    "labelColor": TEXT,
                    "titleColor": TEXT,
                    "symbolType": "circle",
                },
                "mark": {"opacity": 0.6},
                "point": {"filled": True, "size": 6},
                "bar": {"opacity": 0.9},
            }
        }

    alt.themes.register("beige", beige_theme)
    alt.data_transformers.enable("vegafusion")
    alt.themes.enable("beige")
    return BLK, PALETTE, cmap


@app.cell
def _():
    df = pd.read_csv("data/hotel_bookings.csv")
    df.head()
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Estado de los datos
    """)
    return


@app.cell
def _(df):
    (df.shape, df.isna().sum())
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df.select_dtypes(exclude=["int64", "float64"]).nunique()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Contexto de los datos
    El dataset contiene informacion de reservaciones realizadas en diferentes _city hotels_ y _resort hotels_ de diferentes paises. El dataset recopila informacion que data desde 2015 a 2017. Por lo que desde inicio se advierte que los resultados pueden no extrapolarse al dia de hoy, desde que se presume que la pandemia pudo haber cambiado radicalmente la dinamica del sector hotelero.

    ## Resumen del estado de los datos

    Tenemos 32 variables con 119,390 observaciones, solamente 3 variables contienen datos nulos, una casi en su totalidad. Ademas de estas 3 variables con nulos 2 son numericas. Puesto que si se desea tratarlas o imputarlas de alguna forma pues entonces se debe de cuidar el leakage.

    Tambien en primera instancia los dtypes parecen ser adecuados y que no se requiere hacer un tratamiento adicional. Lo unico que puede protestarse es que variables booleanas usan `int64`, lo cual tecnicamente desperdicia espacio.

    ## Pregunta a resolver

    El dataset tiene una variable `is_canceled` que indica que dicha reservacion fue cancelada, la pregunta que guia nuestro estudio es si es que existe informacion de las reservaciones de los hoteles que a priori nos ayude a discriminar entre reservaciones _normales_ y reservaciones que despues son canceladas.

    De confirmarse la existencia y estudiarse dichas relaciones puede llegar a desvelar informacion de negocio valiosa para diseñar estrategias y toma de decisiones para los hoteles de contextos similares. Un ejemplo de estas _estrategias_ son del estilo de _"si a priori se que hay una alta probabilidad de que una reservacion sea cancelada, entonces no le asignare a dicha reservacion mis mejores suites"_. Otras estrategias interesantes aunque mas _greedys_ serian cosas como un ajuste en mis reportes, estimaciones de ingreso, demanda e insumos necesarios bajo la hipotesis de que  no todas las reservaciones van a ser efectivas.

    ### Informacion util

    Aunque el estudio busca ser exahustivo se buscara priorizar el uso de informacion que se sabe a priori de que la reservacion tenga lugar. Especialmente en la parte del modelado. Teniendo esto en cuenta a continuacion nombramos algunas variables que no cumplen esta caracteristica de _saberse a priori_

    - `reservation_status`
    - `reservation_status_date`
    """)
    return


@app.cell
def _(TARGET, df):
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=SEED
    )
    df_train.shape, df_test.shape
    return (df_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # EDA
    """)
    return


@app.cell
def _(df):
    NUM_COLS = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    NUM_COLS.remove("is_canceled")
    NUM_COLS.remove("is_repeated_guest")
    NUM_COLS.remove("company")
    NUM_COLS.remove("agent")
    CAT_COLS = df.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
    CAT_COLS.append("is_repeated_guest")
    CAT_COLS.append("company")
    CAT_COLS.append("agent")
    TARGET = ["is_canceled"]
    return CAT_COLS, NUM_COLS, TARGET


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## EDA variables numericas
    """)
    return


@app.cell
def _(NUM_COLS, df_train):
    df_train[NUM_COLS].nunique()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    hay algunas variables que se debe de poner un asterisco porque son categoricas, nominales u ordinales. Estas son `is_repeated_guess`, `arrive_date_year`, entre otras que tienen una granularidad muy baja
    """)
    return


@app.cell
def _(NUM_COLS, df):
    df[NUM_COLS].describe()
    return


@app.cell(expand_output=True)
def _(NUM_COLS, TARGET, df_train):
    pair_plot = PairPlot(
        df_train[NUM_COLS], columns=NUM_COLS, color=df_train[TARGET]
    )
    plot = mo.ui.anywidget(pair_plot)
    return pair_plot, plot


@app.cell
def _(TARGET, df_train, pair_plot, plot):
    if plot.value["selection"]:
        idx = pair_plot.get_indices()
        filtered = df_train.iloc[idx][TARGET]
        summary = {
            "is_canceled_prop": filtered.values.mean(),
            "len": filtered.shape[0],
        }
    else:
        summary = {
            "is_canceled_prop": df_train[TARGET].values.mean(),
            "len": df_train.shape[0],
        }
    chart = sns.barplot(
        x=[1, 0],
        y=[summary["is_canceled_prop"], 1 - summary["is_canceled_prop"]],
        hue=[1, 0],
    )
    plt.title("proporcion de is_canceled")
    mo.vstack([plot, chart])
    return


@app.cell
def _():
    return


@app.cell
def _(NUM_COLS, PALETTE, TARGET, df_train):
    def _make_boxplot(col):
        # _df = df_train[col]
        _chart = (
            alt.Chart(df_train)
            .mark_boxplot(color=PALETTE[0])
            .encode(
                y=f"{col}:Q",
                color=f"{TARGET[0]}:N",
                xOffset=alt.XOffset(f"{TARGET[0]}:N"),
            )
            .properties(width=600)
        )
        return _chart

    _charts = list(map(_make_boxplot, NUM_COLS))
    alt.vconcat(*_charts)
    return


@app.cell
def _(NUM_COLS, cmap, df_train):
    corr = df_train[NUM_COLS].corr()
    sns.heatmap(corr, cmap=cmap)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    cosas por hacer

    ADR -> cost
    (days_weekend,days_week) -> total_days

    analisis considerando el tiempo
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## EDA variables categoricas
    """)
    return


@app.cell
def _(PALETTE, TARGET, df_train):
    _df = df_train[TARGET].value_counts(normalize=True).reset_index()

    alt.Chart(_df).mark_bar(color=PALETTE[-4]).encode(
        x=f"{TARGET[0]}:N", y="proportion:Q"
    ).properties(width=600, height=400)
    return


@app.cell
def _(BLK, CAT_COLS, PALETTE, TARGET, df_train):
    target_prop = df_train[TARGET].values.mean()

    negative_desv_predicate = eval(
        f"alt.datum.{TARGET[0]}_proportion_desviation < 0"
    )
    is_negative_col = (
        alt.when(negative_desv_predicate)
        .then(alt.value(PALETTE[-1]))
        .otherwise(alt.value(PALETTE[3]))
    )

    def proportion_desviation_chart(col):
        new_prop = df_train.groupby(col)[TARGET].mean().reset_index()
        new_prop[TARGET] -= target_prop
        new_prop = new_prop.rename(
            columns={TARGET[0]: f"{TARGET[0]}_proportion_desviation"}
        )
        new_prop["desv_abs"] = new_prop[
            f"{TARGET[0]}_proportion_desviation"
        ].abs()
        _chart = (
            alt.Chart(new_prop)
            .mark_bar(color=PALETTE[-2])
            .encode(
                x=f"{col}:N",
                y=f"{TARGET[0]}_proportion_desviation:Q",
                color=is_negative_col,
            )
        ).properties(width=600, height=400)
        _line = (
            alt.Chart(pd.DataFrame({"y": [-0.1, 0, 0.1]}))
            .mark_rule(strokeWidth=2, strokeDash=[5, 3], color=BLK, opacity=1)
            .encode(y="y:Q")
        )
        return _chart + _line

    def proportion_chart(col):
        _df = df_train[col].value_counts(normalize=True).reset_index()

        _chart = (
            alt.Chart(_df)
            .mark_bar(color=PALETTE[-4])
            .encode(x=f"{col}:N", y="proportion:Q")
            .properties(width=600, height=400)
        )
        _line = (
            alt.Chart(pd.DataFrame({"y": [1 / len(_df)]}))
            .mark_rule(strokeWidth=2, strokeDash=[10, 3], color=BLK, opacity=1)
            .encode(y="y:Q")
        )
        return _chart + _line

    _charts = list(
        map(
            lambda x: proportion_desviation_chart(x) | proportion_chart(x),
            CAT_COLS,
        )
    )
    alt.vconcat(*_charts)
    return


@app.cell
def _(cmap, df_train):
    def crosstab_viz(col1, col2):
        _df = pd.crosstab(df_train[col1], df_train[col2])
        _df /= _df.sum()
        sns.heatmap(_df, cmap=cmap)
        plt.title(f"distribucion de apariciones\n {col1} vs {col2}")
        plt.show()

    return (crosstab_viz,)


@app.cell
def _(crosstab_viz):
    crosstab_viz("agent","company")
    return


@app.cell
def _(crosstab_viz):
    crosstab_viz("reserved_room_type","assigned_room_type")
    return


@app.cell
def _(crosstab_viz):
    crosstab_viz("distribution_channel","market_segment")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusiones del EDA

    ### Variables numericas

    #### Correlacion
    Pareciera no haber correlaciones altas entra las variables numericas de nuestro dataset solamente 2 pares de variables presentan una relacion fuerte el $\text{numero de semana}- \text{año}$ y el otro par $\text{dias entre semana}-\text{dias fin de semana}$. La segunda relacion es sencilla de entender, la primera nos habla de
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
