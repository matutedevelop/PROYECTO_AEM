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
    from scipy import stats as stats
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from matplotlib import pyplot as plt
    from wigglystuff import PairPlot
    from sklearn.model_selection import train_test_split
    from hastyplot import qplot

    SEED = 67


@app.cell
def _():
    PALETTE = [
        "#2F6FBF",  # azul
        "#D95F2B",  # naranja quemado
        "#E0A00F",  # ocre / amarillo
        "#1F9D6B",  # verde
        "#EF233C",  # rosa
    ]

    BG = "#F9F8F0"
    PANEL = "#F9F8F0"
    TEXT = "#F9F8F0"
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Muestreo
    """)
    return


@app.cell
def _(df):
    n_obs = df.shape[0]
    n_obs
    _p = 0.5
    _z = 1.96
    _e = 0.05

    n_sample = np.round(
        (n_obs * _z**2 * _p * (1 - _p))
        / (_e**2 * (n_obs - 1) + _z**2 * _p * (1 - _p))
    )

    sample_proportion = n_sample / n_obs
    n_sample, sample_proportion
    return (sample_proportion,)


@app.cell
def _(TARGET, df, sample_proportion):
    df_sample, df_test = train_test_split(
        df,
        train_size=sample_proportion,
        stratify=df[TARGET],
        random_state=SEED,
    )
    df_sample.shape, df_test.shape
    return (df_sample,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Se realizo el muestreo del dataset completo con el cual realizaremos nuestros estudios y pruebas estadisticas. Dicha muestra es de 383 observaciones, durante el muestreo se uso estratificacion para garantizar que las proporciones de la variable `is_canceled` se mantuvieran constantes.
    """)
    return


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
def _(NUM_COLS, df_sample):
    df_sample[NUM_COLS].nunique()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    hay algunas variables que se debe de poner un asterisco porque son categoricas, nominales u ordinales. Estas son `is_repeated_guess`, `arrive_date_year`, entre otras que tienen una granularidad muy baja y es algo a tomar en cuenta
    """)
    return


@app.cell
def _(NUM_COLS, df_sample):
    df_sample[NUM_COLS].describe()
    return


@app.cell
def _(NUM_COLS, df_sample):
    df_sample[NUM_COLS].skew()
    return


@app.cell
def _(NUM_COLS, df_sample):
    df_sample[NUM_COLS].kurt()
    return


@app.cell
def _(NUM_COLS, TARGET, df_sample):
    pair_plot = PairPlot(
        df_sample[NUM_COLS], columns=NUM_COLS, color=df_sample[TARGET]
    )
    plot = mo.ui.anywidget(pair_plot)
    return pair_plot, plot


@app.cell(expand_output=True)
def _(TARGET, df_sample, pair_plot, plot):
    if plot.value["selection"]:
        idx = pair_plot.get_indices()
        filtered = df_sample.iloc[idx][TARGET]
        summary = {
            "is_canceled_prop": filtered.values.mean(),
            "len": filtered.shape[0],
        }
    else:
        summary = {
            "is_canceled_prop": df_sample[TARGET].values.mean(),
            "len": df_sample.shape[0],
        }
    _chart = sns.barplot(
        x=[1, 0],
        y=[summary["is_canceled_prop"], 1 - summary["is_canceled_prop"]],
        hue=[1, 0],
    )
    plt.title("proporcion de is_canceled")
    mo.vstack([plot, _chart])
    return


@app.cell
def _(NUM_COLS, PALETTE, TARGET, df_sample):
    def _make_boxplot(col):
        # _df = df_sample[col]
        _chart = (
            alt.Chart(df_sample)
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
def _(NUM_COLS, cmap, df_sample):
    _corr = df_sample[NUM_COLS].corr()
    sns.heatmap(_corr, cmap=cmap)
    plt.title("matriz de correlacion de pearson")
    return


@app.cell
def _(NUM_COLS, cmap, df_sample):
    _corr = df_sample[NUM_COLS].corr(method="kendall")
    sns.heatmap(_corr, cmap=cmap)
    plt.title("matriz de correlacion de kendall")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Posibles relaciones
    el pairplot y los boxplots desvelan varias relaciones utiles entre `is_canceled` y diversas variables numericas. Por nombrar algunas de estas relaciones:
    - `lead_time` a  mayor tiempo mas chances de que cancele
    -   A mayor numero de dias que una persona se queda `stays_in_week_nights,stays_in_weekend_nights`, mayor es la proporcion de cancelados
    -   Si la persona reserva con niños o bebes `children, babies` es muy poco probable que cancele
    -   Si la persona ha cancelado antes es probable que vuelva a cancelar `previous_cancelations`
    -   Si la persona hace `booking_changes` es poco probable que cancele
    -   Si la persona ha estado mucho tiempo en `waiting_list` tiende a cancelar
    -   A mayor `adr` mayor proporcion de cancelados
    -   si la persona requiere espacio de estacionamiento rara vez cancela
    -   si la persona hace solicitudes especiales rara vez cancela



    #### Correlacion
    Pareciera no haber correlaciones altas entra las variables numericas de nuestro dataset solamente 2 pares de variables presentan una relacion fuerte el $\text{numero de semana}- \text{año}$ y el otro par $\text{dias entre semana}-\text{dias fin de semana}$. La segunda relacion es sencilla de entender, la primera nos habla de

    ### Posibles problemas

    observando los scatterplots, boxplots, cuartiles, sesgos y kurtosis. Podemos concluir que hay una cantidad considerable de variables (en especial las dicretas) muy sesgadas y con mucha _masa_ en un valor especifico. por nombrar algunas de estas variables `adults, children, babies, previous_cancelation, days_in_waiting_list,` etc. Se sabe a priori tan solo con los histogramas que estas variables presentan, no son procesos normalmente distribuidos, y queda pendiente decidir como se trataran particularmente. Ya sea con _transformaciones de potencia_ (box-cox), o _discretizandolas_ de la forma de $\text{has canceled} \to \{0,1\}$, qcuts, reducirlas, kmeans, etc.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pruebas estadisticas


    #### Normalidad Shapiro-Wilk
    A continuacion realizaremos pruebas estadisticas de normalidad sobre las variables numericas. desde que tenemos un $n$ lo suficientemente grande utilizamos Shapiro-Wilk como pruba estadistica. La prueba de Shapiro-Wilk tiene como hipotesis nula que los datos provienen de una distribucion normal


    $$
    H_0: X \sim \cal{N}(\mu, \sigma)
    $$
    $$
    H_a: X \nsim \cal{N}(\mu, \sigma)
    $$

    realizaremos las pruebas con un nivel de significancia del 5% $\alpha = 0.05$
    """)
    return


@app.cell
def _(NUM_COLS, df_sample):
    shapiro_tests = {
        col: stats.shapiro(df_sample[col]).pvalue for col in NUM_COLS
    }

    _df = pd.Series(shapiro_tests).reset_index(name="p_value")
    _df["null_hyphotesis"] = (_df["p_value"] < 0.05) ^ 1
    _df
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    no existe evidencia para decir que con el 5% de significancia las variables numericas de nuestro dataset estan normalmente distribuidas. antes ya habiamos advertido esta posibilidad, a continuacion se presentan los histogramas de forma detallada para decidir como tratar cada variable
    """)
    return


@app.cell
def _(NUM_COLS, PALETTE, df_sample):
    def make_histogram(col):
        _df = df_sample[col].reset_index()
        _df[f"log1p_{col}"] = np.log1p(_df[col])
        _histogram = (
            alt.Chart(_df)
            .mark_bar(color=PALETTE[3])
            .encode(x=f"{col}:Q", y="count()")
            .properties(title=f"histograma de {col}", width=500, height=400)
        )
        _log_histogram = (
            alt.Chart(_df)
            .mark_bar(color=PALETTE[4])
            .encode(x=f"log1p_{col}:Q", y="count()")
            .properties(
                title=f"histograma de log1p_{col}", width=500, height=400
            )
        )
        return _histogram | _log_histogram

    _charts = list(map(make_histogram, NUM_COLS))
    alt.vconcat(*_charts)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Observando los histogramas anteriores existen algunas variables que podrian verse beneficiadas de transformaciones de potencia. Dichas variables son
    - `stays_in_week_nights`
    - `adr`

    por otro lado por su alta concentracion en valores en `0` se considera adecuado hacer binarias las siguientes variables.  No obstante es pertinente revisarlo y probarlo caso por caso
    - `total_of_special_requests`
    - `required_car_parking_space`
    - `days_in_waiting_list`
    - `booking_changes`
    - previous_cancelation
    - children
    - babies

    Algunas variables de las cuales se consideran son candidatas para usarse en algun proceso de feature engineerng son las siguientes
    - `Adults,Children,babies` y crear una variable tipo total guesses y complementarla con dummies indicadores de niños y bebes, o bien reducir estas 3 variables con PCA
    - `Adr` esta variable tiene encodeada el monto que un cliente pago por lo que puede ser util aplicarle algun tipo de feature engineering para analizar los datos desde otra perspectiva e.g. precio vs tipo de suite


    Ademas las siguientes variables se considera candidatas para desecharse debido a que su informacion variabilidad o interpretacion no resulta interesante para el estudio
    - `arrival_date_year`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Pruebas de correlacion de spearman y pearson

    La correlacion de spearman es un estadistico que nos permite saber si hay evidencia estadistica de una _correlacion monotonica_ entre dos variables _i.e._ cuando una aumenta la otra aumenta o disminuye. Pero sin restringirse a buscar solamente relaciones lineales. por ejemplo si tenemos la relacion $X \sim \cal N(\mu,\sigma)$ y $Y \sim \ln(X) + \epsilon$ entonces es mejor capturada por la correlacion de spearman que por la correlacion de pearson. El coeficiente de correlacion de spearman de 2 variables $X,Y$ se anota $r_{X Y }$ o con $\rho_{X Y}$ y al igual que la correlacion de pearson esta acotada entre -1,1 indicando el signo si la relacion es monotonica postiva o inversa

    en stats tambien tenemos un p-value asociado que nos dice de grosa forma la probabilidad de que $r_{X Y} \geq r_{\text{obs}}$ dado que $X,Y$ no esten correlacionados

    por lo que la hipotesis nula seria que los coeficientes $rho$ sean 0, e igualmente realizaremos las pruebas con $\alpha=0.05$

    $$
    H_0: \rho = 0
    $$
    """)
    return


@app.cell
def _(NUM_COLS, df_sample, spearman_coefs):
    pearson_coefs = []
    _alredy_passed = set()

    for _col_1 in NUM_COLS:
        for _col_2 in NUM_COLS:
            if _col_1 == _col_2 or _col_2 in _alredy_passed:
                continue
            pearson_coefs.append(
                {
                    "columns": f"{_col_1}-{_col_2}",
                    "results": stats.pearsonr(
                        df_sample[_col_1], df_sample[_col_2]
                    ).statistic,
                    "p_value": stats.pearsonr(
                        df_sample[_col_1], df_sample[_col_2]
                    ).pvalue,
                }
            )
        _alredy_passed.add(_col_1)

    _df = pd.DataFrame(spearman_coefs)
    _df["null_hypothesis"] = (_df["p_value"] < 0.05) ^ 1
    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X(field="columns", type="nominal"),
            y=alt.Y(field="results", type="quantitative", aggregate="mean"),
            color="null_hypothesis:N",
            tooltip=[
                alt.Tooltip(field="columns"),
                alt.Tooltip(field="results", aggregate="mean", format=",.2f"),
                alt.Tooltip(field="p_value", aggregate="mean", format=",.4f"),
            ],
        )
        .properties(
            title="pearson_correlations",
            height=397,
            width=600,
        )
    )

    _chart
    mo.hstack([_df, _chart])
    return


@app.cell
def _(NUM_COLS, PALETTE, df_sample):
    spearman_coefs = []
    _alredy_passed = set()

    for _col_1 in NUM_COLS:
        for _col_2 in NUM_COLS:
            if _col_1 == _col_2 or _col_2 in _alredy_passed:
                continue
            spearman_coefs.append(
                {
                    "columns": f"{_col_1}-{_col_2}",
                    "results": stats.spearmanrho(
                        df_sample[_col_1], df_sample[_col_2]
                    ).statistic,
                    "p_value": stats.spearmanrho(
                        df_sample[_col_1], df_sample[_col_2]
                    ).pvalue,
                }
            )
        _alredy_passed.add(_col_1)

    _df = pd.DataFrame(spearman_coefs)
    _df["null_hypothesis"] = (_df["p_value"] < 0.05) ^ 1
    _chart = (
        alt.Chart(_df)
        .mark_bar(color=PALETTE[-1])
        .encode(
            x=alt.X(field="columns", type="nominal"),
            y=alt.Y(field="results", type="quantitative", aggregate="mean"),
            color="null_hypothesis:N",
            tooltip=[
                alt.Tooltip(field="columns"),
                alt.Tooltip(field="results", aggregate="mean", format=",.2f"),
                alt.Tooltip(field="p_value", aggregate="mean", format=",.4f"),
            ],
        )
        .properties(
            title="spearman_correlations",
            height=397,
            width=600,
        )
    )

    _chart
    mo.hstack([_df, _chart])
    return (spearman_coefs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### VIF analysis

    A continuacion evaluamos el VIF de cada una de nuestras features numericas para detectar posibles problemas de correlacion
    """)
    return


@app.cell
def _(BLK, NUM_COLS, PALETTE, df_sample):
    _df = df_sample[NUM_COLS]

    vif_data = pd.DataFrame()
    vif_data["feature"] = _df.columns

    vif_data["VIF"] = [
        variance_inflation_factor(_df.values, i)
        for i in range(len(_df.columns))
    ]

    _chart = alt.Chart(vif_data).mark_bar(color=PALETTE[3]).encode(
        x=alt.X(field="feature", type="nominal"),
        y=alt.Y(field="VIF", type="quantitative", aggregate="mean"),
        tooltip=[
            alt.Tooltip(field="feature"),
            alt.Tooltip(field="VIF", aggregate="mean", format=",.2f"),
        ],
    ).properties(
        title="VIF de las features",
        height=350,
        width=600,
    ) + alt.Chart(pd.DataFrame({"y": [5]})).mark_rule(
        strokeWidth=2, strokeDash=[10, 3], color=BLK, opacity=1
    ).encode(y="y:Q")
    _chart
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    El analisis del VIF nos arroja que no tenemos problemas graves de colinearidad y los test de correlacion estan en relaciones bajas con la excepcion de el numero de la semana con el año, desde que establecimos que el numero de año no seria tomado en cuenta en el modelado y la inferencia, no tenemos problemas. La unica relacion que podria valer la pena investigar es la de `lead_time` con `stays_in_week_days`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## EDA variables categoricas
    """)
    return


@app.cell
def _(PALETTE, TARGET, df_sample):
    _df = df_sample[TARGET].value_counts(normalize=True).reset_index()

    alt.Chart(_df).mark_bar(color=PALETTE[-4]).encode(
        x=f"{TARGET[0]}:N", y="proportion:Q"
    ).properties(width=600, height=400)
    return


@app.cell
def _(BLK, CAT_COLS, PALETTE, TARGET, df_sample):
    target_prop = df_sample[TARGET].values.mean()

    negative_desv_predicate = eval(
        f"alt.datum.{TARGET[0]}_proportion_desviation < 0"
    )
    is_negative_col = (
        alt.when(negative_desv_predicate)
        .then(alt.value(PALETTE[-1]))
        .otherwise(alt.value(PALETTE[3]))
    )

    def proportion_desviation_chart(col):
        new_prop = df_sample.groupby(col)[TARGET].mean().reset_index()
        new_prop[TARGET[0]] -= target_prop
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
        _df = df_sample[col].value_counts(normalize=True).reset_index()

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
def _(cmap, df, df_sample):
    def crosstab_viz(col1, col2, use_all_data=False):
        _df = pd.crosstab(df_sample[col1], df_sample[col2])
        if use_all_data:
            _df = pd.crosstab(df[col1], df[col2])
        try:
            _df /= _df.sum()
            sns.heatmap(_df, cmap=cmap)
            plt.title(f"distribucion de apariciones\n {col1} vs {col2}")
            plt.show()
        except Exception as e:
            print("no common entries")

    return (crosstab_viz,)


@app.cell
def _(crosstab_viz):
    crosstab_viz("agent", "company", use_all_data=False)
    return


@app.cell
def _(crosstab_viz):
    crosstab_viz("reserved_room_type", "assigned_room_type")
    return


@app.cell
def _(crosstab_viz):
    crosstab_viz("distribution_channel", "market_segment")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pruebas estadisticas

    realizaremos la prueba $\chi^2$ de contingencia para detectar _correlaciones_ entre variables categoricas. Los graficos anteriores pueden anticiparnos una relacion estadisticamente significativa con `is_canceled`. usamos significacnia de $\alpha = 0.05$


    $$
    H_0: X \perp Y
    $$
    """)
    return


@app.cell
def _(CAT_COLS, PALETTE, TARGET, df_sample):
    chi2_contingency_tests = []
    # _alredy_passed = set()

    for _col_1 in CAT_COLS:
        observed_table = pd.crosstab(df_sample[_col_1], df_sample[TARGET[0]])

        if observed_table.shape[0] == 0:
            continue

        _test = stats.chi2_contingency(observed_table)

        chi2_contingency_tests.append(
            {
                "columns": f"{_col_1}-{TARGET[0]}",
                "results": _test.statistic,
                "dof": _test.dof,
                "p_value": _test.pvalue,
            }
        )
    # _alredy_passed.add(_col_1)

    _df = pd.DataFrame(chi2_contingency_tests)
    _df["null_hypothesis"] = (_df["p_value"] < 0.05) ^ 1

    _chart = (
        alt.Chart(_df)
        .mark_bar(color=PALETTE[-1])
        .encode(
            x=alt.X(field="columns", type="nominal"),
            y=alt.Y(field="results", type="quantitative", aggregate="mean"),
            color="null_hypothesis:N",
            tooltip=[
                alt.Tooltip(field="columns"),
                alt.Tooltip(field="results", aggregate="mean", format=",.2f"),
                alt.Tooltip(field="p_value", aggregate="mean", format=",.4f"),
            ],
        )
        .properties(
            title="test chi2 de contingencia",
            height=397,
            width=600,
        )
    )

    _chart
    mo.hstack([_df, _chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    con un nivel de significancia del 5% hay evidencia para decir que las siguientes variables

    - `agent`
    - `assigned_room_type`
    - `country`
    - `customer_type`
    - `deposit_type`
    - `distribution_chanel`
    - `hotel`
    - `is_repeated_guess`


    `reservation_status` siempre
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Conclusiones del EDA

    ### Variables numericas
    """)
    return


@app.cell
def _():
    ### P
    return


if __name__ == "__main__":
    app.run()
