import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", css_file="custom.css")

with app.setup:
    import sys
    import itertools
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import altair as alt
    import matplotlib as mpl
    from scipy import stats as stats
    from scipy.stats import studentized_range
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import lilliefors
    from statsmodels.tools.tools import add_constant
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.oneway import anova_oneway
    from matplotlib import pyplot as plt
    from wigglystuff import PairPlot
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from hastyplot import qplot

    SEED = 42


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
    **Poblacion.** Todas las reservaciones registradas entre julio de 2015 y agosto de 2017 en un _city hotel_ (Lisboa) y un _resort hotel_ (Algarve) de Portugal, $N = 119{,}390$.

    **Tamaño de muestra.** Con la formula para poblaciones finitas ($z=1.96$, $e=0.05$, $p=0.5$) obtenemos $n = 383$, es decir, apenas el $0.32\%$ de la poblacion. Notese que al ser $N$ tan grande la correccion por poblacion finita es practicamente nula ($n_0 = 384.16$ para poblacion infinita).

    **Metodo de muestreo.** Se uso un muestreo aleatorio **estratificado por `is_canceled`** con `random_state = 42`. Se justifica porque `is_canceled` es nuestra variable dependiente: con una muestra tan pequeña, un muestreo aleatorio simple podria desbalancear por azar la proporcion de cancelaciones, lo cual sesgaria directamente las estimaciones del modelo logistico. Los estratos son $\{0, 1\}$ y conservan la proporcion poblacional (ver tabla siguiente).
    """)
    return


@app.cell
def _(NUM_COLS, TARGET, df, df_sample):
    # Comparacion poblacion vs muestra: proporcion del target y estadisticas de las numericas
    _prop = pd.DataFrame(
        {
            "poblacion": df[TARGET[0]].value_counts(normalize=True),
            "muestra": df_sample[TARGET[0]].value_counts(normalize=True),
        }
    )

    _rows = []
    for _col in NUM_COLS:
        _pop = df[_col].dropna()
        _smp = df_sample[_col].dropna()
        _ks = stats.ks_2samp(_pop, _smp)
        _rows.append(
            {
                "variable": _col,
                "media_pob": _pop.mean(),
                "media_mue": _smp.mean(),
                "mediana_pob": _pop.median(),
                "mediana_mue": _smp.median(),
                "std_pob": _pop.std(),
                "std_mue": _smp.std(),
                "KS_2samp": _ks.statistic,
                "p_value": _ks.pvalue,
            }
        )
    representatividad = pd.DataFrame(_rows).round(4)
    representatividad["misma_distribucion"] = representatividad["p_value"] >= 0.05

    mo.vstack(
        [
            mo.md("**Proporcion de `is_canceled` (estratos)**"),
            _prop.round(4),
            mo.md(
                "**Estadisticas poblacion vs muestra + prueba K-S de dos muestras** "
                "($H_0$: la muestra y la poblacion provienen de la misma distribucion)"
            ),
            representatividad,
        ]
    )
    return (representatividad,)


@app.cell(hide_code=True)
def _(representatividad):
    _rechazos = representatividad.loc[
        ~representatividad["misma_distribucion"], "variable"
    ].tolist()
    mo.md(
        rf"""
    **Interpretacion.** La proporcion de cancelaciones es identica en poblacion y muestra (por construccion del estrato). Para las variables numericas, las medias, medianas y desviaciones de la muestra son cercanas a las poblacionales, y la prueba K-S de dos muestras con $\alpha = 0.05$ rechaza $H_0$ en **{len(_rechazos)} de {len(representatividad)}** variables {"(" + ", ".join(f"`{c}`" for c in _rechazos) + ")" if _rechazos else ""}. Por lo tanto la muestra es representativa de la poblacion en cuanto a la distribucion de las variables de estudio.
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Datos faltantes
    """)
    return


@app.cell
def _(df, df_sample):
    _cols = ["children", "country", "agent", "company"]
    faltantes = pd.DataFrame(
        {
            "nulos_pob": df[_cols].isna().sum(),
            "pct_pob": (df[_cols].isna().mean() * 100).round(2),
            "nulos_mue": df_sample[_cols].isna().sum(),
            "pct_mue": (df_sample[_cols].isna().mean() * 100).round(2),
        }
    )
    faltantes
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Tratamiento de faltantes.** Ninguno de los nulos parece ser un error de captura, sino informacion con significado:

    - `company` (~94% nulos) y `agent` (~14% nulos): el nulo significa que la reservacion **no** se hizo a traves de una empresa / agencia. No se imputan: se recodifican como indicadoras binarias `has_company` y `has_agent`. Ademas los IDs tienen cientos de niveles, inmanejables con $n = 383$.
    - `country` (<1% nulos): se imputa con la categoria `"UNK"` (desconocido). Por la alta cardinalidad se agrupara despues (p. ej. Portugal vs. resto del mundo).
    - `children` (4 nulos en toda la poblacion): se imputa con 0, que es la moda y el valor mas plausible.

    Estas reglas no usan informacion del target ni de otras observaciones (salvo la moda de `children`), por lo que no introducen _leakage_.
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
    Pareciera no haber correlaciones altas entra las variables numericas de nuestro dataset solamente 2 pares de variables presentan una relacion fuerte el $\text{numero de semana}- \text{año}$ y el otro par $\text{dias entre semana}-\text{dias fin de semana}$. La segunda relacion es sencilla de entender: quien se queda mas noches entre semana tambien suele incluir un fin de semana. La primera no es una relacion de negocio sino un artefacto de la ventana de captura: los datos van de julio de 2015 a agosto de 2017, asi que 2015 solo tiene semanas altas (27 en adelante) y 2017 solo semanas bajas (hasta la 35). Otra relacion notable es la de `children`/`adults` con `adr`: las reservaciones con mas huespedes pagan una tarifa mayor (habitaciones mas grandes).

    ### Posibles problemas

    observando los scatterplots, boxplots, cuartiles, sesgos y kurtosis. Podemos concluir que hay una cantidad considerable de variables (en especial las dicretas) muy sesgadas y con mucha _masa_ en un valor especifico. por nombrar algunas de estas variables `adults, children, babies, previous_cancelation, days_in_waiting_list,` etc. Se sabe a priori tan solo con los histogramas que estas variables presentan, no son procesos normalmente distribuidos, y queda pendiente decidir como se trataran particularmente. Ya sea con _transformaciones de potencia_ (box-cox), o _discretizandolas_ de la forma de $\text{has canceled} \to \{0,1\}$, qcuts, reducirlas, kmeans, etc.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Pruebas estadisticas


    #### Normalidad: Kolmogorov-Smirnov (Lilliefors) y D'Agostino-Pearson
    A continuacion realizaremos pruebas estadisticas de normalidad sobre las variables numericas. Como $n = 383 > 50$, usamos la prueba de **Kolmogorov-Smirnov**. Dado que la media y la desviacion estandar de la normal de referencia se estiman con la misma muestra, aplicamos la correccion de **Lilliefors** (el K-S clasico con parametros estimados es demasiado conservador). Como segunda prueba reportamos **D'Agostino-Pearson**, que combina la asimetria y la curtosis. Ambas pruebas tienen como hipotesis nula que los datos provienen de una distribucion normal


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
    _rows = []
    for _col in NUM_COLS:
        _x = df_sample[_col].dropna()
        if _x.nunique() < 2:  # variable constante en la muestra: la prueba no aplica
            _rows.append({"variable": _col})
            continue
        _ks_stat, _ks_p = lilliefors(_x, dist="norm")
        _dp = stats.normaltest(_x)
        _rows.append(
            {
                "variable": _col,
                "KS_Lilliefors": _ks_stat,
                "p_value_KS": _ks_p,
                "DAgostino_K2": _dp.statistic,
                "p_value_DP": _dp.pvalue,
            }
        )

    normality_tests = pd.DataFrame(_rows)
    normality_tests["normal_KS"] = normality_tests["p_value_KS"] >= 0.05
    normality_tests["normal_DP"] = normality_tests["p_value_DP"] >= 0.05
    normality_tests.round(4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Con un nivel de significancia del 5%, ambas pruebas rechazan $H_0$ para todas las variables numericas: hay evidencia de que **ninguna** sigue una distribucion normal. Esto ya se anticipaba por la asimetria y la curtosis: muchas son conteos discretos con gran masa en 0. Implicaciones:

    - Para medir asociacion entre variables numericas daremos prioridad a **Spearman** (no parametrica) sobre Pearson.
    - La regresion logistica **no** supone normalidad de los predictores, por lo que esto no invalida el modelo principal. Sin embargo, una asimetria extrema puede volver no lineal la relacion con el logit, lo que se revisara con Box-Tidwell.
    - Por el TCL ($n = 383 \geq 30$), las medias muestrales son aproximadamente normales, lo que respalda el uso de ANOVA sobre las medias de grupo.

    A continuacion se presentan los histogramas con y sin transformacion $\log(1+x)$ para decidir como tratar cada variable.
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
def _(NUM_COLS, df_sample):
    def correlation_tests(method):
        """Prueba H0: rho = 0 para cada par de variables numericas.
        Reporta r, estadistico t = r*sqrt(n-2)/sqrt(1-r^2) y p-valor."""
        _test = stats.pearsonr if method == "pearson" else stats.spearmanrho
        _rows = []
        for _i, _col_1 in enumerate(NUM_COLS):
            for _col_2 in NUM_COLS[_i + 1 :]:
                _pair = df_sample[[_col_1, _col_2]].dropna()
                if _pair[_col_1].nunique() < 2 or _pair[_col_2].nunique() < 2:
                    continue  # variable constante en la muestra
                _res = _test(_pair[_col_1], _pair[_col_2])
                _r, _n = _res.statistic, len(_pair)
                _rows.append(
                    {
                        "columns": f"{_col_1}-{_col_2}",
                        "results": _r,
                        "t_stat": _r * np.sqrt(_n - 2) / np.sqrt(1 - _r**2),
                        "p_value": _res.pvalue,
                    }
                )
        _df = pd.DataFrame(_rows)
        _df["null_hypothesis"] = (_df["p_value"] < 0.05) ^ 1
        return _df

    return (correlation_tests,)


@app.cell
def _(correlation_tests):
    pearson_coefs = correlation_tests("pearson")
    _df = pearson_coefs
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
    return (pearson_coefs,)


@app.cell
def _(PALETTE, correlation_tests):
    spearman_coefs = correlation_tests("spearman")
    _df = spearman_coefs
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


@app.cell
def _(pearson_coefs, spearman_coefs):
    # Correlaciones mas relevantes: significativas (p < 0.05) y |r| >= 0.2
    top_correlations = (
        spearman_coefs.rename(
            columns={"results": "r_spearman", "t_stat": "t_spearman", "p_value": "p_spearman"}
        )
        .drop(columns="null_hypothesis")
        .merge(
            pearson_coefs.rename(
                columns={"results": "r_pearson", "t_stat": "t_pearson", "p_value": "p_pearson"}
            ).drop(columns="null_hypothesis"),
            on="columns",
        )
        .query("p_spearman < 0.05 and abs(r_spearman) >= 0.2")
        .sort_values("r_spearman", key=abs, ascending=False)
        .round(4)
    )
    top_correlations
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** La tabla anterior muestra los pares con correlacion significativa ($p < 0.05$, se rechaza $H_0: \rho = 0$) y de magnitud al menos debil ($|r| \geq 0.2$). Como las variables no son normales, la referencia principal es Spearman; Pearson se reporta para comparar. Cuando Spearman es claramente mayor que Pearson, la relacion es monotona pero no lineal, o esta dominada por valores atipicos. Con $n = 383$, cualquier $|r| > 0.10$ ya resulta significativo, por lo que ademas de la significancia hay que fijarse en la **magnitud**: la mayoria de las correlaciones son debiles, lo que anticipa que la multicolinealidad no sera un problema serio.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### VIF analysis

    A continuacion evaluamos el VIF de cada una de nuestras features numericas para detectar posibles problemas de correlacion
    """)
    return


@app.cell
def _(BLK, NUM_COLS, PALETTE, df_sample):
    # Se agrega la constante: sin intercepto el VIF queda inflado y no es interpretable.
    # Se omiten variables constantes en la muestra (su VIF no esta definido).
    _cols = [c for c in NUM_COLS if df_sample[c].nunique() > 1]
    _df = add_constant(df_sample[_cols].fillna(0))

    vif_data = pd.DataFrame(
        {
            "feature": _cols,
            "VIF": [
                variance_inflation_factor(_df.values, i + 1)  # i+1: saltar la constante
                for i in range(len(_cols))
            ],
        }
    )

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
    El VIF se calculo incluyendo el intercepto en la regresion auxiliar ($VIF_j = 1/(1-R_j^2)$, con $R_j^2$ el coeficiente de determinacion de regresar la variable $j$ contra las demas). Usamos como umbral $VIF = 5$ (linea punteada). Todas las variables numericas tienen $VIF < 2$ (el maximo es `arrival_date_year` con $\approx 1.7$), muy por debajo del umbral, por lo que **no hay problemas de multicolinealidad** que afecten la estimacion o los errores estandar del modelo. Esto concuerda con las pruebas de correlacion: la mayoria de las relaciones significativas son debiles. La mas fuerte, la del numero de semana con el año ($r \approx -0.53$), es un artefacto de la ventana de captura, y el año no se usara en el modelado. La unica relacion que podria valer la pena investigar es la de `lead_time` con `stays_in_week_nights` (Spearman $\approx 0.29$): las estancias largas se reservan con mas anticipacion.
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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Modelado (Ruta B: regresion logistica)

    ## Preparacion de los datos para el modelo

    Antes de ajustar el modelo revisamos un problema propio de la regresion logistica: la **separacion (cuasi) completa**. Si en algun nivel de un predictor **todas** (o ninguna) de las reservaciones se cancelan, la verosimilitud no tiene maximo finito, el coeficiente diverge a $\pm\infty$ y sus errores estandar y pruebas $z$ dejan de ser validos. La siguiente tabla muestra la tasa de cancelacion por nivel para las variables sospechosas.
    """)
    return


@app.cell
def _(df, df_sample):
    def _rate_table(name, pop_key, smp_key):
        _t = pd.concat(
            [
                df.groupby(pop_key)["is_canceled"].agg(["mean", "size"]).add_suffix("_pob"),
                df_sample.groupby(smp_key)["is_canceled"].agg(["mean", "size"]).add_suffix("_mue"),
            ],
            axis=1,
        ).round(3)
        _t.index = [f"{name} = {i}" for i in _t.index]
        return _t

    separation_check = pd.concat(
        [
            _rate_table("deposit_type", df["deposit_type"], df_sample["deposit_type"]),
            _rate_table(
                "parking > 0",
                df["required_car_parking_spaces"] > 0,
                df_sample["required_car_parking_spaces"] > 0,
            ),
            _rate_table("country == PRT", df["country"] == "PRT", df_sample["country"] == "PRT"),
            _rate_table(
                "waiting_list > 0",
                df["days_in_waiting_list"] > 0,
                df_sample["days_in_waiting_list"] > 0,
            ),
        ]
    )
    separation_check
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Decisiones de preparacion** (todas se toman sin mirar el desempeño del modelo):

    | Variable | Problema | Decision |
    |---|---|---|
    | `deposit_type` | `Non Refund`: 100% cancela en la muestra (99.4% en la poblacion) → separacion completa | Se excluye del modelo; se reporta como hallazgo descriptivo |
    | `required_car_parking_spaces` | 0% cancela cuando se pide estacionamiento → separacion completa; ademas se registra al llegar | Se excluye |
    | `country` | Portugal cancela 57% vs 24% del resto: el pais se confirma al hacer check-in, por lo que filtra informacion del resultado (_leakage_) | Se excluye |
    | `reservation_status(_date)`, `assigned_room_type`, `booking_changes` | Solo se conocen **despues** de reservar | Se excluyen |
    | `days_in_waiting_list` | Solo 6 casos > 0 en la muestra | Se excluye |
    | `distribution_channel` | Redundante con `market_segment` | Se excluye |
    | `agent`, `company` | IDs con cientos de niveles; el nulo significa "sin agencia / empresa" | Indicadoras `has_agent`, `has_company` |
    | `market_segment` | Niveles con 1-3 observaciones | `Aviation`, `Complementary` → `Corporate` (5 niveles) |
    | `customer_type` | Niveles con 1-11 observaciones | Indicadora `transient` |
    | `lead_time`, `adr` | Muy asimetricas, con atipicos (p. ej. `adr` = 5400) | Transformacion $\log(1+x)$: reduce la influencia de atipicos sin eliminar observaciones reales |
    | `previous_cancellations`, `children`/`babies`, `total_of_special_requests` | Masa concentrada en 0 | Indicadoras binarias `had_prev_cancel`, `has_kids`, `has_special_requests` |
    | `stays_in_*_nights` | Correlacionadas entre si ($r = 0.51$) | Se suman en `total_nights` |
    """)
    return


@app.cell
def _(df_sample):
    def prepare_model_data(d):
        """Construye las variables del modelo. Solo usa reglas fijas
        (no estadisticas de otras observaciones), por lo que no hay leakage."""
        out = pd.DataFrame(index=d.index)
        out["is_canceled"] = d["is_canceled"]
        out["city_hotel"] = (d["hotel"] == "City Hotel").astype(int)
        out["log_lead_time"] = np.log1p(d["lead_time"])
        out["log_adr"] = np.log1p(d["adr"].clip(lower=0))
        out["total_nights"] = d["stays_in_weekend_nights"] + d["stays_in_week_nights"]
        out["adults"] = d["adults"]
        out["has_kids"] = ((d["children"].fillna(0) + d["babies"]) > 0).astype(int)
        out["is_repeated_guest"] = d["is_repeated_guest"]
        out["had_prev_cancel"] = (d["previous_cancellations"] > 0).astype(int)
        out["has_special_requests"] = (d["total_of_special_requests"] > 0).astype(int)
        out["has_agent"] = d["agent"].notna().astype(int)
        out["has_company"] = d["company"].notna().astype(int)
        out["transient"] = (d["customer_type"] == "Transient").astype(int)
        out["segment"] = d["market_segment"].replace(
            {"Aviation": "Corporate", "Complementary": "Corporate", "Undefined": "Corporate"}
        )
        return out

    df_model = prepare_model_data(df_sample)
    df_model.describe().T.round(3)
    return (df_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Seleccion de variables

    Partimos de un **modelo completo** con los 12 predictores candidatos mas `segment` (5 niveles, referencia `Online TA`):

    $$
    \text{logit}\,P(\text{cancela}) = \beta_0 + \beta_1\,\text{city\_hotel} + \beta_2\log(1+\text{lead\_time}) + \dots + \sum_k \gamma_k\,\text{segment}_k
    $$

    Aplicamos **eliminacion hacia atras (_backward stepwise_) por AIC**: en cada paso se quita la variable cuya exclusion reduce mas el $AIC = -2\ell + 2k$, y se para cuando ninguna exclusion lo reduce. El AIC penaliza la complejidad, asi que solo se conservan las variables cuyo aporte a la verosimilitud compensa el parametro extra.
    """)
    return


@app.cell
def _(df_model):
    SEGMENT_TERM = "C(segment, Treatment('Online TA'))"
    PREDICTORS_FULL = [
        "city_hotel",
        "log_lead_time",
        "log_adr",
        "total_nights",
        "adults",
        "has_kids",
        "is_repeated_guest",
        "had_prev_cancel",
        "has_special_requests",
        "has_agent",
        "has_company",
        "transient",
        SEGMENT_TERM,
    ]

    def fit_logit(predictors, data=df_model):
        _rhs = " + ".join(predictors) if predictors else "1"
        return smf.logit(f"is_canceled ~ {_rhs}", data=data).fit(disp=0)

    def coef_table(model):
        """Coeficientes con EE, z, p-valor, IC 95% y odds ratios."""
        _ci = model.conf_int()
        _t = pd.DataFrame(
            {
                "coef": model.params,
                "std_err": model.bse,
                "z": model.tvalues,
                "p_value": model.pvalues,
                "IC95_inf": _ci[0],
                "IC95_sup": _ci[1],
                "odds_ratio": np.exp(model.params),
                "OR_IC95_inf": np.exp(_ci[0]),
                "OR_IC95_sup": np.exp(_ci[1]),
            }
        )
        _t.index = _t.index.str.replace(f"{SEGMENT_TERM}[T.", "segment[", regex=False)
        _t["significativo"] = _t["p_value"] < 0.05
        return _t.round(4)

    logit_full = fit_logit(PREDICTORS_FULL)
    mo.vstack([mo.md("**Modelo completo**"), coef_table(logit_full)])
    return PREDICTORS_FULL, coef_table, fit_logit, logit_full


@app.cell
def _(PREDICTORS_FULL, fit_logit):
    # Backward stepwise por AIC
    _current = PREDICTORS_FULL.copy()
    _model = fit_logit(_current)
    _steps = [{"paso": 0, "variable_eliminada": "-", "AIC": _model.aic, "BIC": _model.bic}]

    while True:
        _candidates = [
            (fit_logit([p for p in _current if p != x]).aic, x) for x in _current
        ]
        _best_aic, _drop = min(_candidates)
        if _best_aic >= _model.aic:
            break
        _current.remove(_drop)
        _model = fit_logit(_current)
        _steps.append(
            {
                "paso": len(_steps),
                "variable_eliminada": _drop,
                "AIC": _model.aic,
                "BIC": _model.bic,
            }
        )

    PREDICTORS_REDUCED = _current
    logit_reduced = _model
    selection_steps = pd.DataFrame(_steps).round(2)
    selection_steps
    return PREDICTORS_REDUCED, logit_reduced


@app.cell
def _(logit_full, logit_reduced):
    # Comparacion modelo completo vs reducido: prueba de razon de verosimilitudes
    _lr = 2 * (logit_full.llf - logit_reduced.llf)
    _df_diff = int(logit_full.df_model - logit_reduced.df_model)
    _p = stats.chi2.sf(_lr, _df_diff)

    def _row(name, m):
        return {
            "modelo": name,
            "parametros": int(m.df_model + 1),
            "log_verosimilitud": m.llf,
            "devianza": -2 * m.llf,
            "AIC": m.aic,
            "BIC": m.bic,
            "pseudo_R2_McFadden": m.prsquared,
        }

    model_comparison = pd.DataFrame(
        [
            {
                "modelo": "nulo",
                "parametros": 1,
                "log_verosimilitud": logit_full.llnull,
                "devianza": -2 * logit_full.llnull,
            },
            _row("completo", logit_full),
            _row("reducido (AIC)", logit_reduced),
        ]
    ).round(3)

    lrt_full_vs_reduced = {"chi2": _lr, "gl": _df_diff, "p_value": _p}
    model_comparison
    return (lrt_full_vs_reduced,)


@app.cell(hide_code=True)
def _(lrt_full_vs_reduced):
    _r = lrt_full_vs_reduced
    mo.md(rf"""
    **Prueba de razon de verosimilitudes (completo vs reducido)**

    $H_0$: los coeficientes de las {_r["gl"]} variables eliminadas son todos cero.

    $G^2 = 2(\ell_{{completo}} - \ell_{{reducido}}) = {_r["chi2"]:.3f}$, $gl = {_r["gl"]}$, $p = {_r["p_value"]:.4f}$
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion de la seleccion.**

    - El backward por AIC elimino, en orden, `log_adr`, `has_agent`, `city_hotel`, `has_kids`, `has_company`, `total_nights` y `adults`. El AIC bajo de 410.9 a 401.9 y el BIC de 478.1 a 441.4: ambos criterios prefieren el modelo reducido.
    - La prueba de razon de verosimilitudes **no rechaza** $H_0$ ($G^2 = 4.99$, $gl = 7$, $p = 0.66$): las 7 variables eliminadas no aportan informacion significativa una vez que se conocen las demas.
    - **Cambio en la devianza:** al quitar las 7 variables la devianza residual solo sube de 376.9 a 381.9 (+5.0), mientras que respecto al modelo nulo (505.1) el modelo reducido explica una reduccion de 123.1 unidades de devianza con solo 9 parametros. El pseudo-$R^2$ de McFadden apenas pasa de 0.254 a 0.244.
    - Llama la atencion que `log_adr` (el precio) y `city_hotel` salgan del modelo aunque en el EDA parecian asociados con la cancelacion: su efecto se explica por otras variables (p. ej. el segmento de mercado y la anticipacion de la reserva), con las que estan correlacionados.

    El **modelo reducido** conserva `log_lead_time`, `is_repeated_guest`, `had_prev_cancel`, `has_special_requests`, `transient` y `segment`. Sin embargo, ninguno de los 4 niveles de `segment` es significativo por separado, por lo que en la siguiente seccion probamos su significancia conjunta con una prueba F parcial (en su version de razon de verosimilitudes) antes de fijar el modelo principal.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modelo principal y pruebas de significancia

    ### Prueba conjunta de `segment`

    $H_0: \gamma_{Corporate} = \gamma_{Direct} = \gamma_{Groups} = \gamma_{Offline} = 0$ (el segmento de mercado no aporta una vez que se conocen las demas variables).
    """)
    return


@app.cell
def _(PREDICTORS_REDUCED, fit_logit, logit_reduced):
    PREDICTORS_FINAL = [p for p in PREDICTORS_REDUCED if not p.startswith("C(segment")]
    logit_final = fit_logit(PREDICTORS_FINAL)

    _g2 = 2 * (logit_reduced.llf - logit_final.llf)
    _gl = int(logit_reduced.df_model - logit_final.df_model)
    _p = stats.chi2.sf(_g2, _gl)

    segment_test = pd.DataFrame(
        {
            "modelo": ["con segment", "sin segment"],
            "parametros": [int(logit_reduced.df_model + 1), int(logit_final.df_model + 1)],
            "devianza": [-2 * logit_reduced.llf, -2 * logit_final.llf],
            "AIC": [logit_reduced.aic, logit_final.aic],
            "BIC": [logit_reduced.bic, logit_final.bic],
        }
    ).round(3)

    mo.vstack(
        [
            segment_test,
            mo.md(rf"$G^2 = {_g2:.3f}$, $gl = {_gl}$, $p = {_p:.4f}$"),
        ]
    )
    return PREDICTORS_FINAL, logit_final


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Con $\alpha = 0.05$ **no se rechaza** $H_0$ ($p \approx 0.087$): el segmento de mercado no tiene un efecto significativo en conjunto. Ademas, el AIC de ambos modelos es practicamente identico (401.9 vs 402.0), mientras que el BIC, que penaliza mas la complejidad, prefiere claramente el modelo sin `segment` (441.4 vs 425.7). Por parsimonia, **eliminamos `segment`**: se ganan 4 grados de libertad sin perder capacidad explicativa significativa.
    """)
    return


@app.cell(hide_code=True)
def _(PREDICTORS_FINAL, logit_final):
    _b = logit_final.params
    _names = {
        "log_lead_time": r"\log(1+\text{lead\_time})",
        "is_repeated_guest": r"\text{repeated\_guest}",
        "had_prev_cancel": r"\text{prev\_cancel}",
        "has_special_requests": r"\text{special\_requests}",
        "transient": r"\text{transient}",
    }
    _terms = " ".join(
        f"{'+' if _b[p] >= 0 else '-'} {abs(_b[p]):.3f}\\,{_names[p]}"
        for p in PREDICTORS_FINAL
    )

    mo.md(
        rf"""
    ### Especificacion del modelo ajustado

    $$
    \log\frac{{\hat p}}{{1-\hat p}} = {_b['Intercept']:.3f} {_terms}
    $$

    donde $\hat p = \hat P(\text{{is\_canceled}} = 1 \mid X)$. Equivalentemente, $\hat p = 1/(1 + e^{{-\hat\eta}})$ con $\hat\eta$ el lado derecho.
    """
    )
    return


@app.cell(hide_code=True)
def _(logit_final):
    mo.md(rf"""
    ### Significancia global

    Prueba de razon de verosimilitudes frente al modelo nulo (solo intercepto):
    $H_0: \beta_1 = \beta_2 = \dots = \beta_5 = 0$.

    | $\chi^2 = 2(\ell_{{modelo}} - \ell_{{nulo}})$ | gl | p-valor | Devianza nula | Devianza residual | Pseudo-$R^2$ (McFadden) |
    |---|---|---|---|---|---|
    | {logit_final.llr:.3f} | {int(logit_final.df_model)} | {logit_final.llr_pvalue:.2e} | {-2 * logit_final.llnull:.3f} | {-2 * logit_final.llf:.3f} | {logit_final.prsquared:.3f} |

    Se rechaza $H_0$ contundentemente ($p \approx 10^{{-23}}$): **al menos una** de las variables del modelo esta asociada con la probabilidad de cancelacion. El modelo reduce la devianza en {logit_final.llr:.1f} unidades usando solo 5 grados de libertad. Un pseudo-$R^2$ de {logit_final.prsquared:.2f} se considera un ajuste aceptable en regresion logistica (valores de 0.2 a 0.4 suelen indicar un buen ajuste en la escala de McFadden).
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, coef_table, fit_logit, logit_final):
    # Significancia individual: prueba z (Wald) y prueba de razon de verosimilitudes por variable
    _lrt = {}
    for _v in PREDICTORS_FINAL:
        _m0 = fit_logit([p for p in PREDICTORS_FINAL if p != _v])
        _g2 = 2 * (logit_final.llf - _m0.llf)
        _lrt[_v] = {"G2_LRT": _g2, "p_value_LRT": stats.chi2.sf(_g2, 1)}

    final_coefs = coef_table(logit_final).join(pd.DataFrame(_lrt).T.round(4))
    mo.vstack([mo.md("### Coeficientes del modelo final (IC al 95%)"), final_coefs])
    return (final_coefs,)


@app.cell
def _(BLK, PALETTE, final_coefs):
    _df = final_coefs.drop(index="Intercept").reset_index(names="variable")
    _base = alt.Chart(_df).encode(
        y=alt.Y("variable:N", sort="-x", title=None),
    )
    _ci = _base.mark_rule(strokeWidth=3, color=PALETTE[0]).encode(
        x=alt.X(
            "OR_IC95_inf:Q",
            scale=alt.Scale(type="log"),
            title="odds ratio (escala log) con IC 95%",
        ),
        x2="OR_IC95_sup:Q",
    )
    _pt = _base.mark_point(filled=True, size=120, color=PALETTE[1], opacity=1).encode(
        x="odds_ratio:Q",
        tooltip=["variable", "odds_ratio", "OR_IC95_inf", "OR_IC95_sup", "p_value_LRT"],
    )
    _one = (
        alt.Chart(pd.DataFrame({"x": [1]}))
        .mark_rule(strokeDash=[6, 3], color=BLK, opacity=1)
        .encode(x="x:Q")
    )
    (_ci + _pt + _one).properties(
        title="Efecto de cada variable sobre los momios de cancelar", width=600, height=250
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretacion de los coeficientes ($\alpha = 0.05$)

    Todas las variables son significativas segun la prueba de razon de verosimilitudes. El odds ratio $e^{\beta}$ indica por cuanto se **multiplican los momios** de cancelar ante un cambio de una unidad, manteniendo lo demas constante.

    - **`log_lead_time`** ($\hat\beta = 0.598$, OR = 1.82, IC 95% [1.48, 2.23], $p < 0.001$). Como la variable esta en logaritmo, la lectura natural es: **duplicar la anticipacion** de la reserva multiplica los momios de cancelar por $2^{0.598} \approx 1.51$ (+51%, IC [+31%, +74%]). Reservar con mucha anticipacion deja mas tiempo para cambiar de planes o para encontrar una mejor oferta.
    - **`transient`** ($\hat\beta = 1.839$, OR = 6.29, IC [3.26, 12.13], $p < 0.001$). Una reservacion individual (no ligada a contrato ni a grupo) tiene unos momios de cancelar **6.3 veces** mayores. Es el efecto mas robusto del modelo: no hay un compromiso contractual que la ate.
    - **`has_special_requests`** ($\hat\beta = -1.068$, OR = 0.34, IC [0.21, 0.57], $p < 0.001$). Hacer al menos una solicitud especial **reduce los momios de cancelar en 66%** (IC: entre 43% y 79%). Es una señal de compromiso: quien pide una cuna o un piso alto ya esta planeando su estancia.
    - **`had_prev_cancel`** ($\hat\beta = 3.351$, OR = 28.5, IC [3.4, 240.6], $p = 0.002$). Un cliente con cancelaciones previas tiene momios de cancelar **unas 28 veces** mayores. La direccion y la significancia son claras, pero el intervalo es muy amplio porque solo hay 17 casos en la muestra: la magnitud exacta es incierta.
    - **`is_repeated_guest`** ($\hat\beta = -3.201$, OR = 0.04, IC [0.002, 1.10]). Los huespedes recurrentes cancelan mucho menos. Aqui la prueba $z$ de Wald es marginal ($p = 0.057$) y su intervalo toca el 1, pero la prueba de razon de verosimilitudes si es significativa ($G^2 = 5.40$, $p = 0.020$). Con pocos casos (13 huespedes recurrentes) y un coeficiente grande, el error estandar de Wald se infla y la prueba pierde potencia (efecto **Hauck-Donner**); en esa situacion la prueba de razon de verosimilitudes es la mas confiable. Interpretamos el efecto como **significativo pero estimado con poca precision**.
    - **Intercepto** ($\hat\beta_0 = -4.075$). Corresponde a una reserva hecha el mismo dia, no transitoria, sin historial y sin solicitudes: su probabilidad estimada de cancelar es de apenas ~1.7%.

    **Limitacion.** Los intervalos amplios de `had_prev_cancel` e `is_repeated_guest` son consecuencia directa del tamaño de muestra ($n = 383$) y de lo raros que son esos perfiles; con la poblacion completa se estimarian con mucha mas precision.
    """)
    return


@app.cell
def _(logit_final):
    # Interpretacion practica: probabilidad estimada de cancelar en perfiles tipicos
    _profiles = pd.DataFrame(
        {
            "perfil": [
                "Transitoria, 70 dias, sin solicitudes",
                "Transitoria, 70 dias, con solicitud especial",
                "Transitoria, 7 dias, con solicitud especial",
                "Transitoria, 200 dias, sin solicitudes",
                "Contrato/grupo, 70 dias, sin solicitudes",
                "Transitoria, 70 dias, cancelo antes",
                "Huesped recurrente transitorio, 70 dias",
            ],
            "lead_time": [70, 70, 7, 200, 70, 70, 70],
            "transient": [1, 1, 1, 1, 0, 1, 1],
            "has_special_requests": [0, 1, 1, 0, 0, 0, 0],
            "had_prev_cancel": [0, 0, 0, 0, 0, 1, 0],
            "is_repeated_guest": [0, 0, 0, 0, 0, 0, 1],
        }
    )
    _profiles["log_lead_time"] = np.log1p(_profiles["lead_time"])
    _profiles["P_cancelar"] = logit_final.predict(_profiles).round(3)
    _profiles.drop(columns="log_lead_time")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    La tabla anterior traduce el modelo a lenguaje de negocio: una reserva transitoria tipica (70 dias de anticipacion, mediana de la muestra) tiene una probabilidad estimada de cancelar cercana al 58%. Si el cliente hace una solicitud especial baja a ~32%, y si ademas reserva con una semana de anticipacion cae a ~11%. Un cliente con cancelaciones previas llega a ~97%, y un huesped recurrente baja a ~5%. En cambio, el mismo perfil ligado a un contrato o grupo tiene ~18%. Estos contrastes son la base de las recomendaciones del reporte ejecutivo.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verificacion de supuestos

    La regresion logistica no supone normalidad ni homocedasticidad, pero si requiere:

    1. **Linealidad en el logit** de los predictores continuos.
    2. **Ausencia de multicolinealidad** entre los predictores.
    3. **Buen ajuste**: que las probabilidades predichas concuerden con las frecuencias observadas, y que la devianza residual no muestre sobredispersion ni observaciones excesivamente influyentes.
    4. **Independencia** de las observaciones: se garantiza por diseño, pues la muestra es aleatoria (cada reservacion se selecciono de forma independiente).

    ### 1. Linealidad en el logit: prueba de Box-Tidwell

    Solo aplica a `log_lead_time`, porque las demas variables son binarias (una variable binaria siempre es lineal en el logit). Se agrega al modelo el termino $x\ln(x)$ y se prueba $H_0: \beta_{x\ln x} = 0$. Si se rechaza, la relacion entre $x$ y el logit **no** es lineal. Como $x$ debe ser positiva, usamos $x + 1$.

    Aplicamos la prueba a la variable original (`lead_time`) y a la transformada (`log_lead_time`) para evaluar si la transformacion corrige el problema.
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, df_model):
    _d = df_model.copy()
    _others = [p for p in PREDICTORS_FINAL if p != "log_lead_time"]

    # Variable original: x = lead_time + 1
    _d["x_raw"] = np.expm1(_d["log_lead_time"]) + 1
    _d["bt_raw"] = _d["x_raw"] * np.log(_d["x_raw"])
    # Variable transformada: x = log(1 + lead_time) + 1
    _d["x_log"] = _d["log_lead_time"] + 1
    _d["bt_log"] = _d["x_log"] * np.log(_d["x_log"])

    _rows = []
    for _name, _x, _bt in [
        ("lead_time (original)", "x_raw", "bt_raw"),
        ("log(1 + lead_time) (transformada)", "x_log", "bt_log"),
    ]:
        _m = smf.logit(
            f"is_canceled ~ {_x} + {_bt} + " + " + ".join(_others), data=_d
        ).fit(disp=0)
        _rows.append(
            {
                "variable": _name,
                "coef_x_ln_x": _m.params[_bt],
                "z": _m.tvalues[_bt],
                "p_value": _m.pvalues[_bt],
                "lineal_en_logit": _m.pvalues[_bt] >= 0.05,
            }
        )
    box_tidwell = pd.DataFrame(_rows).round(4)
    box_tidwell
    return


@app.cell
def _(PALETTE, df_model):
    # Logit empirico por decil: visualiza la (no) linealidad
    _d = df_model.assign(lead_time=np.expm1(df_model["log_lead_time"]))
    _d["decil"] = pd.qcut(_d["lead_time"], 10, duplicates="drop")
    _g = _d.groupby("decil", observed=True).agg(
        lead_time=("lead_time", "mean"),
        log_lead_time=("log_lead_time", "mean"),
        p=("is_canceled", "mean"),
        n=("is_canceled", "size"),
    )
    # correccion de 0.5 para evitar logit infinito en deciles sin cancelaciones
    _g["logit_empirico"] = np.log(
        (_g["p"] * _g["n"] + 0.5) / ((1 - _g["p"]) * _g["n"] + 0.5)
    )
    _g = _g.reset_index(drop=True)

    def _panel(x, title):
        _base = alt.Chart(_g).encode(
            x=alt.X(f"{x}:Q", title=title),
            y=alt.Y("logit_empirico:Q", title="logit empirico de cancelar"),
        )
        return (
            _base.mark_point(filled=True, size=80, color=PALETTE[1], opacity=1)
            + _base.transform_regression(x, "logit_empirico").mark_line(color=PALETTE[0])
        ).properties(width=300, height=250)

    (
        _panel("lead_time", "lead_time (dias)") | _panel("log_lead_time", "log(1 + lead_time)")
    ).properties(title="Logit empirico por decil de anticipacion")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** Con la variable original se **rechaza** la linealidad en el logit ($p \approx 0.005$): el efecto de la anticipacion se satura, pues pasar de 0 a 30 dias cambia mucho mas el riesgo que pasar de 300 a 330. Esta es una violacion del supuesto, y la **correccion** es justamente la transformacion $\log(1 + \text{lead\_time})$ que aplicamos al preparar los datos. Al **reverificar** con la variable transformada, la prueba ya no rechaza $H_0$ ($p \approx 0.93$): la relacion es lineal en el logit. La grafica del logit empirico por decil lo confirma: con la variable original los puntos se curvan, con la transformada siguen la recta.

    ### 2. Multicolinealidad (VIF) de los predictores finales
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, df_model):
    _X = add_constant(df_model[PREDICTORS_FINAL])
    vif_final = pd.DataFrame(
        {
            "variable": PREDICTORS_FINAL,
            "VIF": [
                variance_inflation_factor(_X.values, i + 1)
                for i in range(len(PREDICTORS_FINAL))
            ],
        }
    ).round(3)
    vif_final
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Todos los VIF son menores a 1.2, lejos del umbral de 5: los predictores del modelo final son practicamente independientes entre si, por lo que los errores estandar de los coeficientes no estan inflados por colinealidad.

    ### 3a. Bondad de ajuste: prueba de Hosmer-Lemeshow

    Se ordenan las observaciones por su probabilidad predicha, se forman $g = 10$ grupos (deciles) y se compara el numero observado de cancelaciones $O_k$ con el esperado $E_k = \sum \hat p_i$:

    $$
    \hat C = \sum_{k=1}^{g} \frac{(O_k - E_k)^2}{E_k\,(1 - E_k/n_k)} \;\sim\; \chi^2_{g-2}
    $$

    $H_0$: el modelo ajusta bien (no hay diferencia entre lo observado y lo esperado), lo que tambien indica que la relacion en la escala logit es adecuada (linealidad en el logit). $H_1$: el modelo no ajusta bien (posible no linealidad en el logit).

    La expresion anterior es equivalente a la forma $\sum_k \left[\frac{(O_k - E_k)^2}{E_k} + \frac{(O^{no}_k - E^{no}_k)^2}{E^{no}_k}\right]$, que suma eventos y no eventos, porque $\frac{1}{E_k} + \frac{1}{n_k - E_k} = \frac{1}{E_k(1 - E_k/n_k)}$.
    """)
    return


@app.cell
def _(BLK, PALETTE, df_model, logit_final):
    _p_hat = logit_final.predict(df_model)
    hosmer_lemeshow_table = (
        pd.DataFrame(
            {
                "y": df_model["is_canceled"],
                "p_hat": _p_hat,
                "grupo": pd.qcut(_p_hat, 10, labels=False, duplicates="drop") + 1,
            }
        )
        .groupby("grupo")
        .agg(
            n=("y", "size"),
            observados=("y", "sum"),
            esperados=("p_hat", "sum"),
            p_media=("p_hat", "mean"),
        )
    )
    hosmer_lemeshow_table["tasa_observada"] = (
        hosmer_lemeshow_table["observados"] / hosmer_lemeshow_table["n"]
    )
    _t = hosmer_lemeshow_table
    _hl = (
        ((_t["observados"] - _t["esperados"]) ** 2)
        / (_t["esperados"] * (1 - _t["esperados"] / _t["n"]))
    ).sum()
    _gl = len(_t) - 2
    _p = stats.chi2.sf(_hl, _gl)

    _calib = (
        alt.Chart(_t.reset_index())
        .mark_point(filled=True, size=100, color=PALETTE[1], opacity=1)
        .encode(
            x=alt.X("p_media:Q", title="probabilidad predicha media", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tasa_observada:Q", title="tasa observada de cancelacion", scale=alt.Scale(domain=[0, 1])),
            tooltip=["grupo", "n", "observados", alt.Tooltip("esperados:Q", format=".2f")],
        )
    )
    _diag = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(strokeDash=[6, 3], color=BLK)
        .encode(x="x:Q", y="y:Q")
    )
    hosmer_lemeshow = {"C": _hl, "gl": _gl, "p_value": _p}
    mo.vstack(
        [
            _t.round(3),
            mo.md(rf"**Hosmer-Lemeshow:** $\hat C = {_hl:.3f}$, $gl = {_gl}$, $p = {_p:.4f}$"),
            (_diag + _calib).properties(title="Calibracion por decil (Hosmer-Lemeshow)", width=350, height=350),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** No se rechaza $H_0$ ($p \approx 0.12$): no hay evidencia de falta de ajuste. En la grafica de calibracion los deciles quedan cerca de la diagonal, es decir, cuando el modelo predice ~70% de probabilidad de cancelar, efectivamente cancelan alrededor del 70% de esas reservas. Las mayores desviaciones estan en los deciles 3 y 4 (probabilidades de 15-25%), donde el modelo sobreestima ligeramente el riesgo, pero sin llegar a ser significativas.

    ### 3b. Devianza residual y su distribucion

    La devianza residual $D = -2\ell(\hat\beta)$ mide cuanto se aleja el modelo de uno saturado. Comparamos $D$ con sus grados de libertad: $D/gl \approx 1$ indica que no hay sobredispersion. **Advertencia:** con respuesta binaria (un ensayo por observacion), $D$ no sigue una $\chi^2_{n-p}$ ni siquiera asintoticamente, por lo que su p-valor se reporta solo como referencia; la prueba formal de ajuste es Hosmer-Lemeshow.

    Revisamos tambien los **residuos de devianza** $d_i$ (valores $|d_i| > 2$ son poco usuales) y la **influencia** de cada observacion con la distancia de Cook ($D_i > 4/n$ como señal de alerta, $D_i > 1$ como influencia grave).
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, df_model, logit_final):
    # El GLM binomial da los mismos coeficientes que Logit y ademas las medidas de influencia
    glm_final = smf.glm(
        "is_canceled ~ " + " + ".join(PREDICTORS_FINAL),
        data=df_model,
        family=sm.families.Binomial(),
    ).fit()
    influence_final = glm_final.get_influence().summary_frame()
    influence_final["resid_devianza"] = glm_final.resid_deviance
    influence_final["p_hat"] = glm_final.fittedvalues
    influence_final["is_canceled"] = df_model["is_canceled"]

    _D = glm_final.deviance
    _gl = glm_final.df_resid
    _n = len(df_model)
    deviance_summary = pd.DataFrame(
        {
            "devianza_residual": [_D],
            "gl": [_gl],
            "D/gl": [_D / _gl],
            "p_value_ref": [stats.chi2.sf(_D, _gl)],
            "residuos |d|>2": [(influence_final["resid_devianza"].abs() > 2).sum()],
            "Cook > 4/n": [(influence_final["cooks_d"] > 4 / _n).sum()],
            "Cook > 1": [(influence_final["cooks_d"] > 1).sum()],
            "Cook max": [influence_final["cooks_d"].max()],
            "coefs_iguales_a_Logit": [np.allclose(glm_final.params, logit_final.params)],
        }
    ).round(4)
    deviance_summary
    return (influence_final,)


@app.cell
def _(BLK, PALETTE, df_model, influence_final):
    _d = influence_final.reset_index(names="obs")
    _d["is_canceled"] = _d["is_canceled"].astype(str)
    _d["indice"] = range(len(_d))
    _n = len(df_model)

    _resid = (
        alt.Chart(_d)
        .mark_point(filled=True, size=30)
        .encode(
            x=alt.X("p_hat:Q", title="probabilidad predicha"),
            y=alt.Y("resid_devianza:Q", title="residuo de devianza"),
            color=alt.Color("is_canceled:N", title="is_canceled"),
        )
        .properties(title="Residuos de devianza vs ajustados", width=330, height=280)
    )
    _bands = (
        alt.Chart(pd.DataFrame({"y": [-2, 2]}))
        .mark_rule(strokeDash=[6, 3], color=BLK, opacity=1)
        .encode(y="y:Q")
    )
    _hist = (
        alt.Chart(_d)
        .mark_bar(color=PALETTE[3])
        .encode(
            x=alt.X("resid_devianza:Q", bin=alt.Bin(maxbins=30), title="residuo de devianza"),
            y=alt.Y("count()", title="frecuencia"),
        )
        .properties(title="Distribucion de los residuos de devianza", width=330, height=280)
    )
    _cook = (
        alt.Chart(_d)
        .mark_bar(color=PALETTE[0])
        .encode(
            x=alt.X("indice:Q", title="observacion"),
            y=alt.Y("cooks_d:Q", title="distancia de Cook"),
            tooltip=["obs", "cooks_d", "p_hat", "is_canceled"],
        )
        .properties(title="Distancia de Cook", width=690, height=200)
    )
    _cook_line = (
        alt.Chart(pd.DataFrame({"y": [4 / _n]}))
        .mark_rule(strokeDash=[6, 3], color=PALETTE[-1], opacity=1)
        .encode(y="y:Q")
    )
    alt.vconcat((_resid + _bands) | _hist, _cook + _cook_line)
    return


@app.cell
def _(PREDICTORS_FINAL, df_model, influence_final):
    # Observaciones mas influyentes
    _top = influence_final["cooks_d"].nlargest(5).index
    top_influential = df_model.loc[_top, PREDICTORS_FINAL + ["is_canceled"]].assign(
        lead_time=np.expm1(df_model.loc[_top, "log_lead_time"]).round(),
        p_hat=influence_final.loc[_top, "p_hat"].round(3),
        cooks_d=influence_final.loc[_top, "cooks_d"].round(3),
    )
    top_influential
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** La devianza residual es practicamente igual a sus grados de libertad ($D/gl \approx 1.03$), por lo que no hay sobredispersion. Los residuos de devianza tienen la forma bimodal esperada en un modelo binario (una banda por cada valor de $y$) y solo 7 de 383 (1.8%) superan $|d| > 2$, menos del 5% que se esperaria por azar. Ninguna observacion tiene $D_i > 1$.

    Sin embargo, **dos observaciones destacan** con $D_i \approx 0.5$, muy por encima del resto ($< 0.08$). Ambas son **huespedes recurrentes**: una es el **unico** huesped recurrente de la muestra que cancelo (y lo hizo con una reserva del mismo dia, a la que el modelo le daba 0.4% de probabilidad), y la otra es un huesped recurrente con cancelaciones previas que no cancelo. Su influencia se debe a que `is_repeated_guest` se estima con solo 13 casos. **No las eliminamos**: son reservas validas, y quitar la primera haria que ningun huesped recurrente cancele, lo que provocaria separacion completa en esa variable.

    En su lugar hacemos un **analisis de sensibilidad**: reestimamos el modelo **sin** `is_repeated_guest` y verificamos si los demas coeficientes cambian.
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, fit_logit, logit_final):
    _m = fit_logit([p for p in PREDICTORS_FINAL if p != "is_repeated_guest"])
    sensitivity = pd.DataFrame(
        {
            "coef_final": logit_final.params,
            "coef_sin_repeated": _m.params,
            "p_final": logit_final.pvalues,
            "p_sin_repeated": _m.pvalues,
        }
    )
    sensitivity["cambio_%"] = (
        100
        * (sensitivity["coef_sin_repeated"] - sensitivity["coef_final"])
        / sensitivity["coef_final"].abs()
    )
    sensitivity = sensitivity.round(4)
    sensitivity
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion de la sensibilidad.** Al quitar `is_repeated_guest`, los coeficientes de `log_lead_time`, `transient` y `has_special_requests` cambian menos de 6% y siguen siendo altamente significativos: **los hallazgos principales son robustos**. El unico que se mueve de forma apreciable es `had_prev_cancel`, que baja 23% (su OR pasa de ~28 a ~13), aunque sigue siendo significativo ($p \approx 0.002$). Esto ocurre porque ambas variables comparten observaciones: uno de los puntos influyentes es un huesped recurrente con cancelaciones previas. La conclusion practica es que **haber cancelado antes multiplica fuertemente el riesgo de cancelar**, pero la magnitud exacta (entre ~13 y ~28 veces) depende de pocos casos y debe reportarse con cautela.

    ### 4. Tamaño de muestra suficiente (formula de Hsieh)

    Verificamos que $n = 383$ alcance para detectar los efectos estimados con $\alpha = 0.05$ y una potencia de 80% ($\beta = 0.20$):

    $$
    N = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{P_0(1-P_0)\,(\ln OR)^2}
    $$

    donde $P_0$ es la proporcion de cancelaciones en la muestra y $OR$ el odds ratio estimado de cada predictor. Para `log_lead_time` (continua) se reporta tambien el OR por una desviacion estandar, que es la escala habitual de la formula para predictores continuos.
    """)
    return


@app.cell
def _(PREDICTORS_FINAL, df_model, logit_final):
    _z_alpha = stats.norm.ppf(1 - 0.05 / 2)
    _z_beta = stats.norm.ppf(1 - 0.20)
    _p0 = df_model["is_canceled"].mean()
    _n = len(df_model)

    _rows = []
    for _v in PREDICTORS_FINAL:
        _b = logit_final.params[_v]
        _scale = df_model[_v].std() if _v == "log_lead_time" else 1.0
        _ln_or = _b * _scale
        _N = (_z_alpha + _z_beta) ** 2 / (_p0 * (1 - _p0) * _ln_or**2)
        _rows.append(
            {
                "variable": _v,
                "escala_OR": "1 desv. est." if _v == "log_lead_time" else "0 → 1",
                "OR": np.exp(_ln_or),
                "N_requerido": _N,
                "n_muestra": _n,
                "suficiente": _n >= _N,
            }
        )
    # Para log_lead_time agregamos tambien el OR por unidad (escala mas exigente)
    _b = logit_final.params["log_lead_time"]
    _rows.append(
        {
            "variable": "log_lead_time",
            "escala_OR": "1 unidad",
            "OR": np.exp(_b),
            "N_requerido": (_z_alpha + _z_beta) ** 2 / (_p0 * (1 - _p0) * _b**2),
            "n_muestra": _n,
            "suficiente": _n >= (_z_alpha + _z_beta) ** 2 / (_p0 * (1 - _p0) * _b**2),
        }
    )
    hsieh_table = pd.DataFrame(_rows).round(3)
    mo.vstack([mo.md(rf"$P_0 = {_p0:.3f}$"), hsieh_table])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** En todos los casos $n = 383$ supera el tamaño requerido (el mas exigente es `log_lead_time` por unidad, con $N \approx 94$), por lo que la muestra es **suficiente** para detectar los efectos del modelo. **Precaucion:** esta version de la formula supone un predictor balanceado. Para predictores binarios muy poco frecuentes, como `is_repeated_guest` (3.4%) y `had_prev_cancel` (4.4%), el tamaño realmente necesario es mayor, lo cual es consistente con los intervalos tan amplios que obtuvimos para esas dos variables.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Conclusion de la verificacion de supuestos.**

    | Supuesto | Prueba | Resultado | Decision |
    |---|---|---|---|
    | Linealidad en el logit | Box-Tidwell | Se viola con `lead_time` ($p \approx 0.005$); se cumple con $\log(1+\text{lead\_time})$ ($p \approx 0.93$) | Corregido con la transformacion log y reverificado |
    | Multicolinealidad | VIF | Todos $< 1.2$ | Se cumple |
    | Bondad de ajuste / linealidad en el logit | Hosmer-Lemeshow | $\hat C = 12.78$, $gl = 8$, $p \approx 0.12$ | Se cumple |
    | Tamaño de muestra | Formula de Hsieh | $N$ requerido $\leq 94 < 383$ | Se cumple (con cautela en predictores raros) |
    | Sobredispersion | $D/gl$ | $\approx 1.03$ | Se cumple |
    | Observaciones influyentes | Distancia de Cook | Ninguna $> 1$; 2 con $D_i \approx 0.5$ (huespedes recurrentes) | Se conservan; ver analisis de sensibilidad |
    | Independencia | Diseño muestral | Muestra aleatoria | Se cumple |

    El modelo final cumple los supuestos de la regresion logistica, por lo que sus pruebas de significancia e intervalos de confianza son validos.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # ANOVA / ANCOVA

    **Pregunta:** ¿la tasa de cancelacion difiere entre los segmentos de mercado (`segment`: Online TA, Offline TA/TO, Groups, Direct, Corporate)?

    Como la variable dependiente es binaria, la media de `is_canceled` en cada grupo es la **tasa de cancelacion** del segmento, de modo que el ANOVA compara directamente esas tasas. La prueba F se justifica por el TCL: con $n = 383$ (el grupo mas pequeño tiene 22), las medias de grupo son aproximadamente normales. Como con una respuesta 0/1 la varianza de cada grupo es $p_k(1-p_k)$ y depende de su propia tasa, verificamos la homogeneidad de varianzas con **Levene** y confirmamos el resultado con el **ANOVA de Welch** (que no supone varianzas iguales) y con **Kruskal-Wallis** (no parametrica).

    $$
    H_0: \mu_{Corporate} = \mu_{Direct} = \mu_{Groups} = \mu_{Offline} = \mu_{Online} \qquad H_1: \text{al menos una media difiere}
    $$
    """)
    return


@app.cell
def _(df_model):
    segment_summary = df_model.groupby("segment").agg(
        n=("is_canceled", "size"),
        tasa_cancelacion=("is_canceled", "mean"),
        desv_est=("is_canceled", "std"),
        media_log_lead_time=("log_lead_time", "mean"),
        mediana_lead_time=("log_lead_time", lambda s: np.expm1(s).median()),
        prop_transient=("transient", "mean"),
    )
    segment_summary.round(3)
    return


@app.cell
def _(df_model):
    _ols = smf.ols("is_canceled ~ C(segment)", data=df_model).fit()
    _a = sm.stats.anova_lm(_ols, typ=1)
    anova_table = pd.DataFrame(
        {
            "SS": _a["sum_sq"],
            "df": _a["df"],
            "MS": _a["mean_sq"],
            "F": _a["F"],
            "p_value": _a["PR(>F)"],
        }
    )
    anova_table.index = ["Entre grupos (segment)", "Dentro de grupos (residual)"]
    anova_table.loc["Total"] = [
        anova_table["SS"].sum(),
        anova_table["df"].sum(),
        np.nan,
        np.nan,
        np.nan,
    ]

    _groups = [g["is_canceled"].values for _, g in df_model.groupby("segment")]
    _res = _ols.resid
    _sw = stats.shapiro(_res)
    _lev = stats.levene(
        *[_res[df_model["segment"] == s] for s in sorted(df_model["segment"].unique())],
        center="median",
    )
    _welch = anova_oneway(df_model["is_canceled"], df_model["segment"], use_var="unequal")
    _kw = stats.kruskal(*_groups)
    _hc3 = sm.stats.anova_lm(_ols, typ=2, robust="hc3")
    anova_robustness = pd.DataFrame(
        [
            {"prueba": "Shapiro-Wilk (H0: residuos normales)", "estadistico": _sw.statistic, "gl": "-", "p_value": _sw.pvalue},
            {"prueba": "Levene sobre residuos (H0: varianzas iguales)", "estadistico": _lev.statistic, "gl": "4, 378", "p_value": _lev.pvalue},
            {"prueba": "ANOVA con errores robustos HC3", "estadistico": _hc3.loc["C(segment)", "F"], "gl": "4, 378", "p_value": _hc3.loc["C(segment)", "PR(>F)"]},
            {"prueba": "ANOVA de Welch", "estadistico": _welch.statistic, "gl": f"4, {_welch.df_denom:.1f}", "p_value": _welch.pvalue},
            {"prueba": "Kruskal-Wallis", "estadistico": _kw.statistic, "gl": "4", "p_value": _kw.pvalue},
        ]
    ).round(4)

    mo.vstack(
        [
            mo.md("**Tabla ANOVA de una via**"),
            anova_table.round(4),
            mo.md("**Pruebas complementarias**"),
            anova_robustness,
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.** Con $\alpha = 0.05$ se **rechaza** $H_0$ ($F_{4,378} = 3.07$, $p = 0.016$): al menos un segmento tiene una tasa de cancelacion distinta. Sobre los residuos del modelo, Shapiro-Wilk rechaza normalidad y Levene rechaza la igualdad de varianzas ($p = 0.007$). Ambas cosas eran de esperarse con una respuesta binaria: los residuos solo toman dos valores por grupo y la varianza de cada grupo es $p_k(1-p_k)$ (el segmento Direct, con tasa baja, tiene menor varianza). Por eso aplicamos correcciones: el ANOVA de Welch (no supone varianzas iguales, $p = 0.004$), el ANOVA con errores robustos HC3 ($p = 0.003$) y Kruskal-Wallis (no parametrica, $p = 0.017$). Las tres mantienen la conclusion. La falta de normalidad pesa poco con $n = 383$, porque el TCL hace que la F basada en medias sea aproximadamente valida. Como $H_0$ se rechaza, aplicamos una prueba post-hoc.

    ### Post-hoc: Tukey HSD

    Tukey compara todos los pares de segmentos (10 comparaciones) controlando la tasa de error por familia en 5%.
    """)
    return


@app.cell
def _(df_model):
    _tk = pairwise_tukeyhsd(df_model["is_canceled"], df_model["segment"], alpha=0.05)
    tukey_table = pd.DataFrame(
        _tk.summary().data[1:], columns=_tk.summary().data[0]
    ).rename(
        columns={
            "meandiff": "diferencia_tasas",
            "p-adj": "p_ajustado",
            "lower": "IC95_inf",
            "upper": "IC95_sup",
            "reject": "difieren",
        }
    )
    tukey_table.sort_values("p_ajustado")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion del post-hoc.** Solo dos pares difieren significativamente:

    - **Direct vs Groups**: las reservas de grupo cancelan 34.6 puntos porcentuales mas que las directas ($p = 0.007$; 50% vs 15%).
    - **Direct vs Online TA**: las reservas por agencia en linea cancelan 23.7 puntos mas que las directas ($p = 0.041$; 39% vs 15%).

    Los demas pares no difieren significativamente. En resumen, el segmento **Direct** (el cliente que reserva directamente con el hotel) es el que se distingue por su baja tasa de cancelacion.

    ## ANCOVA: ajuste por la anticipacion de la reserva

    La tabla de resumen muestra que los segmentos difieren mucho en **anticipacion**: los grupos y las agencias offline reservan con meses de anticipacion, mientras que el cliente directo reserva con pocos dias. Como `lead_time` es el predictor mas fuerte de la cancelacion (ver modelo principal), la diferencia entre segmentos podria deberse a la anticipacion y no al segmento en si. Para separar ambos efectos incorporamos `log_lead_time` como **covariable**:

    $$
    \text{is\_canceled}_{ik} = \mu + \tau_k + \beta\,(\text{log\_lead\_time}_{ik} - \overline{\text{log\_lead\_time}}) + \varepsilon_{ik}
    $$

    **Justificacion de la covariable:** esta fuertemente relacionada con la variable dependiente y se conoce en el momento de la reserva, antes de que ocurra (o no) la cancelacion.

    Antes de interpretar el ANCOVA verificamos sus supuestos propios:

    - **Supuesto 6: la covariable es independiente del factor** ($H_0$: la media de `log_lead_time` es igual en todos los segmentos). Si se rechaza, el ajuste puede ser engañoso: las medias ajustadas comparan a los segmentos en un nivel de anticipacion que algunos casi nunca tienen.
    - **Supuesto 7: homogeneidad de pendientes** ($H_0$: la pendiente de `log_lead_time` es la misma en todos los segmentos), con la prueba F de modelos anidados (aditivo vs con interaccion).
    - **Supuestos 2 y 4 sobre los residuos**: normalidad (Shapiro-Wilk) y homocedasticidad (Levene por grupo).
    """)
    return


@app.cell
def _(df_model):
    def _anova_table(model):
        _a = sm.stats.anova_lm(model, typ=2)
        _a["MS"] = _a["sum_sq"] / _a["df"]
        return _a.rename(columns={"sum_sq": "SS", "PR(>F)": "p_value"})[
            ["SS", "df", "MS", "F", "p_value"]
        ].round(4)

    # Supuesto 6: covariable independiente del factor
    _cov = _anova_table(smf.ols("log_lead_time ~ C(segment)", data=df_model).fit())

    # Supuesto 7: homogeneidad de pendientes (F de modelos anidados: aditivo vs interaccion)
    ancova_model = smf.ols("is_canceled ~ C(segment) + log_lead_time", data=df_model).fit()
    ancova_interaction = smf.ols("is_canceled ~ C(segment) * log_lead_time", data=df_model).fit()
    _comp = sm.stats.anova_lm(ancova_model, ancova_interaction)
    _slopes = pd.DataFrame(
        {
            "F": [_comp.loc[1, "F"]],
            "gl_num": [int(_comp.loc[1, "df_diff"])],
            "gl_den": [int(ancova_interaction.df_resid)],
            "p_value": [_comp.loc[1, "Pr(>F)"]],
        }
    ).round(4)

    # Tabla ANCOVA (SS tipo II) y version con errores robustos HC3
    ancova_table = _anova_table(ancova_model)
    _robust = sm.stats.anova_lm(ancova_model, typ=2, robust="hc3").rename(
        columns={"PR(>F)": "p_value"}
    )[["F", "df", "p_value"]].round(4)

    # Supuestos 2 y 4 sobre los residuos del ANCOVA
    _res = ancova_model.resid
    _sw = stats.shapiro(_res)
    _lev = stats.levene(
        *[_res[df_model["segment"] == s] for s in sorted(df_model["segment"].unique())],
        center="median",
    )
    _resid_tests = pd.DataFrame(
        [
            {"prueba": "Shapiro-Wilk (residuos)", "estadistico": _sw.statistic, "p_value": _sw.pvalue},
            {"prueba": "Levene (residuos por segmento)", "estadistico": _lev.statistic, "p_value": _lev.pvalue},
        ]
    )

    mo.vstack(
        [
            mo.md("**Supuesto 6:** ANOVA de `log_lead_time` por segmento"),
            _cov,
            mo.md("**Supuesto 7:** F de modelos anidados (aditivo vs con interaccion `segment × log_lead_time`)"),
            _slopes,
            mo.md("**Tabla ANCOVA** (suma de cuadrados tipo II)"),
            ancova_table,
            mo.md("**Supuestos 2 y 4:** residuos del ANCOVA"),
            _resid_tests,
            mo.md("**Tabla ANCOVA con errores robustos HC3** (correccion por heterocedasticidad)"),
            _robust,
        ]
    )
    return ancova_interaction, ancova_model


@app.cell
def _(ancova_interaction, df_model):
    # Pendiente de log_lead_time en cada segmento con su IC 95% (modelo con interaccion)
    _idx = {n: i for i, n in enumerate(ancova_interaction.params.index)}
    _rows = []
    for _s in sorted(df_model["segment"].unique()):
        _L = np.zeros(len(ancova_interaction.params))
        _L[_idx["log_lead_time"]] = 1
        _name = f"C(segment)[T.{_s}]:log_lead_time"
        if _name in _idx:  # el nivel de referencia no tiene termino de interaccion
            _L[_idx[_name]] = 1
        _tt = ancova_interaction.t_test(_L)
        _lo, _hi = np.asarray(_tt.conf_int()).ravel()
        _rows.append(
            {
                "segment": _s,
                "pendiente": float(np.squeeze(_tt.effect)),
                "EE": float(np.squeeze(_tt.sd)),
                "IC95_inf": _lo,
                "IC95_sup": _hi,
            }
        )
    slopes_by_segment = pd.DataFrame(_rows).round(4)
    mo.vstack([mo.md("**Pendiente de `log_lead_time` por segmento**"), slopes_by_segment])
    return


@app.cell
def _(PALETTE, ancova_model, df_model):
    # Medias crudas vs medias ajustadas (evaluadas en la media global de la covariable)
    _segs = sorted(df_model["segment"].unique())
    _new = pd.DataFrame({"segment": _segs, "log_lead_time": df_model["log_lead_time"].mean()})
    _pred = ancova_model.get_prediction(_new).summary_frame(alpha=0.05)

    _raw = df_model.groupby("segment")["is_canceled"].agg(["mean", "std", "size"]).loc[_segs]
    _se = _raw["std"] / np.sqrt(_raw["size"])

    adjusted_means = pd.concat(
        [
            pd.DataFrame(
                {
                    "segment": _segs,
                    "tipo": "cruda (ANOVA)",
                    "tasa": _raw["mean"].values,
                    "IC_inf": (_raw["mean"] - 1.96 * _se).values,
                    "IC_sup": (_raw["mean"] + 1.96 * _se).values,
                }
            ),
            pd.DataFrame(
                {
                    "segment": _segs,
                    "tipo": "ajustada por lead_time (ANCOVA)",
                    "tasa": _pred["mean"].values,
                    "IC_inf": _pred["mean_ci_lower"].values,
                    "IC_sup": _pred["mean_ci_upper"].values,
                }
            ),
        ]
    )

    _base = alt.Chart(adjusted_means).encode(
        y=alt.Y("segment:N", title=None),
        yOffset="tipo:N",
        color=alt.Color(
            "tipo:N",
            title=None,
            scale=alt.Scale(range=[PALETTE[0], PALETTE[1]]),
            legend=alt.Legend(orient="bottom"),
        ),
    )
    _chart = (
        _base.mark_rule(strokeWidth=3).encode(
            x=alt.X("IC_inf:Q", title="tasa de cancelacion (IC 95%)"), x2="IC_sup:Q"
        )
        + _base.mark_point(filled=True, size=100, opacity=1).encode(x="tasa:Q")
    ).properties(
        title="Tasa de cancelacion por segmento: cruda vs ajustada", width=550, height=300
    )
    mo.vstack([adjusted_means.round(3), _chart])
    return


@app.cell
def _(ancova_model, df_model):
    # Post-hoc del ANCOVA: Tukey-Kramer sobre medias ajustadas con el CME y los gl del ANCOVA
    # (no se usa Tukey "ingenuo" sobre los valores ajustados porque subestima el error)
    _adj = df_model["is_canceled"] - (
        df_model["log_lead_time"] - df_model["log_lead_time"].mean()
    ) * ancova_model.params["log_lead_time"]
    _medias = _adj.groupby(df_model["segment"]).mean()
    _n = df_model.groupby("segment").size()
    _k, _gl, _cme = len(_medias), ancova_model.df_resid, ancova_model.mse_resid

    _rows = []
    for _a, _b in itertools.combinations(_medias.index, 2):
        _dif = _medias[_b] - _medias[_a]
        _se = np.sqrt(_cme / 2 * (1 / _n[_a] + 1 / _n[_b]))
        _rows.append(
            {
                "grupo1": _a,
                "grupo2": _b,
                "diferencia_ajustada": _dif,
                "p_tukey_kramer": studentized_range.sf(abs(_dif) / _se, _k, _gl),
            }
        )
    tukey_kramer_ancova = pd.DataFrame(_rows)
    tukey_kramer_ancova["difieren"] = tukey_kramer_ancova["p_tukey_kramer"] < 0.05
    mo.vstack(
        [
            mo.md("**Post-hoc del ANCOVA: Tukey-Kramer con el error del ANCOVA**"),
            tukey_kramer_ancova.round(4).sort_values("p_tukey_kramer"),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion del ANCOVA.**

    1. **Supuesto 6 (covariable independiente del factor): NO se cumple** ($F_{4,378} = 21.4$, $p \approx 10^{-16}$). Groups y Offline TA/TO reservan con mucha mas anticipacion que Direct y Corporate. Siguiendo el criterio del curso, revisamos si la covariable es **consecuencia** del factor (en cuyo caso no se deberia ajustar por ella): no lo es. La anticipacion es una decision del cliente que se toma al mismo tiempo que se elige el canal, y no es un efecto que el segmento "produzca". Por eso el ajuste tiene sentido, pero con una precaucion: las medias ajustadas comparan a los segmentos en la anticipacion promedio, un nivel que algunos segmentos casi no tienen (p. ej. Corporate no tiene reservas de anticipacion larga). Por ello, abajo confirmamos la conclusion con la alternativa que propone el curso: **usar la covariable como bloque**.
    2. **Supuesto 7 (homogeneidad de pendientes): se cumple.** La F de modelos anidados no es significativa ($p = 0.35$). Las pendientes por segmento son todas positivas y sus intervalos se traslapan; Groups tiene la pendiente mas pronunciada, pero la diferencia no es significativa. Con $n = 383$ la prueba tiene potencia razonable, por lo que el modelo aditivo (pendiente comun) es adecuado.
    3. **Supuestos 2 y 4 (residuos): no se cumplen**, como era de esperarse con una respuesta 0/1: los residuos solo pueden tomar dos valores por cada nivel de $\hat y$ (Shapiro-Wilk rechaza normalidad) y su varianza depende de la tasa de cada grupo (Levene rechaza homocedasticidad). La correccion que se aplica son los **errores robustos HC3**: la conclusion no cambia (segmento $p = 0.24$, anticipacion $p < 0.001$). La normalidad de los residuos pesa poco aqui, porque con $n = 383$ el TCL garantiza que los estadisticos F basados en medias son aproximadamente validos.
    4. **Cambio de significancia:** al ajustar por `log_lead_time`, el efecto del segmento **deja de ser significativo** ($F_{4,377} = 1.28$, $p = 0.28$, frente a $p = 0.016$ en el ANOVA), mientras que la anticipacion es altamente significativa ($F_{1,377} = 39.3$, $p < 0.001$).
    5. **Medias ajustadas y post-hoc:** comparando segmentos con la **misma** anticipacion, Groups baja de 50% a 41% y Direct sube de 15% a 29%. Con Tukey-Kramer sobre las medias ajustadas, **ningun par difiere significativamente**: Direct vs Groups pasa de $p = 0.007$ (ANOVA) a $p = 0.70$ (ANCOVA).

    ### Alternativa: la anticipacion como bloque (ANOVA de dos factores)

    Como el supuesto 6 no se cumple, seguimos la alternativa del curso: en vez de ajustar por una pendiente, dividimos `lead_time` en **terciles** (bloques de anticipacion corta, media y larga) y hacemos un ANOVA de dos factores `segment + bloque`. Asi, cada segmento se compara contra los otros **dentro del mismo rango de anticipacion**, sin extrapolar.
    """)
    return


@app.cell
def _(df_model):
    _d = df_model.assign(lead_time=np.expm1(df_model["log_lead_time"]))
    _d["bloque_anticipacion"] = pd.qcut(
        _d["lead_time"], 3, labels=["corta (<=35 d)", "media (36-119 d)", "larga (>=120 d)"]
    )

    def _anova_table(model, robust=None):
        _a = sm.stats.anova_lm(model, typ=2, robust=robust)
        _a["MS"] = _a["sum_sq"] / _a["df"]
        return _a.rename(columns={"sum_sq": "SS", "PR(>F)": "p_value"})[
            ["SS", "df", "MS", "F", "p_value"]
        ].round(4)

    _additive = smf.ols("is_canceled ~ C(segment) + C(bloque_anticipacion)", data=_d).fit()
    _inter = smf.ols("is_canceled ~ C(segment) * C(bloque_anticipacion)", data=_d).fit()

    mo.vstack(
        [
            mo.md("**Tamaño de cada celda (segmento × bloque)**"),
            pd.crosstab(_d["segment"], _d["bloque_anticipacion"]),
            mo.md("**Tasa de cancelacion por celda**"),
            _d.pivot_table(
                index="segment",
                columns="bloque_anticipacion",
                values="is_canceled",
                aggfunc="mean",
                observed=False,
            ).round(3),
            mo.md("**ANOVA de dos factores (aditivo)**"),
            _anova_table(_additive),
            mo.md("**ANOVA de dos factores (aditivo) con errores robustos HC3**"),
            _anova_table(_additive, robust="hc3"),
            mo.md("**ANOVA de dos factores con interaccion** (ver advertencia)"),
            _anova_table(_inter),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion del analisis por bloques.** Dentro de cada rango de anticipacion, el segmento **no** tiene un efecto significativo ($F_{4,376} = 1.69$, $p = 0.15$; con HC3 $p = 0.07$), mientras que el bloque de anticipacion es altamente significativo ($p < 0.001$). La tabla de tasas lo muestra con claridad: en **todos** los segmentos la cancelacion crece con la anticipacion (p. ej. Groups: 17% → 33% → 69%; Direct: 8% → 14% → 43%).

    **Advertencia sobre la interaccion.** El modelo con interaccion sale significativo, pero no es confiable: Corporate no tiene ninguna reserva de anticipacion larga (celda vacia) y varias celdas tienen 6-7 observaciones, por lo que esa prueba compara medias estimadas con muy poca informacion. No la usamos para concluir.

    **Conclusion de negocio.** Las tres aproximaciones (ANCOVA con pendiente comun, ANCOVA robusto HC3 y ANOVA con bloques) coinciden: el segmento por si mismo **no** explica la cancelacion. Los grupos cancelan mas porque reservan con mucha anticipacion, y los clientes directos cancelan menos porque reservan con poca. Es la anticipacion, no el canal, lo que determina el riesgo. Esto es consistente con el modelo logistico, donde `segment` tampoco fue significativo una vez controladas las demas variables ($p = 0.087$).

    **Nota sobre el modelo lineal con respuesta binaria.** El ANOVA/ANCOVA sobre una variable 0/1 equivale a un modelo lineal de probabilidad: es valido para comparar medias (tasas) gracias al TCL, pero sus residuos no son normales y son heterocedasticos por construccion. Por eso confirmamos los resultados con Welch, Kruskal-Wallis y errores robustos HC3, y la conclusion coincide con la del modelo logistico, que si respeta la naturaleza binaria de la respuesta.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Reduccion de dimensionalidad con PCA

    Aplicamos Analisis de Componentes Principales a las **12 variables numericas que se conocen al momento de reservar**. Se excluyen las que descartamos por _leakage_ o separacion (`booking_changes`, `required_car_parking_spaces`, `days_in_waiting_list`) y `arrival_date_year`.

    **Preprocesamiento:**

    1. A las variables muy asimetricas (`lead_time`, `adr`, noches, niños, bebes, historial de reservas y solicitudes especiales) se les aplica $\log(1+x)$. El PCA se basa en varianzas y covarianzas, que son muy sensibles a valores extremos: sin la transformacion, una sola reserva con 20 cancelaciones previas dominaria un componente completo.
    2. Todas las variables se **estandarizan** ($z = (x - \bar x)/s$). Sin estandarizar, las variables con mayor escala (p. ej. `lead_time`, en cientos de dias) dominarian los componentes solo por sus unidades. Con datos estandarizados, el PCA equivale a diagonalizar la **matriz de correlaciones**, y cada eigenvalor se mide en "numero de variables": un eigenvalor de 2 significa que el componente resume tanta varianza como 2 variables originales.
    """)
    return


@app.cell
def _(df_sample):
    PCA_LOG_VARS = [
        "lead_time",
        "adr",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "children",
        "babies",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "total_of_special_requests",
    ]
    PCA_RAW_VARS = ["adults", "arrival_date_week_number", "arrival_date_day_of_month"]

    def build_pca_matrix(d):
        """Matriz de entrada del PCA (antes de estandarizar)."""
        return pd.concat(
            [
                np.log1p(d[PCA_LOG_VARS].fillna(0).clip(lower=0)).add_prefix("log_"),
                d[PCA_RAW_VARS],
            ],
            axis=1,
        )

    pca_input = build_pca_matrix(df_sample)
    pca_scaler = StandardScaler().fit(pca_input)
    pca_model = PCA().fit(pca_scaler.transform(pca_input))

    PC_NAMES = [f"PC{i + 1}" for i in range(pca_model.n_components_)]
    pca_variance = pd.DataFrame(
        {
            "eigenvalor": pca_model.explained_variance_,
            "varianza_explicada": pca_model.explained_variance_ratio_,
            "varianza_acumulada": np.cumsum(pca_model.explained_variance_ratio_),
        },
        index=PC_NAMES,
    )
    pca_variance["kaiser (eig > 1)"] = pca_variance["eigenvalor"] > 1

    N_PC_KAISER = int((pca_variance["eigenvalor"] > 1).sum())
    N_PC_80 = int(np.argmax(pca_variance["varianza_acumulada"].values >= 0.80) + 1)

    pca_variance.round(4)
    return (
        N_PC_80,
        N_PC_KAISER,
        PC_NAMES,
        pca_input,
        pca_model,
        pca_scaler,
        pca_variance,
    )


@app.cell
def _(BLK, N_PC_80, N_PC_KAISER, PALETTE, pca_variance):
    _d = pca_variance.reset_index(names="componente")
    _d["orden"] = range(1, len(_d) + 1)

    _scree = (
        alt.Chart(_d)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=70), color=PALETTE[0])
        .encode(
            x=alt.X("orden:O", title="componente principal"),
            y=alt.Y("eigenvalor:Q", title="eigenvalor"),
            tooltip=["componente", alt.Tooltip("eigenvalor:Q", format=".3f")],
        )
    )
    _kaiser = (
        alt.Chart(pd.DataFrame({"y": [1]}))
        .mark_rule(strokeDash=[6, 3], color=BLK, opacity=1)
        .encode(y="y:Q")
    )
    _cum = (
        alt.Chart(_d)
        .mark_bar(color=PALETTE[3])
        .encode(
            x=alt.X("orden:O", title="componente principal"),
            y=alt.Y("varianza_acumulada:Q", title="varianza acumulada", axis=alt.Axis(format="%")),
            tooltip=["componente", alt.Tooltip("varianza_acumulada:Q", format=".1%")],
        )
    )
    _p80 = (
        alt.Chart(pd.DataFrame({"y": [0.8]}))
        .mark_rule(strokeDash=[6, 3], color=PALETTE[-1], opacity=1)
        .encode(y="y:Q")
    )
    mo.vstack(
        [
            (
                (_scree + _kaiser).properties(
                    title=f"Grafico de codo (criterio de Kaiser: {N_PC_KAISER} componentes)",
                    width=330,
                    height=280,
                )
                | (_cum + _p80).properties(
                    title=f"Varianza acumulada (80% con {N_PC_80} componentes)",
                    width=330,
                    height=280,
                )
            )
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **¿Cuantos componentes retener?**

    - **Criterio de Kaiser** (eigenvalor $> 1$, es decir, que el componente explique mas que una variable original): retiene **5 componentes**, que explican el **57.7%** de la varianza.
    - **Criterio de varianza acumulada $\geq 80\%$**: requiere **9 de 12** componentes (85.1%).
    - **Grafico de codo:** no hay un codo claro; los eigenvalores bajan de forma gradual despues de PC1.

    Esta discrepancia entre criterios es un **hallazgo en si mismo**: el PCA comprime bien cuando las variables estan muy correlacionadas (pocas componentes concentran casi toda la varianza). Aqui las correlaciones son debiles (todos los VIF $< 2$), cada variable aporta informacion propia, y reducir de 12 a 9 dimensiones para llegar al 80% casi no simplifica nada. Retenemos **5 componentes por el criterio de Kaiser** como modelo principal reducido, y reportamos tambien el de 9 componentes como referencia.

    ### Cargas (_loadings_)

    Las cargas mostradas son las **correlaciones** entre cada variable original y cada componente ($\text{carga}_{jk} = v_{jk}\sqrt{\lambda_k}$), por lo que se leen en la misma escala que un coeficiente de correlacion. El signo de un componente es arbitrario: lo que importa es que variables con el mismo signo se mueven juntas.
    """)
    return


@app.cell
def _(N_PC_KAISER, PC_NAMES, cmap, pca_input, pca_model):
    pca_loadings = pd.DataFrame(
        pca_model.components_.T * np.sqrt(pca_model.explained_variance_),
        index=pca_input.columns,
        columns=PC_NAMES,
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    sns.heatmap(
        pca_loadings.iloc[:, :N_PC_KAISER],
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        ax=_axes[0],
    )
    _axes[0].set_title("Cargas de los componentes retenidos (Kaiser)")

    _top = pca_loadings[["PC1", "PC2"]].sort_values("PC1")
    _top.plot.barh(ax=_axes[1], color=["#2F6FBF", "#D95F2B"])
    _axes[1].axvline(0, color="#2B2B2B", lw=1)
    _axes[1].set_title("Cargas de PC1 y PC2")
    _axes[1].set_xlabel("carga (correlacion con el componente)")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion de los dos primeros componentes** (consideramos relevantes las cargas con $|carga| \gtrsim 0.4$):

    - **PC1 (17.7%) — "tamaño de la estancia de un cliente nuevo".** Cargas positivas altas en `adults` (0.65), noches entre semana (0.61), `adr` (0.50), noches de fin de semana (0.50) y anticipacion (0.46); carga negativa en reservas previas no canceladas (−0.53) y cancelaciones previas (−0.34). Valores altos de PC1 son reservas de mas personas, mas noches y mas caras, hechas por clientes **sin historial** con el hotel; valores bajos son estancias cortas de clientes recurrentes.
    - **PC2 (11.8%) — "perfil de riesgo de cancelacion".** Cargas positivas en anticipacion (0.61), cancelaciones previas (0.56) y semana del año (0.54, temporada de verano-otoño); negativas en solicitudes especiales (−0.47) y niños (−0.34). Es un componente que, **sin haber visto la variable `is_canceled`**, reconstruye casi exactamente el perfil que el modelo logistico identifico como de alto riesgo: reservar con mucha anticipacion, haber cancelado antes y no hacer solicitudes especiales.

    En la grafica siguiente se ve que las reservas canceladas tienden a ubicarse en valores altos de PC2, mientras que PC1 no las separa.
    """)
    return


@app.cell
def _(PALETTE, df_model, pca_input, pca_model, pca_scaler):
    _scores = pd.DataFrame(
        pca_model.transform(pca_scaler.transform(pca_input))[:, :2],
        columns=["PC1", "PC2"],
        index=pca_input.index,
    )
    _scores["is_canceled"] = df_model["is_canceled"].astype(str)
    (
        alt.Chart(_scores)
        .mark_point(filled=True, size=40, opacity=0.6)
        .encode(
            x=alt.X("PC1:Q", title="PC1: tamaño de la estancia (cliente nuevo)"),
            y=alt.Y("PC2:Q", title="PC2: perfil de riesgo"),
            color=alt.Color(
                "is_canceled:N", title="is_canceled", scale=alt.Scale(range=[PALETTE[0], PALETTE[-1]])
            ),
        )
        .properties(title="Reservas en el plano PC1-PC2", width=550, height=400)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modelo logistico con componentes principales

    Re-estimamos el modelo principal usando los componentes como predictores. Como el PCA solo resume las variables **numericas**, comparamos tres versiones contra el modelo original:

    - **PCA (Kaiser):** solo los 5 componentes retenidos.
    - **PCA (80%):** los 9 componentes que alcanzan el 80% de varianza.
    - **PCA (Kaiser) + binarias:** los 5 componentes mas las dos variables binarias del modelo original que no son numericas y por lo tanto no entran al PCA (`transient`, `is_repeated_guest`). Es la comparacion mas justa, porque ambos modelos usan la misma informacion de origen.

    Una ventaja de los componentes como predictores es que son **ortogonales** (correlacion cero entre si), asi que la multicolinealidad desaparece por construccion ($VIF = 1$).
    """)
    return


@app.cell
def _(
    N_PC_80,
    N_PC_KAISER,
    PC_NAMES,
    df_model,
    logit_final,
    pca_input,
    pca_model,
    pca_scaler,
):
    df_pca = pd.concat(
        [
            df_model,
            pd.DataFrame(
                pca_model.transform(pca_scaler.transform(pca_input)),
                columns=PC_NAMES,
                index=pca_input.index,
            ),
        ],
        axis=1,
    )

    def _fit(terms):
        return smf.logit("is_canceled ~ " + " + ".join(terms), data=df_pca).fit(disp=0)

    PCS_KAISER = PC_NAMES[:N_PC_KAISER]
    logit_pca = _fit(PCS_KAISER)
    logit_pca_80 = _fit(PC_NAMES[:N_PC_80])
    logit_pca_bin = _fit(PCS_KAISER + ["transient", "is_repeated_guest"])

    def _row(name, m):
        return {
            "modelo": name,
            "predictores": int(m.df_model),
            "devianza": -2 * m.llf,
            "AIC": m.aic,
            "BIC": m.bic,
            "pseudo_R2": m.prsquared,
            "LRT_vs_nulo_chi2": m.llr,
            "p_value_LRT": m.llr_pvalue,
            "coefs_significativos": int((m.pvalues.drop("Intercept") < 0.05).sum()),
        }

    pca_comparison = pd.DataFrame(
        [
            _row("original (5 variables)", logit_final),
            _row(f"PCA Kaiser ({N_PC_KAISER} PCs)", logit_pca),
            _row(f"PCA 80% ({N_PC_80} PCs)", logit_pca_80),
            _row(f"PCA Kaiser + binarias", logit_pca_bin),
        ]
    )

    _coefs = pd.DataFrame(
        {
            "coef": logit_pca.params,
            "std_err": logit_pca.bse,
            "z": logit_pca.tvalues,
            "p_value": logit_pca.pvalues,
            "IC95_inf": logit_pca.conf_int()[0],
            "IC95_sup": logit_pca.conf_int()[1],
            "odds_ratio": np.exp(logit_pca.params),
        }
    ).round(4)

    mo.vstack(
        [
            mo.md(f"**Coeficientes del modelo con {N_PC_KAISER} componentes (Kaiser)**"),
            _coefs,
            mo.md("**Comparacion de modelos**"),
            pca_comparison.round(4),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacion.**

    - **Significancia global:** los tres modelos con componentes son significativos frente al nulo (p. ej. PCA Kaiser: $\chi^2_5 = 69.2$, $p \approx 10^{-13}$): los componentes si contienen informacion sobre la cancelacion.
    - **Significancia individual:** en el modelo de 5 componentes, **PC2** (perfil de riesgo, $\hat\beta = 0.50$, $p < 0.001$), **PC4** y **PC5** son significativos, mientras que **PC1** (tamaño de la estancia, $p = 0.23$) y PC3 no lo son. Esto confirma que **el componente que mas varianza explica no es necesariamente el mas util para predecir**: el PCA ordena los componentes por varianza de las $X$, sin mirar la $Y$.
    - **Comparacion con el modelo original:** el modelo original es claramente superior en todas las metricas, con el mismo numero de parametros (5): AIC 402.1 vs 447.9, BIC 425.7 vs 471.6, pseudo-$R^2$ 0.228 vs 0.137. Incluso agregando las binarias (`transient`, `is_repeated_guest`) o usando 9 componentes, el modelo con PCA no alcanza al original.

    **¿Por que pierde el PCA?** Por dos razones: (1) el PCA es **no supervisado**: elige direcciones de maxima varianza de las $X$, no de maxima relacion con la cancelacion, y mezcla en cada componente variables relevantes (anticipacion, cancelaciones previas) con variables irrelevantes (dia del mes, `adr`), diluyendo su efecto; (2) el PCA resuelve un problema que **no teniamos**: la multicolinealidad era baja (VIF $< 2$), de modo que no habia redundancia que eliminar.

    **Conclusion:** el PCA es util como herramienta **descriptiva** (PC2 resume con claridad el perfil de riesgo), pero como base del modelo predictivo es preferible el modelo original: es mas preciso, mas parsimonioso y sus coeficientes se interpretan directamente en terminos de negocio ("hacer una solicitud especial reduce los momios 66%"), algo que no es posible con una combinacion lineal de 12 variables.
    """)
    return


if __name__ == "__main__":
    app.run()
