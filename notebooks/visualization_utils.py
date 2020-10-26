import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import seaborn as sns
from IPython.core.display import HTML, display
from IPython.display import Image
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter

STATIC = None

def load_notebook_config(width=True, static=False):
    global STATIC
    STATIC = static
    pd.options.display.max_columns = 0
    plt.rcParams.update({
        "font.family": ["serif"],
        "font.sans-serif": ["Roboto"],
        "font.size": 9,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        'figure.figsize': (15.0, 4.0),
        'axes.grid': False,
        'axes.spines.left': True,
        'axes.spines.right': True,
        'axes.spines.top': True,
        'axes.spines.bottom': True,
    })
    if width and not static:
        display(HTML("<style>.container { width:90% !important; }</style>"))
    pd.options.display.max_rows = 200
    pd.options.display.min_rows = 200
    pd.options.plotting.backend = "plotly"
    pio.renderers.default = 'notebook'


def show_plotly_figure(fig):
    if STATIC:
        fig.update_layout(font_size=18)
        fig.update_layout(margin=dict(l=2, r=5, t=2, b=2),)
        as_bytes = fig.to_image(format="png", width=900, height=600)
        display(Image(as_bytes))
    else:
        fig.show()


def is_in_static_mode():
    return STATIC is True


def tweak_ticks(ax, axis='x', rotation=0, alternate=False, ha='center', format_=None,
                formatter_function=None):
    get_ticklabels = getattr(ax, f'get_{axis}ticklabels')
    set_ticklabels = getattr(ax, f'set_{axis}ticklabels')
    the_axis = getattr(ax, f'{axis}axis')

    if formatter_function:
        labels = get_ticklabels()
        labels_2 = [formatter_function(x.get_text()) for x in labels]
        set_ticklabels(labels_2)

    labels = get_ticklabels()
    if alternate:
        locs = the_axis.get_major_locator().locs
        major_labels = [x.get_text() for x in labels[::2]]
        minor_labels = [x.get_text() for x in labels[1::2]]

        the_axis.set_major_locator(FixedLocator(locs[::2]))
        the_axis.set_major_formatter(FixedFormatter(major_labels))
        the_axis.set_minor_formatter(FixedFormatter(minor_labels))
        the_axis.set_minor_locator(FixedLocator(locs[1::2]))

        ax.tick_params(which='major', pad=20, axis=axis)
        set_ticklabels(major_labels, rotation=rotation, ha=ha, minor=False)
        set_ticklabels(minor_labels, rotation=rotation, ha=ha, minor=True)
    else:
        set_ticklabels(labels, rotation=rotation, ha=ha)

    if format_ is not None:
        the_axis.set_major_formatter(FuncFormatter(lambda y, _: format_.format(y)))


def show_feature_importance(feature_names, predictor, limit=20, title=None):
    feature_names = np.asarray(feature_names)

    feature_importances = [predictor.feature_importances_]
    feature_importances = np.asarray(feature_importances)
    feature_importances_mean = np.mean(feature_importances, axis=0)
    indices = np.argsort(feature_importances_mean)[::-1]
    features_df = pd.DataFrame(feature_importances, columns=feature_names)

    cols = feature_names[indices][:limit].tolist()
    fig = px.bar(features_df[cols[::-1]].T, orientation='h')

    if not title:
        title = 'Top {} important features.'.format(limit)

    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=400, r=5, t=40, b=5)

    )
    fig.update_yaxes(
        title_text='',
    )
    fig.update_xaxes(
        title_text='',
        showticklabels=False
    )


    show_plotly_figure(fig)


def distplot(s, *args, remove_outliers=(0.05, 0.95), **kwargs):
    '''Same seaborn functionality but with the ability of remove outlierts based on quantiles'''
    to_plot = s
    if remove_outliers:
        min_q, max_q = remove_outliers
        to_plot = s[(s.quantile(min_q) < s) & (s < s.quantile(max_q))]
    return sns.distplot(to_plot, *args, **kwargs)
