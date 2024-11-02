from os.path import exists
import pandas as pd
import plotly.graph_objects as go
import warnings
import plotly.figure_factory as ff
import numpy as np
warnings.filterwarnings('ignore')

# TODO
# ток               min max ia ib ic 
# напряжение        min max ua ub uc 
# мощность          max pa pb pc, сумма максимумов 
# max ITHA	ITHB	ITHC	ITHAvg


# line_colors = ['yellow', 'green', 'red']
# arrow_colors = ['red', 'purple', 'brown']
line_colors = ['yellow', 'green', 'red', 'blue']
arrow_colors = ['black', 'black', 'black', 'black']

def get_df(path: str, rows: int = 0) -> pd.DataFrame:
    if not exists(path):
         print("=== Incorrect path ===")
         exit()
    if rows:
        df = pd.read_csv(path, 
                         skiprows=2, 
                         nrows=rows, 
                         low_memory=True,
                         parse_dates=[[0,1]], 
                         #  index_col=0,
                         )
    else:
        df = pd.read_csv(path, 
                         skiprows=2, 
                         low_memory=True,
                         parse_dates=[[0,1]], 
                         #  index_col=0,
                         )
    df = df.drop(labels='Unnamed: 80', axis=1)

    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # cols = ['Date_Time',
    #         'UA',
    #         'UB',
    #         'UC',
    #         'IA',
    #         'IB',
    #         'IC',
    #         'PA',
    #         'PB',
    #         'PC',
    #         'PSum',
    #         ]
    # df = df[cols]
    # print(df.head())
    df = df[(df['UA'] > 0.001) | (df['UB'] > 0.001) | (df['UC'] > 0.001)] 
    # print(df.head())
    # print(df[['PA', 'PB', 'PC']])
    df[['PA', 'PB', 'PC']] = df[['PA', 'PB', 'PC']].abs() 
    # print(df[['PA', 'PB', 'PC']])

    # df = df[df['UA'] > 0.001]  

    return df


def current_timeline(df: pd.DataFrame, min_max_arrows: bool):
    try: df = df[['Date_Time', 'IA', 'IB', 'IC']]
    except: print('Error - incorrect df in current_timeline()'); return

    keys_dict = ['IA', 'IB', 'IC']
    fig = go.Figure()
    for j, key in enumerate(keys_dict):
        fig.add_trace(go.Scatter(x=df['Date_Time'], y=df[key], 
                                    mode='lines',
                                    name=key, 
                                    line=dict(color=line_colors[j])
                                    ))
    title_font = dict(size=12, color='black')
    fig.update_layout(
        title='Current timeline Chart',
        xaxis=dict(title='Date and Time', title_font=title_font),
        yaxis=dict(title='Values', title_font=title_font),
        hovermode='closest',
    )

    if min_max_arrows:
        for j, key in enumerate(keys_dict):
            max_value_row = df.loc[df[key].idxmax()]
            min_value_row = df.loc[df[key].idxmin()]

            fig.add_annotation(
                text=f'<b>Max {key}</b>',
                x=max_value_row['Date_Time'],
                y=max_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )
            fig.add_annotation(
                text=f'<b>Min {key}</b>',
                x=min_value_row['Date_Time'],
                y=min_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )

    fig.show()
    return

def power_timeline(df: pd.DataFrame, min_max_arrows: bool):
    try: df = df[['Date_Time', 'PA', 'PB', 'PC']]
    except: print('Error - incorrect df in power_timeline()'); return

    df[['PA', 'PB', 'PC']] = df[['PA', 'PB', 'PC']].abs()
    keys_dict = ['PA', 'PB', 'PC']
    fig = go.Figure()
    for j, key in enumerate(keys_dict):
        fig.add_trace(go.Scatter(x=df['Date_Time'], y=df[key], 
                                    mode='lines',
                                    name=key, 
                                    line=dict(color=line_colors[j])
                                    ))
    title_font = dict(size=12, color='black')
    fig.update_layout(
        title='Power timeline Chart',
        xaxis=dict(title='Date and Time', title_font=title_font),
        yaxis=dict(title='Values', title_font=title_font),
        hovermode='closest',
    )

    if min_max_arrows:
        for j, key in enumerate(keys_dict):
            max_value_row = df.loc[df[key].idxmax()]
            min_value_row = df.loc[df[key].idxmin()]

            fig.add_annotation(
                text=f'<b>Max {key}</b>',
                x=max_value_row['Date_Time'],
                y=max_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )
            fig.add_annotation(
                text=f'<b>Min {key}</b>',
                x=min_value_row['Date_Time'],
                y=min_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )

    fig.show()
    return

def act_energ_timeline(df: pd.DataFrame, min_max_arrows: bool):
    try: df = df[['Date_Time', 'EPA', 'EPB', 'EPC', 'EPSum']]
    except: print('Error - incorrect df in act_energ_timeline()'); return

    keys_dict = ['EPA', 'EPB', 'EPC', 'EPSum']
    fig = go.Figure()
    for j, key in enumerate(keys_dict):
        fig.add_trace(go.Scatter(x=df['Date_Time'], y=df[key], 
                                    mode='lines',
                                    name=key, 
                                    line=dict(color=line_colors[j])
                                    ))
    title_font = dict(size=12, color='black')
    fig.update_layout(
        title='Power timeline Chart',
        xaxis=dict(title='Date and Time', title_font=title_font),
        yaxis=dict(title='Values', title_font=title_font),
        hovermode='closest',
    )

    if min_max_arrows:
        for j, key in enumerate(keys_dict):
            max_value_row = df.loc[df[key].idxmax()]
            min_value_row = df.loc[df[key].idxmin()]

            fig.add_annotation(
                text=f'<b>Max {key}</b>',
                x=max_value_row['Date_Time'],
                y=max_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )
            fig.add_annotation(
                text=f'<b>Min {key}</b>',
                x=min_value_row['Date_Time'],
                y=min_value_row[key],
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=arrow_colors[j],
                ax=20,
                ay=-40,
                name=key,
                font=dict(size=14, color=arrow_colors[j]),
            )

    fig.show()
    return

def voltage_gaussian_distribution(df: pd.DataFrame):
    try: df = df[['UA', 'UB', 'UC']]
    except: print('Error - incorrect df in power_timeline()'); return

    hist_data = [df['UA'], df['UB'], df['UC']]
    group_labels = ['UA', 'UB', 'UC']
    fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    fig.show()

def describe(df: pd.DataFrame):
    result = {}
    df = df.describe().T

    max_vals = {'IA':df['max']['IA'], 
                'IB':df['max']['IB'], 
                'IC':df['max']['IC'], 

                'UA':df['max']['UA'], 
                'UB':df['max']['UB'], 
                'UC':df['max']['UC'], 

                'PA':df['max']['PA'], 
                'PB':df['max']['PB'], 
                'PC':df['max']['PC'], 

                'ITHA':df['max']['ITHA'], 
                'ITHB':df['max']['ITHB'], 
                'ITHC':df['max']['ITHC'], 
                'ITHAvg':df['max']['ITHAvg'], 
                }
    min_vals = {'IA':df['min']['IA'], 
                'IB':df['min']['IB'], 
                'IC':df['min']['IC'], 

                'UA':df['min']['UA'], 
                'UB':df['min']['UB'], 
                'UC':df['min']['UC'], 

                }
    p_max_sum = df['max']['PA'] + df['max']['PB'] + df['max']['PC']
    result['max'] = max_vals
    result['min'] = min_vals
    return max_vals, min_vals, p_max_sum

