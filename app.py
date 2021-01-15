
import base64
import io


import dash_table


import sys
sys.path.append('C:/Users/fedir/dash-svm')
from Model_Gen import temp_name
from Comp_test import trials
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn import datasets
import utils.dash_reusable_components as drc


app = dash.Dash(__name__)
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == 'moons':
        return datasets.make_moons(
            n_samples=n_samples,
            noise=noise,
            random_state=0
        )

    elif dataset == 'circles':
        return datasets.make_circles(
            n_samples=n_samples,
            noise=noise,
            factor=0.5,
            random_state=1
        )

    elif dataset == 'linear':
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            'Data type incorrectly specified. Please choose an existing '
            'dataset.')


app.layout = html.Div(children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", children=[
        # Change App Name here
        html.Div(className='container scalable', children=[
            # Change App Name here
            html.H2(html.A(
                'Stacking and Blending for Peaks Prediction​',
                href='https://github.com/RiahiFedi/stacking-blending-1',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit'
                }
            )),

            html.A(
                html.Img(src="https://upload.wikimedia.org/wikipedia/commons/7/70/Logo_ESSAIT.png"),
                href='http://www.essai.rnu.tn/en/'
            )
        ]),
    ]),

    html.Div(id='body', className='container scalable', children=[
        html.Div(className='row', children=[
            
            html.Div(
                className='nine columns',
            style={'margin-top': '5px'},
                id='div-graphs',
                children=dcc.Graph(id='stack-graphs')
            ),
            
            
            html.Div(
                className='three columns',
                style={
                    'min-width': '24.5%',
                    'max-height': 'calc(100vh - 85px)',
                    'overflow-y': 'auto',
                    'overflow-x': 'hidden',
                },
                children=[
                    drc.Card([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                                ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                                },
                            # Allow multiple files to be uploaded
                            multiple=True
                            ),
                        ]),
                    
                    drc.Card([
                        html.Div(dcc.Input(id='input-box', type='text')),
                        html.Button('Submit', id='button'),
                        html.Div(id='output-container-button',
                                 children='Enter file path')
                        ]),
                    
                    drc.Card([
                        drc.NamedRadioItems(
                            name='PCA',
                            id='radio-stack-parameter-pca',
                            labelStyle={
                                'margin-right': '7px',
                                'display': 'inline-block'
                            },
                            options=[
                                {'label': ' Enabled', 'value': True},
                                {'label': ' Disabled', 'value': False},
                            ],
                            value=True,
                        ),
                        drc.NamedSlider(
                            name='Run Time',
                            id='slider-timer',
                            min=1,
                            max=60,
                            step=5,
                            marks={i: i for i in [1, 15, 30, 45, 60]},
                            value=5
                        ),
                        html.Br(),
                        html.Button(
                            'Generate The Stack',
                            id='run'
                        ),
                        ]),
                    
                ]
            ),
        ]),
    ]),
    html.Div(className='row',
             style={"width" : "100%"},
             children=[dash_table.DataTable(id='data-table')])
])
  

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV or TXT file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return df
                                     






@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('input-box', 'value')])
def update_output(n_clicks, value):
    return value
    




@app.callback([Output('div-graphs', 'children')],
              
              [Input('radio-stack-parameter-pca', 'value'),
                  Input('slider-timer', 'value'),
                  Input('run', 'n_clicks')],
              
              [State('output-container-button', 'children')],
              )
def update_stack_graph(pca,
                     timer,
                     n_clicks,
                     children):

    # Data Pre-processing
    if children != 'Please enter a valid file path':
        df = pd.read_csv(children,sep=';')
        df=df.dropna()
        X = df.drop(columns=['Y','Unnamed: 0']).values
        y = df.Y.values
    
    df_score = trials(X,y, pca_ = pca,t_delta_minutes = timer)
    
    df_score= df_score.sort_values(by= ['smape'])
    
    df_best = df_score.head(1).values.tolist()
    best_comp = list()
    for i in df_best[0][-1]:
       best_comp.append(i)
    
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.25,shuffle = False)
    final_model = temp_name(base_models = best_comp)
    final_model.fit(X,y)
    y_res = final_model.predict(X_val)
    figure=dict(
        data=[
            dict(
                x=np.arange(len(y_val)),
                y=y_val,
                name='Original data',
                marker=dict(
                    color='rgb(55, 83, 109)'
                    )
                ),
            dict(
                x=np.arange(len(y_val)),
                y=y_res,
                name='Predicted data',
                marker=dict(
                    color='rgb(26, 118, 255)'
                    )
                )
            ],
        layout=dict(
            title='Representation of the accuracy of the prediction of the Stack',
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
                ),
            margin=dict(l=40, r=0, t=40, b=30)
            )
        )
    children=[
                dcc.Graph(
                    id='stack-graphs',
                    figure=figure,
                )
            ]
    
    return children

@app.callback([Output('data-table', 'data'),
               Output('data-table', 'columns')],
              [ Input('button', 'n_clicks')],
               [State('output-container-button', 'children')],
              
            )
def update_table(n_clicks,children):
    
    if children != 'Please enter a valid file path':
        print('ok')
        print(type(children))
        df = pd.read_csv(children,sep=';')
        df=df.dropna()
    columns = [{'name': col, 'id': col} for col in df.columns]
    data = df.to_dict(orient='records')
    return data, columns


external_css = [
    # Normalize the CSS
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    # Fonts
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
