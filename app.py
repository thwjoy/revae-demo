import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

from models.semisup_vae import REVAE
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from plotly.tools import mpl_to_plotly

revae = REVAE()
data = torch.load('./data/celeba.pt')

batch = data[np.random.choice(data.size(0), 100)]
grid = make_grid(batch, nrow=10)

#fig = px.imshow(grid.permute(1, 2, 0))
#fig.show()

coords = '2,2'

img = batch[(9 - int(coords[2])) * 10 + int(coords[0])]
recon = revae.reconstruct_img(img.unsqueeze(0))[0].detach()
# fig = plt.figure(figsize = (10,4))
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.imshow(img.permute(1, 2, 0))
# im = ax2.imshow(recon.permute(1, 2, 0))

# plotly_fig = mpl_to_plotly(im)
# graph = dcc.Graph(id='myGraph', fig=plotly_fig)

fig = px.imshow(recon.permute(1, 2, 0))
fig.show()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'title': 'Dash Data Visualization'
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
