import csv
import plotly.plotly as py
import plotly.graph_objs as go


def create_table(labels, type, time, error):
    header = dict(values=['<b>' + labels[0] + '</b>',
                          '<b>' + labels[1] + '</b>',
                          '<b>' + labels[2] + '</b>'],
                  line=dict(color='#7D7F80'),
                  fill=dict(color='#DBDBDB'),
                  align=['left'] * 5)

    cells = dict(values=[type,
                         time,
                         error],
                 line=dict(color='#7D7F80'),
                 fill=dict(color='#FFFFFF'),
                 align=['left'] * 5)
    table = go.Table(header=header, cells=cells)

    layout = dict(width=800, height=600)
    data = [table]
    fig = dict(data=data, layout=layout)

    py.plot(fig, filename='fnn_data')
    return fig


def read_csv(filename):
    labels = []
    type = []
    time = []
    error = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
                labels.append(row[0])
                labels.append(row[1])
                labels.append(row[2])
                print("------------------")
                print(labels)
                print("------------------")
            else:
                type.append(row[0])
                time.append(row[1] + 's')
                error.append(row[2])
                print(row[0], row[1], row[2])
    create_table(labels, type, time, error)


read_csv('/home/bachvarv/Neuro-Fuzzy-System/Neuro-Fuzzy/Tensorflow/csvFiles/fnn_data.csv')
