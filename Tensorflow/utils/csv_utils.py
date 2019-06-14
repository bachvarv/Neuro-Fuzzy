import csv

# csvData = [['Type', 'Time', 'Error']]
#
# with open('../csvFiles/fnn_data.csv', 'w+') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csvData)

def write_in_csv(file, rowData):
    with open(file, 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(rowData)