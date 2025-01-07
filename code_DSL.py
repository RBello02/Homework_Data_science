import csv

# Loading the dataset
with open('datasets/development.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    
# Printing the first 5 rows of the dataset
for row in data[:5]:
    print(row)