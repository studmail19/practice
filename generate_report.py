
import json
import csv

with open("history.json", 'r') as json_file:
    history = json.load(json_file)

total_count = 0
count_list = []
filename_list = []

for entry in history:
    filename = entry["filename"]
    count = entry["detections"]

    filename_list.append(filename)
    count_list.append(count)
    total_count += count

average_count = total_count / len(count_list) if count_list else 0
max_count = max(count_list) if count_list else 0
min_count = min(count_list) if count_list else 0

csv_data = [["Filename", "Count", "Total Count", "Average Count", "Max Count", "Min Count"]]
first = True
for filename, count in zip(filename_list, count_list):
    if first:
        csv_data.append([filename, count, total_count, average_count, max_count, min_count])
        first = False
    else:
        csv_data.append([filename, count])

with open("history.csv", 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print("Data has been successfully written to history.csv")
