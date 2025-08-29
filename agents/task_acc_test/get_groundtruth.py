import json
import csv

def json_to_csv(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    with open(csv_path, 'w', encoding='utf-8', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['id', 'text', 'true_label'])
        for idx, item in enumerate(data):
            writer.writerow([idx, item['input'], 0])

if __name__ == '__main__':
    json_to_csv('task_acc_test/data/input.json', 'task_acc_test/data/groundtruth.csv')
    print('已将input字段内容写入groundtruth.csv')