import sys, csv
args = sys.argv

def evaluate(pred_path, gt_path, split = False):
    with open(pred_path, mode='r') as pred:
        reader = csv.reader(pred)
        pred_dict = {rows[0]:rows[1] for rows in reader}

    with open(gt_path, mode='r') as gt:
        reader = csv.reader(gt)
        gt_dict = {rows[0]:rows[1] for rows in reader}

    total_count = 0
    correct_count = 0
    for key, value in pred_dict.items():
        if key not in gt_dict:
            if split:
                continue
            sys.exit("Item mismatch: \"{}\" does not exist in the provided ground truth file.".format(key))
        if value == 'label':
            continue
        if gt_dict[key] == value:
            correct_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100
    print('Accuracy: {}/{} ({}%)'.format(correct_count, total_count, accuracy))
