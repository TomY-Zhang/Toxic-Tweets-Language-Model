# flake8: noqa
import os, csv, math, time
from model import ToxicCommentModel


def parse_train_data(train_path, save_dir, classnames):
    idx = 0
    with open(train_path, 'r') as f:
        reader = csv.DictReader(f)
        for idx, line in enumerate(reader):
            text_save_path = f"{save_dir}/texts/{idx}.txt"
            labels_save_path = f"{save_dir}/labels/{idx}.txt"

            with open(text_save_path, 'w') as text_f, \
                 open(labels_save_path, 'w') as labels_f:
            
                text_f.write(line['comment_text'])
                
                labels = [line[name] for name in classnames]
                labels_f.write(','.join(labels))

    with open(f"{save_dir}/metadata.txt", 'w') as meta:
        meta.write(str(idx + 1))


def parse_test_data(text_path, label_path, save_dir, classnames):
    idx = 0
    with open(text_path, 'r') as texts_file, open(label_path, 'r') as labels_file:
        text_reader = csv.DictReader(texts_file)
        label_reader = csv.DictReader(labels_file)

        for idx, (text_line, label_line) in enumerate(zip(text_reader, label_reader)):
            text_save_path = f"{save_dir}/texts/{idx}.txt"
            labels_save_path = f"{save_dir}/labels/{idx}.txt"

            with open(text_save_path, 'w') as text_save, \
                 open(labels_save_path, 'w') as labels_save:
            
                text_save.write(text_line['comment_text'])
                
                labels = [label_line[name] for name in classnames]
                labels_save.write(','.join(labels))

    with open(f"{save_dir}/metadata.txt", 'w') as meta:
        meta.write(str(idx + 1))


def load_data_from_cache(batch_id, batch_size, total_examples, cache_dir):
    # batch_id begins at 0
    start = batch_id * batch_size
    stop = start + batch_size
    if stop > total_examples:
        stop = total_examples

    texts, labels = [], []
    for i in range(start, stop):
        with open(f"{cache_dir}/texts/{i}.txt", 'r') as t, \
             open(f"{cache_dir}/labels/{i}.txt", 'r') as l:
            
            texts.append(t.read())

            lbl = l.read().split(',')
            lbl = [int(val) for val in lbl]
            labels.append(lbl)

    return texts, labels


def train_model(model, batch_start, batch_stop, train, test):
    
    print(f"{batch_stop - batch_start} batches to train")
    print(f"Training batch size: {train['batch_size']}")
    print(f"Testing batch size: {test['batch_size']}")

    for batch_id in range(batch_start, batch_stop):
        start = time.time()

        print(f"\nLoading batch {batch_id}...")
        train_data = load_data_from_cache(batch_id, train["batch_size"], train["examples"], train["save_dir"])
        test_data = load_data_from_cache(batch_id, test["batch_size"], test["examples"], test["save_dir"])

        print(f"Training model...")
        model.train(*train_data)

        print(f"Testing model...")
        model.test(*test_data)

        stop = time.time()
        print(f"Batch finished. Time elapsed: {(stop-start):.3f} seconds")

    print("\nTraining complete.")


def main():
    train_path = "toxic-comments/train.csv"
    test_text_path = "toxic-comments/test.csv"
    test_label_path = "toxic-comments/test_labels.csv"

    caches = ['train', 'test']
    for cache in caches:
        if not os.path.exists(f'toxic-comments/{cache}'):
            os.makedirs(f'toxic-comments/{cache}')
            os.makedirs(f'toxic-comments/{cache}/texts')
            os.makedirs(f'toxic-comments/{cache}/labels')

    model_save_dir = '../saved_model'
    train_save_dir = 'toxic-comments/train'
    test_save_dir = 'toxic-comments/test'

    classnames = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    model = ToxicCommentModel(save_dir=model_save_dir, classes=classnames)

    # parse_train_data(train_path, train_save_dir, classnames)
    # parse_test_data(test_text_path, test_label_path, test_save_dir, classnames)

    num_batches = 50
    with open(f"{train_save_dir}/metadata.txt", 'r') as train_meta, \
        open(f"{test_save_dir}/metadata.txt", 'r') as test_meta:
        
        train, test = {}, {}

        train["examples"] = int(train_meta.read())
        train["batch_size"] = math.ceil(train["examples"] / num_batches)
        train["save_dir"] = train_save_dir

        test["examples"] = int(test_meta.read())
        test["batch_size"] = test["examples"] // num_batches
        test["save_dir"] = test_save_dir
        
        batch_start = 0
        batch_stop = 1
        train_model(model, batch_start, batch_stop, train, test)

    model.save()


if __name__ == '__main__':
    main()