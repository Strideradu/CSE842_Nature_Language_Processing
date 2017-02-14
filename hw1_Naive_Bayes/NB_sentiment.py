# load all pos text in to pos list, neg text into neg list
from os import listdir
from os.path import isfile, join
import numpy as np
import argparse
from collections import MutableMapping
from collections import Counter
import pickle
import math

class NBmodel(object):
    def __init__(self):
        self.pos_prior = 0.0
        self.neg_prior = 0.0
        self.pos_freq_count = NBfreqdict(x = 1)
        self.neg_freq_count = NBfreqdict(x = 1)

    def train(self, text_with_label, smooth = 0.1):
        """

        :param text_with_label: text_with_label[0] is a list of all text, and [1] is list of label
        :return:
        """
        pos_token = []
        neg_token = []
        pos_count = 0.0
        neg_count = 0.0
        for i, text in enumerate(text_with_label[0]):
            if text_with_label[1][i]:
                pos_token.extend(text.lower().split())
                pos_count += 1
            else:
                neg_token.extend(text.lower().split())
                neg_count += 1

        train_tag(pos_token, self.pos_freq_count, smooth)
        train_tag(neg_token, self.neg_freq_count, smooth)
        self.pos_prior = pos_count/(pos_count + neg_count)
        self.neg_prior = neg_count/(pos_count + neg_count)

    def test(self, text_with_label):
        right_num = 0.0
        total_num = 0.0
        for i, text in enumerate(text_with_label[0]):
            result = infer_sentense(text, self)

            total_num += 1
            if result == text_with_label[1][i]:
                right_num += 1
        accuracy = right_num/float(total_num)
        print "The accuracy is " + str(accuracy)

    def save(self, path):
        save_list = [self.pos_prior, self.neg_prior, self.pos_freq_count, self.neg_freq_count]
        with open(path, "wb") as f:
            pickle.dump(save_list, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.pos_prior, self.neg_prior, self.pos_freq_count, self.neg_freq_count = pickle.load(f)

def train_tag(tokens, freq_count_dict, smooth):
    token_num = len(tokens)
    token_freq = Counter(tokens)
    token_class = len(token_freq)

    for token in tokens:
        freq_count_dict[token] = (token_freq[token] + smooth) / (token_num + token_class*smooth)

    freq_count_dict.set_nocount(smooth/(token_num + token_class*smooth))

def infer_sentense(text, model):
    token_list = text.lower().split()
    pos_log_score = 0
    neg_log_score = 0
    for token in token_list:
        p = model.pos_freq_count[token]
        pos_log_score += math.log(p)
        p = model.neg_freq_count[token]
        neg_log_score += math.log(p)

    if pos_log_score >= neg_log_score:
        return 1
    else:
        return 0


class NBfreqdict(MutableMapping):
    def __init__(self, *args, **kw):
        self._storage = dict(*args, **kw)
        self.nocount = 0.0

    def __getitem__(self, key):
        return self._storage.get(key, self.nocount)

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __iter__(self):
        return iter(self._storage)  # ``ghost`` is invisible

    def __len__(self):
        return len(self._storage)

    def __delitem__(self, key):
        self._storage.__delitem__(key)

    def set_nocount(self, nocount):
        self.nocount = nocount



def load_cross_valid(token_path):
    """
    load crossvalidation dataset
    :param token_path:
    :return: cv: a list of all text in cross validation
    """
    # cv[0] 1st fold of cv, cv[0][0] text cv[0][1] is label
    cv = [[[], []], [[],[]], [[],[]]]
    pos_path = token_path + "pos"
    neg_path = token_path + "neg"

    load_tag(pos_path, cv, 1)
    load_tag(neg_path, cv, 0)
    return cv

def load_tag(tag_path, cv_list, label):
    """

    :param tag_path:
    :param cv_list:
    :param label: 1 means pos tag, 0 means neg tag
    :return:
    """
    for f in listdir(tag_path):
        if isfile(join(tag_path, f)):
            cv_index = int(f.split("_")[0].strip("cv"))
            if cv_index <= 232:
                with open(join(tag_path, f)) as f1:
                    cv_list[0][0].append(f1.read())
                    cv_list[0][1].append(label)

            elif cv_index >= 466:
                with open(join(tag_path, f)) as f3:
                    cv_list[2][0].append(f3.read())
                    cv_list[2][1].append(label)
            else:
                with open(join(tag_path, f)) as f2:
                    cv_list[1][0].append(f2.read())
                    cv_list[1][1].append(label)


def main():
    # Use nargs to specify how many arguments an option should take.
    ap = argparse.ArgumentParser()
    ap.add_argument('-path', required=False, dest = "path", help="file path for token directory",  nargs=1)
    ap.add_argument('-train', dest = "train", help="select two fold for training", nargs=2)
    ap.add_argument('-test', dest = "test", help="select fold for test", nargs=1)
    ap.add_argument('-tmp_path', required=False, dest = "tmp", help="file path for temporary saved files, if no argument then program will not save the trainin result",  nargs=1)

    args = ap.parse_args()

    try:
        if args.path:
            token_path = args.path[0]
        else:
            token_path = "C:/Users/Nan/Documents/IPython Notebooks/CSE842/SentimentData/tokens/"

        if args.tmp:
            save_tmp = True
            save_path = args.tmp[0]
        else:
            save_tmp = False

    except:
        ap.print_help()

    dataset = load_cross_valid(token_path)

    train_data = [[], []]
    train_data[0].extend(dataset[int(args.train[0]) - 1][0])
    train_data[1].extend(dataset[int(args.train[0]) - 1][1])
    train_data[0].extend(dataset[int(args.train[1]) - 1][0])
    train_data[1].extend(dataset[int(args.train[1]) - 1][1])

    test_data = dataset[int(args.test[0]) - 1]

    nb_model = NBmodel()
    nb_model.train(train_data)

    if save_tmp:
        nb_model.save(save_path)
        nb_model.load(save_path)
    nb_model.test(test_data)




if __name__ == '__main__':
    main()