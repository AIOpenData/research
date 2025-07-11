import copy
import logging
import logging.handlers
import numpy as np
import os

from utils.utils import timer

logger = logging.getLogger(__name__)


class TrainUtil(object):
    def __init__(self, train_dir: str):
        with timer("getting training data...", logger):
            self.labels_ = set()  # 标签 {label1, label2...}
            self.authors_ = set()  # 作者 {author1, author2...}
            self.frequency_ = {}  # 标签流行度 {label: frequency}
            self.dic_ = {}  # 数据字典 {author: [label1, label2...]}

            with open(train_dir, "r", encoding="utf-8") as f:
                authors = []
                labels = []
                for idx, item in enumerate(filter(lambda st: st != "\n", f.readlines())):
                    if idx % 2 == 0:
                        authors.append(item.strip())
                    else:
                        labels.append(item.strip())

            # 收集作者池和标签池
            for author, label in zip(authors, labels):
                if not self.dic_.__contains__(author):
                    self.dic_[author] = set()
                labels = label.split(",")

                # self.dic_[author].update({}.fromkeys(labels, 1))
                self.dic_[author].update(labels)
                self.labels_.update(labels)
                self.authors_.add(author)

                # 收集标签流行度
                for i in labels:
                    if not self.frequency_.__contains__(i):
                        self.frequency_.setdefault(i, 0)
                    self.frequency_[i] += 1

            logger.info("labels_ set size: {}, samples: {}\n".format(len(self.labels_), list(self.labels_)[:10]))
            logger.info("authors_ set size: {}, samples: {}\n".format(len(self.authors_), list(self.authors_)[:10]))
            logger.info("frequency_ dict size: {}, samples: {}\n".format(len(self.frequency_.keys()),
                                                                         list(self.frequency_.keys())[:10]))
            logger.info("dic_ dict size: {}, samples: {}\n".format(len(self.dic_.keys()), list(self.dic_.keys())[:10]))

    def train_test_split(self, test_size):
        with timer("splitting into training and testing data...", logger):
            output = []
            temp_frequency = copy.deepcopy(self.frequency_)

            for author, labels in self.dic_.items():
                output.append([author, *labels])

            output = np.array(output)
            np.random.shuffle(output)

            train_idx = []
            test_idx = []

            N = 0
            for i in range(output.shape[0]):
                tag = False
                for j in output[i, 1:]:
                    if temp_frequency[j] < 2:
                        tag = True

                if tag:
                    train_idx.append(i)
                    continue

                test_idx.append(i)
                for j in output[i, 1:]:
                    temp_frequency[j] -= 1
                N += 1

                if N == test_size:
                    train_idx.extend([k for k in range(i + 1, output.shape[0])])
                    break

            train = output[train_idx, :]
            test = output[test_idx, :]
            train_dic = {}
            test_dic = {}

            for i in range(train.shape[0]):
                train_dic[train[i][0]] = set(train[i][1:].tolist())
            for i in range(test.shape[0]):
                test_dic[test[i][0]] = set(test[i][1:].tolist())

            logger.info("train_noiid size: {}, samples: {}\n".format(len(train_dic.keys()), list(train_dic.keys())[:10]))
            logger.info("test size: {}, samples: {}\n".format(len(test_dic.keys()), list(test_dic.keys())[:10]))

        return {"train_noiid": train_dic, "test": test_dic}


