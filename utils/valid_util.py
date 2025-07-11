import logging
import logging.handlers
import os

from utils.utils import timer

logger = logging.getLogger(__name__)


class ValidUtil(object):
    def __init__(self, valid_dir: str):
        with timer("getting valid data...", logger):
            self.labels_ = set()  # 标签 {label1, label2...}
            self.authors_ = set()  # 作者 {author1, author2...}
            self.dic_ = {}  # 数据字典 {author: [label1, label2...]}

            with open(valid_dir, "r", encoding="utf-8") as f:
                authors = []
                labels = []
                flag = False
                for line in f.readlines():
                    if "authorname" in line:
                        continue
                    if "<task2>" in line:
                        flag = True
                        continue
                    if "</task2>" in line:
                        flag = False
                        continue
                    if flag is False:
                        continue
                    author, interest1, interest2, interest3 = line.strip().split("\t")
                    authors.append(author)
                    labels.append(",".join([interest1, interest2, interest3]))

            # 收集作者池和标签池
            for author, label in zip(authors, labels):
                if not self.dic_.__contains__(author):
                    self.dic_[author] = set()
                labels = label.split(",")

                self.dic_[author].update(labels)
                self.labels_.update(labels)
                self.authors_.add(author)

            logger.info("labels_ set size: {}, samples: {}\n".format(len(self.labels_), list(self.labels_)[:10]))
            logger.info("authors_ set size: {}, samples: {}\n".format(len(self.authors_), list(self.authors_)[:10]))
            logger.info("dic_ dict size: {}, samples: {}\n".format(len(self.dic_.keys()), list(self.dic_.keys())[:10]))

