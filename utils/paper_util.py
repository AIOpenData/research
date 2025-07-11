import logging
import logging.handlers
import os
import re
from utils import train_util, valid_util
from utils.utils import timer


r = re.compile(r'\([^\)]+\)', re.S)


logger = logging.getLogger(__name__)


class PaperUtil(object):
    def __init__(self,
                 t_authors,
                 v_authors,
                 paper_dir: str):
        self.t_authors = t_authors  # 训练集作者
        self.v_authors = v_authors  # 验证集作者
        self.dic_ = {}  # 数据字典 { paper_idx: {author: [...], paper_name: [...], time: [...], journal: [...], cite: [...]}

        with timer("getting paper data...", logger):
            info = [[] for _ in range(6)]

            with open(paper_dir, "r", encoding="utf-8") as f:
                cite_ = []
                journal_ = ""
                for st in f:
                    if st.startswith("#index"):
                        info[0].append(st[6:].strip())

                    elif st.startswith("#@"):
                        info[1].append(st[2:].strip())

                    elif st.startswith("#*"):
                        info[2].append(st[2:].strip())

                    elif st.startswith("#t"):
                        info[3].append(st[2:].strip())

                    elif st.startswith("#c"):
                        journal_ = st[2:].strip()
                    elif st.startswith("#%"):
                        cite_.append(st[2:].strip())

                    elif st.startswith("\n"):
                        info[4].append(journal_)
                        info[5].append(",".join(cite_))
                        journal_ = ""
                        cite_ = []
                info[4].append(journal_)
                info[5].append(",".join(cite_))

                author = [j for i in info[1] for j in i.split(",")]
                self.authors_ = set(author) & (t_authors | v_authors)  # 候选作者空间, 与训练集和验证集取交集 {author1, author2...}

                for index, author, paper_name, time, journal, cite in zip(*info):
                    temp_authors = [a for a in author.split(",") if a in self.authors_]
                    if not temp_authors:
                        continue

                    if not self.dic_.__contains__(index):
                        self.dic_[index] = {}

                    self.dic_[index]["author"] = temp_authors
                    self.dic_[index]["paper_name"] = paper_name
                    self.dic_[index]["time"] = time
                    self.dic_[index]["order"] = {author_: author.split(",").index(author_) + 1 for author_ in temp_authors}

                    if journal != "":
                        self.dic_[index]["journal"] = r.sub("", journal).strip()

                    if cite != "":
                        self.dic_[index]["cite"] = cite.split(",")

                    # if int(index) % 10000 == 0:
                    #     logger.info("dic_[{}/{}]: {}".format(index, len(info[0]), self.dic_[index]))

                logger.info("dic_ dict size: {}, samples: {}\n".format(len(self.dic_.keys()), list(self.dic_.keys())[:10]))

