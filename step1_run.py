import logging.handlers
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.linear_model import SGDClassifier
from utils import train_util, paper_util, valid_util
from utils.utils import init_log, timer


logger = logging.getLogger(__name__)


class Task2Model2(object):
    def __init__(self, data_dir: str):

        self.data_dir = data_dir
        self.t_util = train_util.TrainUtil(train_dir=os.path.join(data_dir, "training.txt"))
        self.v_util = valid_util.ValidUtil(valid_dir=os.path.join(data_dir, "scholar_validation_truth.txt"))
        self.p_util = paper_util.PaperUtil(t_authors=self.t_util.authors_, v_authors=self.v_util.authors_, paper_dir=os.path.join(data_dir, "papers.txt"))


    def fit(self):

        with timer("loading training data...", logger):
            train_dic = self.t_util.dic_ # {author: [label1, label2...]}
            # 获得所有的作者
            self.train_author = [author for author in train_dic.keys()]
            # 基于标签特征集
            train_context_based_label, self.train_label_based_label = self.load_info_based_label(train_dic)
            # 基于作者特征集
            train_context_based_author, self.train_label_based_author = self.load_info_based_author(train_dic)
            # 论文整理
            train_paper_info, self.train_context_writing_order, self.train_journal_writing_order = self.load_paper_info(
                train_dic.keys())

        with timer("getting tf-idf features...", logger):
            # onehot编码
            self.context_vec_based_label = CountVectorizer(stop_words="english", decode_error="ignore",
                                                           ngram_range=(1, 2)).fit(train_context_based_label)
            train_context_count_based_label = self.context_vec_based_label.transform(train_context_based_label)
            # chi2特征提取（前50%）
            self.context_selection_based_label = SelectPercentile(chi2, percentile=50).fit(train_context_count_based_label,
                                                                                           self.train_label_based_label)
            train_context_new_based_label = self.context_selection_based_label.transform(train_context_count_based_label)
            # TFIDF编码
            self.context_tf_idf_transformer_based_label = TfidfTransformer().fit(train_context_new_based_label)
            self.train_context_tf_idf_based_label = self.context_tf_idf_transformer_based_label.transform(
                train_context_new_based_label)

            # onehot编码
            self.context_vec_based_author = CountVectorizer(stop_words="english", decode_error="ignore",
                                                            ngram_range=(1, 2)).fit(train_context_based_author)
            train_context_count_based_author = self.context_vec_based_author.transform(train_context_based_author)
            # chi2特征提取（前13%）
            self.context_selection_based_author = SelectPercentile(chi2, percentile=13).fit(
                train_context_count_based_author, self.train_label_based_author)
            train_context_new_based_author = self.context_selection_based_author.transform(
                train_context_count_based_author)

            # TFIDF编码
            self.context_tf_idf_transfer_based_author = TfidfTransformer().fit(train_context_new_based_author)
            self.train_context_tf_idf_based_author = self.context_tf_idf_transfer_based_author.transform(
                train_context_new_based_author)

        # SGD 分类器（梯度下降法）
        with timer("training based on author...", logger):
            self.clf_SGD = SGDClassifier(loss="log", n_jobs=-1, shuffle=True,verbose=0).fit(
                self.train_context_tf_idf_based_author, self.train_label_based_author)#SGDClassifier#loss="log", n_jobs=-1, shuffle=False, verbose=0

        with timer("training based on label...", logger):
            self.clf_SGD_ = SGDClassifier(loss="log", n_jobs=-1, shuffle=True, verbose=0).fit(
                self.train_context_tf_idf_based_label, self.train_label_based_label)#loss="log", n_jobs=-1, shuffle=False, verbose=0


    def recommend(self, author_list):

        with timer("loading testing data...", logger):
            # 获得测试作者列表
            self.test_author = [author for author in author_list]

        with timer("getting tf-idf features...", logger):
            # 获取测试集两个特征的TFIDF
            recommend_list = {}

            test_paper_info, test_context_writing_order, test_journal_writing_order = self.load_paper_info(author_list)
            test_paper_info = [test_paper_info[author] for author in self.test_author]

            test_context_count_based_label = self.context_vec_based_label.transform(test_paper_info)
            test_context_new_based_label = self.context_selection_based_label.transform(test_context_count_based_label)
            self.test_context_tf_idf_based_label = self.context_tf_idf_transformer_based_label.transform(
                test_context_new_based_label)

            test_context_count_based_author = self.context_vec_based_author.transform(test_paper_info)
            test_context_new_based_author = self.context_selection_based_author.transform(test_context_count_based_author)
            self.test_context_tf_idf_based_author = self.context_tf_idf_transfer_based_author.transform(
                test_context_new_based_author)

        with timer("testing...", logger):
            # 预测
            predict_probas_SGD = self.clf_SGD.predict_proba(self.test_context_tf_idf_based_author)
            predict_probas_SGD_ = self.clf_SGD_.predict_proba(self.test_context_tf_idf_based_label)

            clf_labels = self.clf_SGD.classes_
            clf_labels_ = self.clf_SGD_.classes_

        with timer("recommending...", logger):
            for author in author_list:
                recommend_count = {}

                idx = self.test_author.index(author)

                label_clf_ = set(clf_labels_[np.argsort(predict_probas_SGD_[idx])[-23:]])
                label_clf = set(clf_labels[np.argsort(predict_probas_SGD[idx])[-24:]])

                for label in label_clf | label_clf_:

                    metric = predict_probas_SGD[idx][clf_labels.tolist().index(label)] * \
                             predict_probas_SGD_[idx][clf_labels_.tolist().index(label)]

                    if label not in recommend_count.keys():
                        recommend_count[label] = 0
                    recommend_count[label] += metric

                    for paper_name, order in test_context_writing_order.get(author, {}).items():
                        if order <= 2:
                            recommend_count[label] += paper_name.lower().count(label) * metric

                        else:
                            recommend_count[label] += paper_name.lower().count(label) * metric

                    for journal, orders in test_journal_writing_order.get(author, {}).items():
                        for order, freq in orders.items():

                            if order <= 2:
                                recommend_count[label] += freq * journal.lower().count(label) * metric

                            else:
                                recommend_count[label] += 0.8 * freq * journal.lower().count(label) * metric

                recommend_list[author] = [item[0] for item in
                                          sorted(recommend_count.items(), key=lambda x: x[1], reverse=True)[:20]]

        return recommend_list

    def validation(self):
        # 获得推荐的兴趣
        recommend_list = self.recommend(self.v_util.authors_)
        with open('output/recommend_top20.txt','w') as w:
            for item in recommend_list.keys():
                wline = [item]+recommend_list[item]
                line = '\t'.join(wline)
                w.write(line+'\n')
                
        score = 0
        cnt = 0
        for author, labels in self.v_util.dic_.items():
            recommend_labels = set(recommend_list[author])
            score += len(recommend_labels & labels) / len(labels)
            if cnt % 100 == 0:
                logger.info("\ncnt: {}/{}, author: {}, recommend_labels: {}, labels: {}".format(cnt, len(self.v_util.dic_), author, recommend_labels, labels))
            cnt += 1

        score = score / len(recommend_list)

        logger.info("score: {}\nrecommend_list size: {}\n".format(score, len(recommend_list)))

        return score, recommend_list

    def load_paper_info(self, tv_authors):
        paper_info_dic_ = {}#author：(paper_name, journal)
        context_writing_order_ = {}#{author:{paper:order author}}
        journal_writing_order_ = {}#{author:{journal:{order author:num}

        for index, item in self.p_util.dic_.items():
            authors = filter(lambda st: set(tv_authors).__contains__(st), item["author"])

            if not authors:
                continue

            paper_name = item["paper_name"]
            order = item["order"]# 论文的共同作者
            journal = ""

            if "journal" in item.keys():
                journal = item["journal"]

            for author in authors:

                if not paper_info_dic_.__contains__(author):
                    paper_info_dic_[author] = set()
                paper_info_dic_[author].add((paper_name, journal))

                if author not in context_writing_order_.keys():
                    context_writing_order_[author] = {}
                context_writing_order_[author][paper_name] = order[author]

                if journal == "":
                    continue

                if author not in journal_writing_order_.keys():
                    journal_writing_order_[author] = {}

                if journal not in journal_writing_order_[author].keys():
                    journal_writing_order_[author][journal] = {}

                if order[author] not in journal_writing_order_[author][journal].keys():
                    journal_writing_order_[author][journal][order[author]] = 0
                journal_writing_order_[author][journal][order[author]] += 1

        paper_info_dic_ = {author: "\n".join(map(lambda tup: " ".join(tup), paper_names)) for author, paper_names in
                           paper_info_dic_.items()}

        return paper_info_dic_, context_writing_order_, journal_writing_order_

    def load_info_based_label(self, train):

        dic_ = {}#{label:[(paper_name, journal)]}

        for index, item in self.p_util.dic_.items():
            authors = filter(lambda st: set(train.keys()).__contains__(st), item["author"])

            if not authors:
                continue

            paper_name = item["paper_name"]

            journal = ""
            if "journal" in item.keys():
                journal = item["journal"]

            for author in authors:

                labels = train[author]

                for label in labels:

                    if not dic_.__contains__(label):
                        dic_[label] = set()

                    dic_[label].add((paper_name, journal))

        dic_ = {label: "\n".join(set(map(lambda tup: " ".join(tup), paper_names))) for label, paper_names in
                dic_.items()}

        context_ = []
        label_ = []
        for label, context in dic_.items():
            context_.append(context)
            label_.append(label)

        return context_, label_

    def load_info_based_author(self, train):
        context_dic = {}#{label:[author1,author2]}

        for index, item in self.p_util.dic_.items():
            authors = filter(lambda st: set(train.keys()).__contains__(st), item["author"])

            if not authors:
                continue

            paper_name = item["paper_name"]

            journal = ""

            if "journal" in item.keys():
                journal = item["journal"]

            for author in authors:

                if not context_dic.__contains__(author):
                    context_dic[author] = []
                context_dic[author].append((paper_name, journal))

        context_dic = {author: "\n".join(map(lambda x: " ".join(x), context)) for author, context in
                       context_dic.items()}

        context_ = []
        label_ = []
        for author, context in context_dic.items():
            for label in train[author]:
                label_.append(label)
                context_.append(context)

        return context_, label_


if __name__ == "__main__":
    # 初始化日志输出配置
    init_log(logging.INFO)

    remote_data_folder = "../raw_data_bkp"

    model = Task2Model2(data_dir=remote_data_folder)

    model.fit()

    score, recommend_list = model.validation()
