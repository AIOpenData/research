import json

def get_score(preds ,labels):
    '''
    评价指标
    :param preds:
    :param labels:
    :return:
    '''
    score = 0
    for auther in labels.keys():
        pre = preds[auther]
        label = labels[auther]
        score += len(set(pre) & set(label)) / 3

    print(score / len(labels))
    print(len(preds))
    print(len(labels))


if __name__ == "__main__":

    # 读取对标签的打分
    with open('output/top20_sort.json') as f:
        top20_sort = json.load(f)
    print(len(top20_sort))

    # 读取待预测标签
    author_to_pre30 = {}
    with open("output/recommend_top20.txt", 'r') as f:#raw_data_bkp/devresult.txt
        for item in f.readlines():#[2:-1]:
            item = item.strip().split('\t')
            author_to_pre30[item[0]] = item[1:]
    
    print(len(author_to_pre30))
    labels = {}


    # 读取答案
    with open('../raw_data_bkp/scholar_validation_truth.txt') as f:
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
            item = line.strip().split("\t")
            labels[item[0]] = item[1:]
    print(len(labels))
    # 方案1：排序
    pred = {}
    print('method 1 sort')
    for author in list(author_to_pre30.keys()):
        pos_score = top20_sort[author]
        pos_score = sorted(pos_score.items(), key=lambda d: d[1], reverse=True)[:5]
        array = author_to_pre30[author]
    #    print(pos_score[0])
        interests = [array[int(item[0])] for item in pos_score]
        pred[author] = interests
    get_score(pred, labels)

    # 方案2： 过滤
    pred = {}
    print('method 2 filter')
    for author in list(author_to_pre30.keys()):
        interests = []
        pre = author_to_pre30[author]
        pos = top20_sort[author]
        for i, j in zip(pre, pos):
            if len(interests) > 5:
                break
            
            if float(pos[str(pre.index(i))]) >= 0.5:
                interests.append(i)
            
        pred[author] = interests
    get_score(pred, labels)















