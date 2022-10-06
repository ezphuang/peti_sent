# coding:utf-8
import jieba
import numpy as np
import chardet
def check_charset(file_path):
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


# 打开词典文件，返回列表
def open_dict1(path):
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def open_dict2(path):
    dictionary = open(path, 'r', encoding='gbk')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def open_dict3(path):
    dictionary = open(path, 'r', encoding=check_charset(path))
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict

def check_charset(file_path):
    import chardet
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset



def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


# 注意，这里你要修改path路径。
deny_word = open_dict1(path=r'a/noword.txt')
posdict = open_dict3(path=r'd/dl_pos.txt')
negdict = open_dict3(path=r'd/dl_neg.txt')
stopwords = open_dict3(path=r'a/stop.txt')
degree_word = open_dict2(path=r'程度级别词mod.txt')
#print(degree_word)
mostdict = degree_word[degree_word.index('extreme') + 1: degree_word.index('very')]  # 权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very') + 1: degree_word.index('more')]  # 权重3
moredict = degree_word[degree_word.index('more') + 1: degree_word.index('ish')]  # 权重2
ishdict = degree_word[degree_word.index('ish') + 1: degree_word.index('last')]  # 权重0.5

#degree_word = open_dict(Dict='程度级别词语', path='情感极性词典/BosonNLP_sentiment_score/BosonNLP_sentiment_score.txt')


def sentiment_score_list(dataset):
    seg_sentence = dataset.split('%%%')
    count1 = []
    count2 = []
    for sen in seg_sentence:  # 循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False, HMM=True)  # 把句子进行分词，以列表的形式返回
        i = 0  # 记录扫描到的词的位置
        a = 0  # 记录情感词的位置
        poscount = 0  # 积极词的第一次分值
        poscount2 = 0  # 积极词反转后的分值
        poscount3 = 0  # 积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in stopwords:
                segtmp.remove(word)
        for word in segtmp:
            if word in posdict:  # 判断词语是否是情感词
                poscount += 1
                c = 0
                for w in segtmp[a:i]:  # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1
                if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1  # 情感词的位置变化

            elif word in negdict:  # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!':  ##判断句子是否有感叹号
                for w2 in segtmp[::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            i += 1  # 扫描词位置前移
            # 以下是防止出现负数的情况
            neg_count = 0
            pos_count = 0
            if poscount3 < 0 and negcount3 >= 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 < 0 and poscount3 >= 0:
                pos_count += poscount3 - negcount3
                neg_count = 0
            elif poscount3 < 0 and negcount3 < 0:
                neg_count = -poscount3
                pos_count = -negcount3
            else:
                pos_count = poscount3
                neg_count = negcount3
            count1.append([pos_count, neg_count])
        #print(count1)
        count2.append(count1)
        #print(count2)
        count1 = []
    #print(count2)
    return count2


def sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:, 0])
        Neg = np.sum(score_array[:, 1])
        score.append([Pos, Neg,])  # 积极、消极情感值总和(最重要)，积极、消极情感均值，积极、消极情感方差。
    return score

def EmotionByScore(data):
    try:
        result_list = sentiment_score(sentiment_score_list(data))
        return result_list[0][0], result_list[0][1]
    except:
        return ['x', 'x']

def JudgingEmotionByScore(Pos, Neg):
    if Pos > Neg:
        str = '1'
    elif Pos < Neg:
        str = '-1'
    elif Pos == Neg:
        str = '0'
    return str

def relativesocre(data):
    Pos, Neg = EmotionByScore(data)
    ass = Pos - Neg
    if Pos+Neg == 0:
        rs=0
    else:
        rs=ass/((Pos+Neg)/2)
    return [rs,ass]


data1 = '今天上海的天气真好！我的心情非常高兴！如果去旅游的话我会非常兴奋！和你一起去旅游我会更加幸福！'
data11 = '今天上海的天气真好！我的心情非常高兴! 。如果去旅游的话我会非常兴奋！和你一起去旅游我会更加幸福！'
data2 = '救命，你是个坏人，救命，你不要碰我，救命，你个大坏蛋！'
data3 = '美国华裔科学家,祖籍江苏扬州市高邮县,生于上海,斯坦福大学物理系,电子工程系和应用物理系终身教授!'

print(sentiment_score(sentiment_score_list(data1)))
print(sentiment_score(sentiment_score_list(data2)))
print(sentiment_score(sentiment_score_list(data3)))

#a, b = EmotionByScore(data1)
#print(emotion)


