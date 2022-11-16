"""
INTRODUCTION
This program is for textual sentiment analysis of Chinese texts.
The input is could be any sentences or paragraph of Chinese texts, and output is the absolute value of quantified measure
(integer) of both positive and negative sentiment in the given text.

For measuring the sentiment, we modify the conventional bag-of-words algorithm with additional consideration of "weight",
i.e., the sentimental intensity within punctuations (namely "!") and adverbs modifying sentimental adjectives.

Due to the complicated nature of the Chinese expression, it is burdensome and technically challenging to build a novel
corpus from our text sample. Therefore, we use several commonly accepted corpora in the field (including the Chinese
sentimental word sets published by the labs of Tsinghua University, National Taiwan University and a rated dictionary
of adverbs derived with HOWNET).

In addition, since the target sample of this program consists of more than 1.3 million individual texts which lengths
vary across a few tens to thousands of words, the program is adapted to Mysql to fetch input text and write
output directly from the database for ease of data management.
"""

# jieba is for Chinese sentence cutting (into words), pkuseg is an alternative.
import jieba
import numpy as np
import pandas as pd
import chardet
import pymysql

# To check the encoding format of corpus files.
def check_charset(file_path):
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset


# To read the corpus files as list.
def open_dict(path):
    dictionary = open(path, 'r', encoding=check_charset(path))
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict


# Tell odd or even number
def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


# Open required corpora from given paths.
# Words of negation ("no" words)
deny_word = open_dict(path=r'a/noword.txt')
# Positive sentimental words
posdict = open_dict(path=r'd/dl_pos.txt')
# Negative sentimental words.
negdict = open_dict(path=r'd/dl_neg.txt')
# Form words and punctuation marks that need to be eliminated.
stopwords = open_dict(path=r'a/stop.txt')
# Rated adverbs.
degree_word = open_dict(path=r'程度级别词mod.txt')

# Assigning weight to the adverbs
mostdict = degree_word[degree_word.index('extreme') + 1: degree_word.index('very')]  # weight=4
verydict = degree_word[degree_word.index('very') + 1: degree_word.index('more')]  # weight=3
moredict = degree_word[degree_word.index('more') + 1: degree_word.index('ish')]  # weight=2
ishdict = degree_word[degree_word.index('ish') + 1: degree_word.index('last')]  # weight=0.5


# Modified bag-of-words algorithm
def sentiment_score_list(dataset):
    # cutting paragraph into sentences
    # To treat the input as a whole, replace the symbols into something unlikely to be in the text, like '%%%'.
    seg_sentence = dataset.split('。')
    count1 = []
    count2 = []
    for sen in seg_sentence:
        # cutting a sentence in to a word list
        segtmp = jieba.lcut(sen, cut_all=False, HMM=True,)
        i = 0  # the location of word under processing
        a = 0  # the location of identified sentimental word under processing
        # Positive sentiment
        poscount = 0  # Raw measurement
        poscount2 = 0  # After adjusting for "no" word.
        poscount3 = 0  # After adjusting for weight.
        # Negative sentiment
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        # Eliminating form words
        for word in segtmp:
            if word in stopwords:
                segtmp.remove(word)
        for word in segtmp:
            if word in posdict:  # if positive sentiment
                poscount += 1
                c = 0
                # finding adverbs modifying the sentimental
                for w in segtmp[a:i]:
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
            # Determine the number of "no" words (e.g., after double negation, the sentimental polarity is unchanged)
                if judgeodd(c) == 'odd':
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i + 1

            elif word in negdict:  # if negative sentiment
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
            elif word == '！' or word == '!':  # If the sentence ends with "!"
                for w2 in segtmp[::-1]:  # Weignting the sentimental word ahead of "!"
                    if w2 in posdict or negdict:
                        poscount3 += 2
                        negcount3 += 2
                        break
            # Adjust to non-negative measurement
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
            i += 1  # Target at next words
        count2.append(count1)
        count1 = []
    # Return the list of positive and negative measurement of the given paragraph
    return count2


# Calculate the overall sentiment of the paragraph, for simplicity we only use the arithmetic sum.
def sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:, 0])
        Neg = np.sum(score_array[:, 1])
        score.append([Pos, Neg,])
        # For additional purpose, we can obtain other statistics of the sentiments in the paragraph as well.
        # VarPos =np.var(score_array[:, 0])
        # score.append([Pos, Neg,VarPos])
    return score


# In case that the input text is a blank, the following function may prevent errors.
def EmotionByScore(data):
    try:
        result_list = sentiment_score(sentiment_score_list(data))
        return result_list[0][0], result_list[0][1]
    except:
        return ['x', 'x']

# Positive Sample
data1 = '今天上海的天气真好！我的心情非常高兴！如果去旅游的话我会非常兴奋！和你一起去旅游我会更加幸福！'
# Negative Sample
data2 = '救命，你是个坏人，救命，你不要碰我，救命，你个大坏蛋！'
# Neutral Sample
data3 = '美国华裔科学家,祖籍江苏扬州市高邮县,生于上海,斯坦福大学物理系,电子工程系和应用物理系终身教授!'

print(sentiment_score(sentiment_score_list(data1)))
print(sentiment_score(sentiment_score_list(data2)))
print(sentiment_score(sentiment_score_list(data3)))

# The following is for connecting to the database and processing the text set.
# Connecting to Mysql server
db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='123456cba', db='留言板', charset='utf8mb4')
cursor = db.cursor()
# Test if the connection is successful
cursor.execute("select version()")
data = cursor.fetchone()
print("Database Version:%s" % data)

# Fetching texts from database
sql_fetch = 'select * from lm_content_mod'
cursor.execute(sql_fetch)
raw_text = cursor.fetchall()
raw_text_pd = pd.DataFrame(raw_text, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', ])

# Processing the fetched text
senti_score = []
i = 0
for data in raw_text_pd.c7:  # c7 is the key corresponding to the text message
   i = i+1
   print("Processing No." + i)
   print(data)
   if not data is None:
      senti = EmotionByScore(data)
   else:  # In case that the text is a blank.
      senti = ['y', 'y', ]
   print(senti)
   senti_score.append(senti)
# Saving as csv file
senti_score_pd=pd.DataFrame(senti_score, columns=['Pos', 'Neg'])
senti_score_text = raw_text_pd.join(senti_score_pd)
print(senti_score_text)
senti_score_text.to_csv('senti_score_all.csv')


