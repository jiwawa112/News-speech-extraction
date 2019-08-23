#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import jieba
import math
import time
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine

from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import Segmentor

LTP_DATA_DIR = 'F:/python/NLP_course/Project_01/ltp_data_v3.4.0'    # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')    #词性标注模型路径
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径

def split_sentece_1(sentence):
    N = len(sentence)
    """按照引号切分句子"""
    res = []
    stack = []
    for i,char in enumerate(sentence):
        if char == '“':
            stack.append(i)
            continue
        if char == "”":
            start = stack.pop()
            # if sentence[i-1] == '。': # direct quotation
            #    res.append(sentence[l:i+1])
            if i + 1 < N and sentence[i + 1] == '。':  # direct quotation
                res.append(sentence[start:i+1])
    return res
# print(split_sentece_1(sentence))

def split_sentece_2(sentence):
    """按照句号切分句子"""
    res = list()
    start = 0
    N = len(sentence)
    for i,char in enumerate(sentence):
        if char == "。" or char == "？" or char == "！":
            res.append(sentence[start:i])
            start = i + 1
    return res

def cut(str):
    return ' '.join(jieba.cut(str))

def token(string):
    return ' '.join(re.findall('[\w|\d]+',string))

# 正则表达式将段落切分为句子
# 缺点：引号里存在句号的句子会直接分开，而不是保留引号内容的完整性
def cut_sentence(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    # para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    # para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。

    return para.split('\n')

# 获取词频
def get_count(word):
    # vector = np.zeros(1)

    if word in word_dict:
        wf = word_dict[word].count
        wv = model.wv[word]
    else:
        wf = 1
        wv = np.zeros(dim)
    return wf / word_total_count,wv

# 获取句子向量
def sentence_embedding(sentence):
    a = 1e-3

    words = cut(token(sentence)).split()
    sum_vector = np.zeros(dim)
    for i,w in enumerate(words):
        wf,wv = get_count(w)
        sum_vector += a / (a + wf) * wv
    return sum_vector / len(words)

# 句子相关性分析
def get_corrlations(text,person_sentence):
    if isinstance(text,list):
        text = ' '.join(text)

    sub_sentences = cut_sentence(text)

    # 要与其他句子进行相似度判断的人物言论句子向量
    person_sentence_vector = sentence_embedding(person_sentence)

    correlations = {}

    index = sub_sentences.index(person_sentence)
    try:
        rest_sentences = sub_sentences[index+1:]
    except IndexError:
        print('没有更多的言论！')

    for sub_sentence in rest_sentences:
        # if sub_sentence != person_sentence:
        sub_sen_vec = sentence_embedding(sub_sentence)
        correlation = cosine(person_sentence_vector,sub_sen_vec)
        correlations[sub_sentence] = correlation

    return sorted(correlations.items(),key=lambda x:x[1],reverse=True)

# 欧式距离
def euclidSimilar(inA,inB):
    return np.sqrt(np.sum(np.square(inA - inB)))
# print(euclidSimilar(a,b))

# 皮尔森相关系数
def pear(inA,inB):
    return 0.5 + 0.5 * np.corrcoef(inA,inB)
# print(pear(a,b))

# 余弦相似度
def CosSimilar(inA,inB):
    inA = np.mat(inA)
    inB = np.mat(inB)
    num = float(inA * inB.T)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    cos = num / denom
    sim = 0.5 + 0.5*cos
    return sim

# pyltp中文分词
def pyltp_cut(sentence):
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(sentence)  # 分词
    segmentor.release()  # 释放模型
    return words

# pos词性标注
def get_pos(words):
    """
    词性标注
    :param words: 分词
    :return: postags
    """
    postagger = Postagger()  # 初始化实例
    postagger.load(pos_model_path)  # 加载模型
    postags = postagger.postag(words)  # 词性标注
    # print('\t'.join(postags))
    postagger.release()  # 释放模型

    return postags

# postags句法分析
def get_parser(words,postags):
    """
    依存句法分析
    :param words:分词
    :param postags: 词性标注
    :return: arcs
    """
    parser = Parser()  # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words,postags)  # 句法分析
    # print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
    parser.release()  # 释放模型
    return arcs

# ner命名实体识别
def get_name_entity(words,postags):
    """
    命名实体识别
    :param words:分词
    :param postags:词性标注
    :return: netags
    """
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型
    netags = recognizer.recognize(words,postags)  # 命名实体识别
    # print('\t'.join(netags))
    recognizer.release()  # 释放模型
    return netags

# 获取发言实体
def get_person(name,words,postags):
    index = words.index(name)
    cut_postags = postags[:index] # 截取name的所有词性标注
    pre = words[:index] #前半部分
    person = ''

    # print(cut_postags)
    # print(index)
    # print('pre',pre)

    # 查找前面的发言人
    if pre:
        while pre:
            w = pre.pop()
            w_index = words.index(w)
            if cut_postags[w_index] in ['ns','ni','nh'] and w not in ['”']:
                person = w
    else:
        return name
    return person

# 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语
# 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语
def get_name(name,verb,words,arcs):
        """传入的是单个句子"""
        index = words.index(name)
        arcs_relation = [arc.relation for arc in arcs]
        cut_arcs = arcs_relation[index + 1:]  # 截取name后第一个词语
        pre = words[:index]  # 前半部分
        pos = words[index + 1:]  # 后半部分

        # print(index)
        # print(cut_arcs)
        # print('pre',pre)
        # print('pos',pos)

        # 向前拼接主语的定语
        while pre:
            w = pre.pop()
            w_index = words.index(w)

            if arcs_relation[w_index] == 'ADV':
                continue
            if arcs_relation[w_index] in ['ATT','SVB'] and (w not in ['，','。','（','）','、']):
                name = w + name
            else:
                pre = False
        print(name)
        # 向后拼接
        while pos:
            w = pos.pop(0)
            p = cut_arcs.pop(0)
            #if p in ['WP','ADV','ATT','LAD','COO','RAD'] and (w not in ['，','。','、','）','（']):
            if p not in ['WP','HED']:
                name = name + w
            else:
                return name
        print(name)
        return name

# 获取谓语之后的言论
def get_says(verb,words,arcs_relation):
    index = words.index(verb)
    words = words[index + 1:]
    #cut_arcs = list(arcs_relation[index + 1:])

    # print(words)
    # print(cut_arcs)

    for word in words:
        if word != '。':
            verb = verb + word
        else:
            break
    return verb

# 从单个句子中提取言论
def single_sentence(similar_word,words,postags,arcs):
    """解析的句子是单个句子"""
    name = ''
    stack = list()
    for k,v in enumerate(arcs):
        if postags[k] in ['nh','ni','ns']:
            stack.append(words[k])

        if v.relation == 'SBV' and (words[v.head - 1] in similar_word):  # 确定第一个主谓句
            print(words[k])
            print(words[v.head - 1])
            name = get_name(words[k],words[v.head - 1],words,arcs)  # 曼城内部消息
            person = get_person(words[k],words,postags)
            print(name)
            print(person)
            says = get_says(words[v.head - 1],words,arcs)
            print(says)
            print(name + says)

            return person,name+says


def main():
    # 将一个段落切分成多个句子组成的列表
    sentences = cut_sentence(text)
    print(sentences)

    person_speech = list()
    for sentence in sentences:
        # 1.分词
        words = list(pyltp_cut(sentence))
        print('\t'.join(words))

        # 2.词性标注
        postags = list(get_pos(words))
        print('\t'.join(postags))

        # 3.句法分析
        arcs = get_parser(words,postags)
        arcs_relation = [arc.relation for arc in arcs]
        print(arcs_relation)

        # 4.命名实体
        netags = get_name_entity(words,postags)

        # 判断是否有'说'的相关词
        similar_word = [word for word in words if word in say_similary_words]
        print(similar_word)

        # 从单个句子中提取言论及发言人实体
        if similar_word:
            person,says = single_sentence(similar_word,words,postags,arcs)
            print(says)
            print('发言人:',person)
            person_speech.append((person,says))
    print()
    print(person_speech)

    # # 获取句子向量、及句子相关性分析
    # sentences_corrlations = list()
    # for person_says in person_speech:
    #     sentence_corrlations = get_corrlations(text,person_says[1])
    #     for sentence_corrlation in sentence_corrlations:
    #         if sentence_corrlation[1] > 0.8:
    #             return person_says + sentence_corrlation[0]


if __name__ == '__main__':
    dim = 200
    say_similary_words = ['指出','声称','表示','说','问道','认为','称','宣称','说道','表明','辩称','相信','写道','觉得','坦言','提到','坚称','问','否认','反问','质问','并不认为','确信',
                          '告诉','答道','看来','指责','询问','强调','谈到','知道','坦承','回忆起','问起','所说','直言','谈及','得知','坚信','暗示','回答','感叹','视为','斥责','怒斥','谈起',
                          '说出','深信','察觉到','告知','透露','表达','聊到','追问','责问','慨叹','请问','知晓','矢口否认','详述','承认','责怪','提及','查问','察觉','普遍认为','说明','知悉',
                          '所称','证实','痛斥','痛骂','发觉','所指','证明','所言','地问','明确指出','获知','阐述','指证','视作','得悉','指控','指斥','实在','看上去','真是','中称','中说','却说',
                          '详细描述','确认','怀疑','所述','感慨','闻言','引用','提过','认作','痛批','无聊','感到','感觉','有点','谈论','回想','显示','看作','看做','看成','断言','并不知道',
                          '发问','答','明言','写给','叹息','坦白','反指','明知','当成','始终认为','一再强调','驳斥','诋毁','明说','说起','聊起','记述','特别强调','保持沉默','反驳','澄清',
                          '指摘','问及','责备']
    valid_sentence = list()

    # text = "据英国媒体《太阳报》消息，曼城核心大卫-席尔瓦即将和球队续约两年，以保证他在俱乐部的长期未来。席尔瓦非常享受在瓜迪奥拉麾下效力的时光，他希望继续为曼城征战。曼城也愿意留下这位前场核心。大卫-席尔瓦和曼城的现有合同要到2019年，曼城希望能和他续约至2021年，足以见出球队对他的肯定和信任。尽管已经31岁了，但是大卫-席尔瓦依然曼城进攻线的核心。《太阳报》称，新赛季由于贝尔纳多-席尔瓦的加盟，大卫-席尔瓦的位置可能会回撤更深一些，不过在瓜帅的体系中，他依然是不可或缺的一环。据悉，大卫-席尔瓦的经纪人已经和曼城就新合同展开了谈判。曼城内部消息透露：“大卫明确表态他喜欢这里，他和教练都非常喜欢上个赛季他的位置和表现，他已经告诉自己的团队希望留队。” "
    # text = "码农对记者说，我今晚一直加班。码农告诉记者，我今晚一直加班。码农对电视台记者说，我今晚一直加班。码农说记者今晚一直加班。他还说今晚他也要加班。记者说码农刚刚说他今晚要一直加班。“明晚也要加班”。"
    # text ="“我今晚要一直加班。”码农说，“明天也要加班”。"

    text = "据巴西《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量。《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”这不是巴西第一次放宽枪支限制。今年1月，博索纳罗上台后第15天就签署了放宽公民持枪的法令。根据该法令，希望拥有枪支的公民须向联邦警察提交申请，通过审核者可以在其住宅内装备最多4把枪支，枪支登记有效期由5年延长到10年。《环球报》称，博索纳罗在1月的电视讲话中称，要让“好人”更容易持有枪支。“人民希望购买武器和弹药，现在我们不能对人民想要的东西说不”。2004年，巴西政府曾颁布禁枪法令，但由于多数民众反对，禁令被次年的全民公投否决。博索纳罗在参加总统竞选时就表示，要进一步放开枪支持有和携带条件。他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”。资料显示，2017年，巴西发生约6.4万起谋杀案，几乎每10万居民中就有31人被杀。是全球除战争地区外最危险的国家之一。不过，“以枪制暴”的政策引发不少争议。巴西《圣保罗页报》称，根据巴西民调机构Datafolha此前发布的一项调查，61%的受访者认为应该禁止持有枪支。巴西应用经济研究所研究员塞奎拉称，枪支供应增加1%，将使谋杀率提高2%。1月底，巴西民众集体向圣保罗联邦法院提出诉讼，质疑博索纳罗签署的放宽枪支管制法令。巴西新闻网站“Exame”称，博索纳罗7日签署的法案同样受到不少批评。公共安全专家萨博称，新的法令扩大了少数人的特权，不利于保护整个社会。（向南）"

    print(text)
    main()
    # # 加载词向量模型
    # model = Word2Vec.load('wiki_and_news_word2vec_200.model')
    # word_dict = model.wv.vocab
    # word_count = model.corpus_count



