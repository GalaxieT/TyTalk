"""
当前版本号： 0.6.3 （未发布）
本版本更新内容：
    更新了拼音改变的规则（得）
    修改了一些参数的值
    调整了等待qps的时间
    修复bugs:
        末尾有空字符导致baidu parser输出错误
        分词错误将符号分到别的词最后，混淆了节奏的短语识别系统（二次修复）

叹号问号等
    问号（疑问句）目的语气 √
    感叹句目的语气
    祈使
根据情感改变
根据语义改变
    句子级语调
        区分疑问句与陈述句 √
        根据词性以及词的构成结构确定词内边界调
        根据语义确定句子调
        句子的下降调附加 √
    停顿
        根据分词确定停顿 √
    重音
        焦点预测与正式度系统（参数综合控制系统）
    综合韵律基础
        韵律描述架构基础 √
        根据语义确定韵律边界
            根据语义确定停顿倾向 √
            根据停顿倾向确定韵律短语边界
            综合字、词、短语信息产生总的韵律预测
词调域
    大略调整 √
字级音调优化
    由严格形状向主要特征转化
        主要工作 √
        算法细节完善
    表达连接性
    细节
        多音字
            还得.
            是不.？
        “一”、“不”的变调
            主要工作 √
            “一”单独成义时不变调（考虑使用注音词典，但是找不到）
        儿化音
            主要工作 √
            一些音节后跟儿化时发生的音变
        广泛的轻声属性
vsqx字间连接性
    爆破音 √
    鼻音声母
    鼻音韵母
    其他声母时长增加以增强特征性和连续性
    解决vel导致的声音提前发出，从而使得参数变化不同步的问题
英文支持
手动调节各层级属性（如声调修正等）
vsqx音量均衡化
多说话人支持
方法：专家规则，之后考虑机器学习
"""
try:
    from reliance import vocalDict
    DCTION = vocalDict.diction()
except ImportError:
    vocalDict = None
    DCTION = {}
try:
    from reliance.baiduCleint import client
    BAIDU_AVALIABLE = True
except ImportError:
    BAIDU_AVALIABLE  = False
    client = lambda x: None
try:
    from reliance import pyCorrect
    PY_CORR = pyCorrect.inf
except ImportError:
    pyCorrect = None
    PY_CORR = []
from pypinyin import pinyin, Style
from reliance import arabToChi
import jieba.posseg
import logging
jieba.setLogLevel(logging.INFO)
import re
import math
import os
from collections import OrderedDict
from time import sleep

class Talker:
    def __init__(self, online: bool=True):
        self.text = []
        self.mark_list = []
        self.note_list = []
        self.total_time = 0
        if BAIDU_AVALIABLE:
            self.default_parser = 'baidu'
        else:
            self.default_parser = 'jieba'
        self.online = online
        # error = os.system("ping www.baidu.com -n 1")
        # if error:
        #     print('网络不通，使用本地模式')
        #     self.default_parser = 'jieba'
        try:
            open('input.txt', 'r').close()
        except FileNotFoundError:
            open('input.txt', 'w', encoding='gbk').close()

    def load(self, cmd, ipt=''):  # text to phonemes
        """
        数据标准化，清洗无用信息，使得每一个字符都有意义，在symbolize()中保证各个model输出的对齐
        阿拉伯数字转化为中文数量
        """
        if self.default_parser == 'baidu' and self.online:
            parser = 'baidu'
        else:
            parser = 'jieba'
        self.text = ''
        self.mark_list.clear()
        self.note_list.clear()
        self.total_time = 0
        command = cmd
        if command == 'type':
            text = ipt
        if command == 'read':
            try:
                with open('input.txt', 'r', encoding='gbk') as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open('input.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
        text = arabToChi.convert(text)
        fil = re.compile(
            r'[^0-9\u4e00-\u9fa5.， ,\-。；%《*》/•、&＆(—)（+）…：？!！“”·\n]+')  # https://www.jianshu.com/p/ebc46dc9c2ee
        text = fil.sub('', text)
        text = re.sub('\n+', '\n', text)
        text = re.sub('[ \n]+$', '', text)
        print('共{}字，开始处理...'.format(len(text)))
        self.text = text
        self.mark(parser)
        self.vocalize()

    def mark(self, parser):
        # 引入数据，处理高级表示成为符号表示，此时各个音节独立拥有自己的属性，不再依赖临近文字
        """处理说明：
        利用pypinyin转化为拼音+标调
        停顿转化为名为‘-’的音节
        分句
        分词？

        结构：text --> [each model (内部数据互通) --> each 字符序列 containing tuples of attributes] --> 字符序列attributes合并
        """
        text = self.text
        text_d = OrderedDict({i: c for i, c in enumerate(text)})  # text key == parse_marks key

        # 分词模块，文本格式化，一切的基础（并不应该是这样）
        # 使用字典来确定同一个字/词对应的所有marks，需要前后顺序信息时，使用列表，否则使用字典。修改时修改字典，在sorted后同步给列表
        parse_marks = []  # [(序号，长度，词性，词序号id)]
        words = []
        if parser == 'jieba':
            word_key = 0
            for x in jieba.posseg.cut(text):
                word = x.word
                flag = x.flag
                ln = len(word)
                no = 0
                for ch in word:
                    no += 1
                    parse_marks.append((no, ln, flag, word_key))
                words.append(word)
                word_key += 1
        elif parser == 'baidu':
            result = client().lexer(text)
            word_key = 0
            for i in result['items']:
                word = i['item']
                flag = i['pos']
                if not flag:
                    flag = 'n'
                if len(word) <= 4:
                    ps = [(word, flag)]
                else:
                    ps = [(pair.word, pair.flag) for pair in jieba.posseg.lcut(word)]
                for p in ps:
                    ln = len(p[0])
                    no = 0
                    for ch in p[0]:
                        no += 1
                        parse_marks.append((no, ln, p[1], word_key, ch))
                    words.append(p[0])
                    word_key += 1
        else:
            raise Exception('No proper parser. ("jieba" or "baidu" only)')
        parse_marks_d = OrderedDict({i: m for i, m in enumerate(parse_marks)})
        words_d = OrderedDict({i: w for i, w in enumerate(words)})

        # 拼音模块
        py_output = pinyin(words, style=Style.TONE3, errors=lambda item: (item,))  # 对无法识别的，返回一个tuple
        py_output = [part[0] for part in py_output]  # 二维列表降维
        insert_count = 0
        for i, py in enumerate(py_output):
            if isinstance(py, tuple) and len(py[0])>1:
                del py_output[i]
                insert_count -= 1
                for k in range(len(py[0])):
                    insert_count += 1
                    py_output.insert(i+insert_count, py[0][k])
        py_output_d = OrderedDict({i: p for i, p in enumerate(py_output)})

        # 自动标注 儿化音；采用以text中的chrc为基础单位的遍历
        def er_del(idx):
            p0 = parse_marks_d[idx][0]

            del text_d[idx]
            text = ''.join([text_d[k] for k in sorted(text_d)])

            del py_output_d[idx]
            py_output = [py_output_d[k] for k in sorted(py_output_d)]

            wk = parse_marks_d[idx][3]  # word key
            words_d[wk] = words_d[wk][:p0-1] + words_d[wk][p0:]
            words = [words_d[k] for k in sorted(words_d) if words_d[k]]

            del parse_marks_d[idx]
            for k in parse_marks_d:
                pm = parse_marks_d[k]
                if pm[3] == wk:
                    new_i = pm[0]  # 新的词内idx
                    new_l = pm[1] - 1
                    if new_i > p0:
                        new_i = pm[0] - 1
                    l = list(pm)
                    l[0] = new_i
                    l[1] = new_l
                    parse_marks_d[k] = tuple(l)
            parse_marks = [parse_marks_d[k] for k in sorted(parse_marks_d)]
            return text, text_d, py_output, py_output_d, words, words_d, parse_marks, parse_marks_d
        jump = ['男']  # 此列表内的字之后的“儿”不会儿化
        er_inf = []
        del_count = 0
        flag_filter = ()
        if parser == 'jieba':
            flag_filter = ('j', 'l', 'm', 'Ng', 'n', 'nr', 'ns', 'nt', 'nz', 'o', 'q', 'r', 's', 'tg', 't')
        if parser == 'baidu':
            flag_filter = ('n', 'f', 's', 't', 'nr', 'ns', 'nt', 'nw', 'nz', 'm', 'q', 'r', 'vn', 'an')
        for idx, chrc in enumerate(text):
            current_idx = idx - del_count
            p0 = parse_marks_d[idx][0]
            p1 = parse_marks_d[idx][1]

            if chrc == '儿' and current_idx != 0 and text[current_idx-1] not in jump:
                flag = ''
                if p1 == 1:  # 单一个“儿”词
                    flag = parse_marks[current_idx - 1][2]
                elif p0 != 1 and p1 > 1:  # 不是词内第一个字
                    word_seg = text[current_idx-p0+1 : current_idx]
                    flag = jieba.posseg.lcut(word_seg)[-1].flag
                    if flag not in flag_filter:
                        if parser == 'baidu':  # 可以考虑递归一个函数
                            try:
                                r = client().lexer(word_seg)
                                flag = r['items'][-1]['pos']
                            except KeyError:
                                sleep(0.6)
                                print('等待QPS，延时0.6秒...')
                                try:
                                    r = client().lexer(word_seg)
                                    flag = r['items'][-1]['pos']
                                except KeyError:
                                    print('已跳过')
                if flag in flag_filter:
                    er_inf[-1] = True
                    text, text_d, py_output, py_output_d, words, words_d, parse_marks, parse_marks_d = er_del(idx)
                    del_count += 1
                else:
                    er_inf.append(False)
            else:
                er_inf.append(False)

        # 自动标注 “一”、“不”变调
        # “一”规则过于宽泛，没有限制“一”单独成词的情况，如“不一而足”，但“一”在词尾的情况已经排除
        # ☆标记需要进行变调的“一”
        chr_index = 0
        rule1 = re.compile('^[^☆零一二三四五六七八九十第]*一[^☆零一二三四五六七八九十]')  # 有一点逻辑漏洞，暂无伤大雅
        for word in words:
            while rule1.search(word):  # 每一循环做一个标记
                word = rule1.sub(lambda match: match.group()[:-2] + '☆' + match.group(0)[-1], word)
            for i, c in enumerate(word):
                if c == '☆' and py_output[chr_index + i] == 'yi1':
                    if py_output[chr_index + i + 1][-1] in '123':
                        py_output[chr_index + i] = 'yi4'
                    elif py_output[chr_index + i + 1][-1] in '4aoeiuvngr':
                        py_output[chr_index + i] = 'yi2'
                if c == '不' and py_output[chr_index + i] == 'bu4':
                    try:
                        if py_output[chr_index + i + 1][-1] == '4':  # 同时排除后面是符号的情况
                            py_output[chr_index + i] = 'bu2'
                    except IndexError:
                        pass
            chr_index += len(word)


        rule = PY_CORR  # [(index, py, er)] 前面的规则会被后面的覆盖
        modify = []  # [(index, pinyin, er)]
        chr_index = 0
        for i, word in enumerate(words):
            if len(word) == 2 and word[0] == word[1] and parse_marks[chr_index][2] == 'n':  # 名叠词后轻声
                modify.append((chr_index+1, ''.join([x for x in py_output[chr_index+1] if x not in '1234']), None))
            for i_w, chrc in enumerate(word):
                if chrc in '种中' and word not in ('中断', '集中'):
                    flag = parse_marks[i_w+chr_index][2]
                    if flag == 'v':
                        modify.append((i_w + chr_index, 'zhong4', None))
                if chrc == '得':
                    flag = parse_marks[i_w+chr_index][2]
                    if flag in ('u', 'ud'):
                        modify.append((i_w + chr_index, 'de', None))
                if chrc == '处' and len(word) <= 2:
                    flag = parse_marks[i_w + chr_index][2]
                    if flag == 'v':
                        modify.append((i_w + chr_index, 'chu3', None))
            for item in rule:
                if item[0] in word:
                    for inf in item[1]:
                        if inf[2]:
                            if len(item[0]) > 1:
                                modify.append(((word.index(item[0]) + inf[0] + chr_index), inf[1], inf[2]))
                            else:
                                modify.append(((word.index(item[0]) + inf[0] + chr_index), inf[1], None))
                                print('有儿化规则不合理')
                        else:
                            modify.append(((word.index(item[0]) + inf[0] + chr_index), inf[1], inf[2]))
            chr_index += len(word)
        for mod in modify:
            if mod[1]:
                py_output[mod[0]] = mod[1]
        del_list = []
        for mod in modify:
            if mod[2]:
                if text[mod[0]+1:] and text[mod[0]+1] == '儿':
                    er_inf[mod[0]] = True
                    del_list.append(mod[0]+1)
                else:
                    print('有儿化规则失效')
                if mod[2] == False:  # 用于取消儿化，与None 相区别
                    pass
        for i in del_list:
            text, text_d, py_output, py_output_d, words, words_d, parse_marks, parse_marks_d = er_del(i)

        py_marks = []  # (拼音名，声调)
        for py in py_output:
            if isinstance(py, str):
                if py[-1] in '1234':
                    py_marks.append((py[:-1], py[-1]))
                else:  # 轻声
                    py_marks.append((py, '0'))
            else:  # 停顿长的优先检测
                py_marks.append(('sil', py[0]))
        # 结合儿化信息和其他音变
        for i in range(len(py_marks)):
            py_marks[i] = py_marks[i] + (er_inf[i],)

        # 分句模块
        sent_marks = []  # [(序号，长度，目的语气)]
        chr_index = 0
        l = len(words)
        for i, word in enumerate(words):
            sent_goal = 'dcl'  # 陈述dcl, 疑问qst, 感叹exc, 祈使imp
            chr_index += len(word)
            if re.match('[,，。；（）()？！?!…—]+', word) or '\n' in word or i + 1 == l:
                try:  # 主要依靠标点
                    if re.match('[？?]+', word) or (words[i - 1]) in '吗么':
                        sent_goal = 'qst'
                    if re.match('[！!]+', word):
                        sent_goal = 'exc'
                except IndexError:
                    pass
                for i in range(chr_index):
                    sent_marks.append((i + 1, chr_index - 1, sent_goal))
                chr_index = 0
        sent_marks_d = OrderedDict([(key, sent_marks[i]) for i, key in enumerate(parse_marks_d)])

        # 韵律层级模块，（语法词）→韵律词（单位）→韵律短语（意群）→语调（最高）。韵律词：声明一个预设变量；韵律短语：限定一个变量；语调短语：表述一个命题
        # 有网情况下：百度分词+语法词附着=语法词到韵律词、长词jieba分词+附着得到韵律词；无网情况下：jieba分词，附着得到韵律词
        # 语法规则得到韵律短语
        # 预测焦点，合成语调短语
        # 基本思想：名词和动词构成主体，其他成分附着
        # 时长：先确定节拍大致长度，后根据一般规律修正长度（可以放在vocalize里面）
        def to_marks(bag):

            start_length = 5
            dep_analyzed = False
            discontinuity = {}
            average_disc = None
            # 利用句法信息划分停顿等级
            items = None
            if len(bag) >= start_length and self.online and BAIDU_AVALIABLE:
                words_bag = [words_d[inf[1]] for inf in bag]
                try:
                    sent_result = client().depParser(''.join(words_bag))
                    items = sent_result['items']
                except KeyError:
                    sleep(0.6)
                    print('等待句法QPS，延时0.6秒...')
                    try:
                        sent_result = client().depParser(''.join(words_bag))
                        items = sent_result['items']
                    except KeyError:
                        print('已跳过句法分析')
            if items is not None:
                dep_analyzed = True
                word_i = 0
                word_i_bag = []
                for word in words_bag:
                    word_i_bag.append(word_i)
                    word_i += len(word)
                items_d = {item['id']: item for item in items}
                children = {id_: [] for id_ in items_d}
                for id_ in items_d:
                    try:
                        children[items_d[id_]['head']].append(id_)
                    except KeyError:  # 根词
                        pass
                words_result = [item['word'] for item in items]
                word_i = 0
                word_i_result = []  # 依存句法进行的分词中，词首字的index
                for word in words_result:
                    word_i_result.append(word_i)
                    word_i += len(word)
                differences = []  # (start_i, end_i)
                if word_i_bag != word_i_result:
                    first_differ = None
                    continual_common = False
                    last_common = 0
                    for i_c in range(sum([x[0] for x in bag])):
                        in_bag = i_c in word_i_bag
                        in_result = i_c in word_i_result
                        if in_bag and in_result:
                            if continual_common and first_differ is not None:
                                differences.append((first_differ, last_common-1))
                                first_differ = None
                            last_common = i_c
                            continual_common = True
                        elif in_bag or in_result and not first_differ:
                            first_differ = last_common
                            continual_common = False
                        end = i_c
                    if first_differ:
                        differences.append((first_differ, end))

                utterred = {id_: False for id_ in items_d}
                disconnected = {id_: False for id_ in items_d}

                def get_children_num(id_, out=None):
                    """
                    :return: [expressed_num, inexpressed_num]
                    """
                    if not out:
                        out = [0, 0]
                    if utterred[id_] and not disconnected[id_]:
                        out[0] += 1
                    else:
                        out[1] += 1
                    rely_ids = children[id_]
                    for rely_id in rely_ids:
                        get_children_num(rely_id, out)
                    return out

                def get_distance(ia, ib, default=2):
                    """
                    求a到b的距离（b-a，距离逆箭头方向增加，即箭头方向为距离的负方向）
                    :return 带正负号的距离，若不在同一树，则返回默认距离
                    """

                    def get_children(id, out=None, cnt=0):
                        if not out:
                            out = []
                        cnt += 1
                        direct = children[id]
                        for child in direct:
                            out.append((child, cnt))
                            get_children(child, out, cnt)
                        return out

                    def get_masters(id, out=None, cnt=0):
                        if id == 0:
                            return out
                        if not out:
                            out = []
                        cnt += 1
                        direct = items_d[id]['head']
                        out.append((direct, cnt))
                        get_masters(direct, out, cnt)
                        return out

                    dist = 0
                    ia_inf = dict(get_children(ia))
                    ib_inf = dict(get_children(ib))
                    if ia in ib_inf:
                        dist = -ib_inf[ia]
                    elif ib in ia_inf:
                        dist = ia_inf[ib]
                    else:  # 并行修饰情况
                        m_ia_inf = get_masters(ia)  # 用list保持顺序
                        m_ia_list = [inf[0] for inf in m_ia_inf]
                        m_ib_inf = get_masters(ib)
                        m_ib_inf_d = dict(m_ib_inf)
                        for mst_b in m_ib_inf:
                            if mst_b[0] in m_ia_list:
                                dist = - m_ib_inf_d[mst_b[0]]
                                break
                    # print(items_d[ia], dist)
                    return dist

                # 按照语序，渲染，以依赖链接为单位

                def get_discontinuity(fa, fb, fl, i):
                    def sigmoid(x, ex=1):
                        return 1 / (1 + math.exp(-x * ex))

                    ib = i + 1

                    if items_d[ib]['deprel'] in ('DE', 'DI', 'DEI'):
                        fa = 1

                    if fl > 0:
                        fl = fl / 3
                    else:
                        fl = - fl
                    return sigmoid(1.5*(3*math.log(fa, 20) + 0.4*math.log(sigmoid((-1 / fl) + 0.5*math.log(fb), 2))))

                relation_inf = {}  # {id:(A_inf, B_inf, distance)}
                for id_ in list(items_d)[:-1]:
                    utterred[id_] = True
                    relation_inf.update(
                        {id_: (get_children_num(id_)[0], get_children_num(id_ + 1)[1], get_distance(id_, id_ + 1))})
                    inf = relation_inf[id_]
                    dis = round(get_discontinuity(inf[0], inf[1], inf[2], id_), 2)
                    discontinuity.update({id_: dis})
                    if dis > 0.7:
                        for i in range(1, id_ + 1):
                            disconnected[i] = True
                try:
                    average_disc = round(sum(discontinuity.values())/len(discontinuity), 2)
                except ZeroDivisionError:  # 这里偶尔会报一个错，因为会把一个习语按照一个词来分词
                    average_disc = 0.5  # ?

            # 模拟从句首开始构建
            max_chrc = 3
            func_tempo = 0.8
            result = []
            units = []
            unit = []
            unit_l = 0
            end = len(bag) - 1
            for i, word_inf in enumerate(bag):
                if word_inf[0] + unit_l <= max_chrc:
                    unit.append(word_inf)
                    unit_l += word_inf[0]
                else:
                    units.append(unit)
                    unit = [word_inf]
                    unit_l = word_inf[0]
                if i == end:
                    units.append(unit)
            chr_i = 0
            for unit in units:
                u_counter = 0
                s = sum([wi[0] for wi in unit])
                for word_in in unit:
                    for n in range(word_in[0]):
                        u_counter += 1
                        if dep_analyzed and chr_i+1 in word_i_result:
                            disc = discontinuity.get(word_i_result.index(chr_i+1), 0)
                        else:
                            disc = None
                        result.append(((u_counter, s), (disc, average_disc, dep_analyzed)))
                        chr_i += 1
            return result
        proso_marks = []  # [((韵律词内序号,韵律词长), (韵律短语内词序号，韵律短语长))]
        bags = []
        bag = []  # [(词长，词性, 词id)]
        end_marks = len(parse_marks_d) - 1
        insert_l = []
        for i_marks, id_ in enumerate(parse_marks_d):
            mark = parse_marks_d[id_]
            if mark[0] == mark[1]:
                py = py_output_d[id_]
                if isinstance(py, tuple):
                    if py[0] in '\n\r':
                        insert_l.append(((1, 1), (None, None, False)))
                    elif py[0] in '。！？!?…':
                        insert_l.append(((1, 1), (None, None, False)))
                    elif py[0] in '，,;；…—':
                        insert_l.append(((1, 1), (None, None, False)))
                    else:
                        insert_l.append(((1, 1), (None, None, False)))
                    if mark[1] > 1:  # 修bug：一个符号被分词成了一个词的结尾;方法：把[:-1]范围内的所有字符加入bag
                        last_chr_id = list(parse_marks_d.keys())[i_marks-1]
                        for n in range(mark[1]-1):
                            sub_mark_id = list(parse_marks_d.keys())[i_marks-mark[1]+1+n]
                            sub_mark = parse_marks_d[sub_mark_id]
                            bag.append((sub_mark[1], sub_mark[3], last_chr_id))
                    bags.append(bag)
                    bag = []

                else:
                    bag.append((mark[1], mark[3], id_))  # 词长，词id，末位字id
                    if i_marks == end_marks:  # 文末标记+一般bag间标记可能共存，导致bag末尾有两个insert
                        insert_l.append(None)
                        bags.append(bag)
                        bag = []
        for i, bag in enumerate(bags):
            result = to_marks(bag)
            insert = [insert_l[i]]
            if not insert[0]:  # 排除文本结尾处的insert
                insert = []
            final = result + insert
            for tempo in final:
                proso_marks.append(tempo)

        proso_marks_d = OrderedDict([(key, proso_marks[i]) for i, key in enumerate(parse_marks_d)])

        # # 计算重要性
        # for

        for i in range(len(parse_marks)):
            self.mark_list.append({'py': py_marks[i], 'par': parse_marks[i], 'sent': sent_marks[i], 'proso': proso_marks[i]})



    def vocalize(self):
        # 从符号成为波形
        """算法说明：
        分四声，轻声视为四声
        逗号处理为小停顿，句号处理为大停顿
        每个音有目标形态，各个目标形态相互叠加
        """
        # 全局属性
        standard_length = 210  # ms 240 阅读 200 chatting 300 朗诵
        step = 2  # ms

        # 全局节奏属性
        er_per = 1.3  # 儿化音时长比例
        sent_prolong = 1.05

        # 参数形成，序列：每个值是一个采样点
        # 每个phone不再是独立渲染了，而是建立在全局共同渲染的基础上

        # note:[py, start, length, pitch, intns]
        for mark in self.mark_list:
            self.note_list.append([(mark['py'][0], mark['py'][2])])

        # 对时长的渲染，以单字为单位循环的架构有点问题，和其他架构并不平行

        # 参数
        ms_counter = 0
        std_tempo_unit = 2
        four_char_tempo = 3
        func_tempo_weight = 0.8
        tempo_range = (0.6, 1.5)
        extreme = 2  # 时长极化程度

        disc_per = 0.1  # 不连续性时间占原始时间长度
        disc_per_analyzed = 0.25

        t_range = tempo_range[1] - tempo_range[0]
        t_center = (tempo_range[1] + tempo_range[0]) / 2

        # 声明相对全局变量
        unit_i_lists = []
        chr_tempos = []
        for i, mark in enumerate(self.mark_list):
            proso_mark = mark['proso']
            if proso_mark[0][0] == proso_mark[0][1]:
                unit_i_lists.append([x for x in range(i-proso_mark[0][1]+1, i+1)])
        for unit_i_list in unit_i_lists:
            word_list = []
            for i in unit_i_list:
                if self.mark_list[i]['par'][0] == self.mark_list[i]['par'][1]:
                    word_list.append(([x for x in range(i - self.mark_list[i]['par'][1] +1, i+1)], self.mark_list[i]['par'][2]))
            # unit内语法词间节奏比例
            word_tempos = []
            weights = []
            for word_inf in word_list:
                if word_inf[1] in ('d', 'm', 'q', 'p', 'u', 'xc', 'dg', 'uj'):
                    weight = func_tempo_weight
                else:
                    weight = 1
                weights.append(weight)
            s = sum(weights)
            for weight in weights:
                word_tempos.append((weight/s)*std_tempo_unit)
            # 词内字节奏比例
            chr_tempos_u = []
            for i, word_inf in enumerate(word_list):
                total_tempo = word_tempos[i]
                chrs = []
                l = len(word_inf[0])
                if l == 1:
                    weights_c = (1,)
                elif l == 2:
                    weights_c = (1, 0.5)
                elif l == 3:
                    weights_c = (1, 0.8, 1.1)
                elif l == 4:
                    total_tempo = four_char_tempo
                    weights_c = (1, 0.8, 1, 0.8)
                else:
                    weights_c = tuple(1 for n in range(l))
                s_c = sum(weights_c)
                for weight_c in weights_c:
                    tempo_c = weight_c/s_c * total_tempo
                    tempo_c = (1 / (1 + math.exp(-(tempo_c-t_center)*extreme / t_range)))*t_range + tempo_range[0] # sigmoid
                    chr_tempos_u.append(tempo_c)
            chr_tempos = chr_tempos + chr_tempos_u
        for i, chr_tempo in enumerate(chr_tempos):
            mark = self.mark_list[i]
            py = mark['py']
            if py[0] == 'sil':
                if py[1] in '\n\r':
                    length = standard_length*2
                elif py[1] in '。！？!?…':
                    length = standard_length*2
                elif py[1] in '，,;；…—':
                    length = standard_length*1.5
                else:
                    length = standard_length*0.5
            else:
                length = standard_length * chr_tempo
                if py[1] == '0':
                    length = length * 0.6
                if py[2]:
                    length = length * er_per
                if mark['sent'][0] == mark['sent'][1]:
                    length = length * sent_prolong

            # 临时插入基于句法分析的停顿（韵律词组的前身依据）
            if mark['par'][0] == mark['par'][1]:
                disc = length * disc_per
            else:
                disc = 0
            if mark['proso'][1][2] and mark['proso'][1][0] is not None:
                disc = length * disc_per_analyzed
                disc *= (mark['proso'][1][0] / mark['proso'][1][1])

            length, disc = round(length), round(disc)
            self.note_list[i].append(ms_counter)
            self.note_list[i].append(length)
            ms_counter += length + disc
        self.total_time = ms_counter

        # 对音高的渲染
        # 从固定值的点，变成核心调形。
        # 默认调是基调：0
        tone0 = [(0, 0), (1, -0.5)]
        tone1 = [(0, 0.6), (1, 0.6)]
        tone2 = [(0, -0.6), (1, 0.6)]
        tone3 = [(0, -0.6), (0.5, -1.2), (1, -0.6)]
        tone4 = [(0, 1), (1, -1)]

        def to_point_track():
            def get_inter(p1, p2, x):  # 得到关键点列表中任意时刻对应的值
                k = (p2[1] - p1[1]) / (p2[0] - p1[0])
                return (x - p1[0]) * k + p1[1]

            def tone_constrain(ori):
                limit = 2  # 倍原有range
                return (1 / (1 + math.exp(-ori * 1.2 / (limit * tone_range))) - 0.5) * limit * tone_range

            # 全局参数列表
            std_tone = 0
            std_tone_range = 3200
            pre_per = 0.2  # preparation time percent
            fastest_change = 10
            preset_per = 0.7  # 为后一音节的相对特征做准备的开始时间
            preset_aim = 0.5  # 为后做准备的程度 局部绝对值，需要改成全局相对值
            preset_aim_per = 0.8  # 适应后面发音的程度，调高会使波动变平
            sent_downward_per = 0.25  # 语调下倾程度

            goal_base_per = 1  # 目的语气音高增加量占range比例
            goal_range_per = 0.4  # 目的语气幅度增加量占range比例
            goal_cover = 3  # 目的语气覆盖最后几个音节

            tone_reset = (0.2, 0.5)  # 韵律词、韵律短语的音高重置比例

            tone_adjust_per = 0.2  # 语法词内字音高和range调整比例

            timer = 0
            track = []  # [keyframe[time, pitch]]
            for idx, mark in enumerate(self.mark_list):
                # 局部参数
                total_length = self.note_list[idx][2]
                start = self.note_list[idx][1]
                base_tone = std_tone
                tone_range = std_tone_range
                pre_length = pre_per * total_length
                core_length = total_length - pre_length
                point_list = []
                tone_adjust = 0  # 词内，正相关与base和range的同步正相关变化
                basic_points = [(0, 0), (1, 0)]
                if mark['py'][1] == '1':
                    basic_points = tone1[:]  # 核心调形
                if mark['py'][1] == '2':
                    basic_points = tone2[:]  # 核心调形
                if mark['py'][1] == '3':
                    basic_points = tone3[:]  # 核心调形
                if mark['py'][1] == '4':
                    basic_points = tone4[:]  # 核心调形
                if mark['py'][1] == '0':
                    basic_points = tone0[:]  # 核心调形

                # 为了配合连去变声，将这一模块提前
                try:
                    mark_nxt = self.mark_list[idx + 1]
                    start_nxt = self.note_list[idx + 1][1]
                    core_length += start_nxt - start - total_length  # 补偿discontinue，这些时间也视为前一音节的时长
                    if mark_nxt['py'][1] == '3':
                        if mark['py'][1] == '3':  # 无奈之举，将三声拉出来，确实是特殊情况
                            basic_points = [(0, -0.6), (1, 0.6)]
                            for i, p in enumerate(basic_points):
                                preset_per_3 = 0.1
                                preset_aim_3 = 0.6
                                if p[0] >= preset_per_3:
                                    new_p = (preset_per_3, get_inter(basic_points[i - 1], p, preset_per_3))
                                    basic_points = basic_points[:i] + [new_p, (1, preset_aim_3)]
                                    break
                        else:
                            for i, p in enumerate(basic_points):
                                if p[1] >= preset_aim:  # 已经够高了
                                    break
                                if p[0] >= preset_per:
                                    new_p = (preset_per, get_inter(basic_points[i - 1], p, preset_per))
                                    basic_points = basic_points[:i] + [new_p, (1, preset_aim)]
                                    break
                    elif mark_nxt['py'][1] in '0124':  # 绝对特征要适应性变化
                        bp_next = [(0, 0)]
                        for i, p in enumerate(basic_points):
                            if p[0] > preset_per:
                                new_p = (preset_per, get_inter(basic_points[i - 1], p, preset_per))
                                if mark_nxt['py'][1] == '1':
                                    bp_next = tone1[:]  # 核心调形
                                if mark_nxt['py'][1] == '2':
                                    bp_next = tone2[:]  # 核心调形
                                if mark_nxt['py'][1] == '4':
                                    bp_next = tone4[:]  # 核心调形
                                if mark_nxt['py'][1] == '0':
                                    bp_next = tone0[:]  # 核心调形
                                basic_points = basic_points[:i] + \
                                               [new_p, (1, new_p[1] - (new_p[1] - bp_next[0][1]) * preset_aim_per)]
                                break
                    elif mark_nxt['py'][0] == 'sil':
                        pass
                except IndexError:
                    # copy
                    bp_next = [(0, 0)]
                    if mark['py'][1] == '3':
                        for i, p in enumerate(basic_points):
                            if p[0] > preset_per:
                                new_p = (preset_per, get_inter(basic_points[i - 1], p, preset_per))
                                basic_points = basic_points[:i] + \
                                               [new_p, (1, new_p[1] - (new_p[1] - bp_next[0][1]) * preset_aim_per)]
                                break
                # 覆盖性改调。需要重构成求和型
                # base_tone 控制，加减（外加handle文本起始情况）
                bt_ctrl = []
                try:
                    before = track[idx - 1]
                    point_list.append(before[-1])
                    if self.mark_list[idx - 1]['sent'][0] < self.mark_list[idx - 1]['sent'][1]:
                        point_list.append(before[-1])
                    slope = (before[-1][1] - std_tone * basic_points[0][1]) / pre_length  # pit/ms
                    if abs(slope) > fastest_change and mark['sent'][0] != 1:  # 留在那个高度
                        bt_ctrl.append((before[-1][1] + slope / abs(slope) * fastest_change * pre_length)
                                       - basic_points[0][1] * tone_range - base_tone)
                except IndexError:  # 文本起始处
                    pass

                bt_ctrl.append(- sent_downward_per * tone_range * (mark['sent'][0] - 1))


                # 基于句子结构的语调修改，同时改变了base_tone和range
                tr_ctrl = []
                sent_dist = mark['sent'][1] - mark['sent'][0]
                if mark['sent'][2] == 'qst' and -1 < sent_dist < goal_cover:
                    bt_ctrl.append(goal_base_per * tone_range * (1 - sent_dist / goal_cover))
                    tr_ctrl.append(goal_range_per)
                # 基于词内重要程度的语调修改（词调域），可以与时长渲染的对应部分合并，以成为正式度参数
                word_l =  mark['par'][1]
                if word_l == 2:
                    if mark['par'][0] == word_l:
                        tone_adjust = -1
                if word_l == 3:
                    if mark['par'][0] == word_l - 2:
                        tone_adjust = 1
                    if mark['par'][0] == word_l - 1:
                        tone_adjust = -0.5
                    if mark['par'][0] == word_l:
                        tone_adjust = -1
                if word_l == 4:
                    if mark['par'][0] == word_l - 2:
                        tone_adjust = -0.5
                    if mark['par'][0] == word_l:
                        tone_adjust = -0.5

                if tone_adjust:
                    bt_ctrl.append(tone_adjust*tone_adjust_per*tone_range)
                    tr_ctrl.append((1-tone_adjust_per)**(-tone_adjust)-1)



                base_tone = tone_constrain(base_tone + sum(bt_ctrl))
                for ctrl in tr_ctrl:
                    tone_range *= 1 + ctrl
                # 韵律词级音高重置
                try:
                    if mark['py'][0] != 'sil' and mark['proso'][0][1] > 1 and mark['proso'][0][0] == 1 and base_tone < std_tone:
                        base_tone += (std_tone - base_tone) * tone_reset[0]
                except IndexError:
                    print(mark)
                    exit()
                # 短停顿级音高重置，参考韵律短语重置程度
                if mark['py'][0] == 'sil' and mark['sent'][0] <= mark['sent'][1] and base_tone < std_tone:
                    base_tone += (std_tone - base_tone) * tone_reset[1]


                if point_list:
                    pre_length_l = pre_length
                    core_length_l = core_length
                else:  # 一句话开头
                    pre_length_l = 0
                    core_length_l = total_length
                for point in basic_points:  # 转化为基于时长，而不是比例
                    point_list.append(
                        (round(start + pre_length_l + point[0] * core_length_l), base_tone + point[1] * tone_range))
                track.append(point_list)
            return track

        # 音高渲染
        trck = to_point_track()
        for i, chrc in enumerate(trck):
            # print(self.mark_list[i]['py'], chrc)
            tone_list = []
            for ind, point in enumerate(chrc[:-1]):
                next_point = chrc[ind + 1]
                start_time = point[0]
                end_time = next_point[0]
                step_num = round((end_time - start_time) / step)
                if step_num == 0:
                    continue
                start_tone = point[1]
                end_tone = next_point[1]
                tone_step = (end_tone - start_tone) / step_num
                for s in range(step_num):
                    tone_list.append((start_time + step * s, start_tone + tone_step * s))
            self.note_list[i].append(tone_list)

        # 对响度的渲染
        i_points_track = []
        for i, mark in enumerate(self.mark_list):
            start_time = self.note_list[i][1]
            length = self.note_list[i][2]
            try:
                points_before = i_points_track[i-1]
                i_points_start = [(0, points_before[-1][1])]
            except IndexError:
                i_points_start = [(0, 1)]  # 响度，0~1
            if mark['py'][1] == '0':
                i_points = i_points_start + [(0.2, 0.4), (0.7, 0.4), (1, 0.8)]
            elif mark['py'][2]:
                i_points = i_points_start + [(0.3, 1), (0.75, 0.7), (1, 0.2)]
            else:
                i_points = i_points_start + [(0.1 ,1)]
            if len(i_points)>1:
                intns_list = []
                for ind, point in enumerate(i_points[:-1]):
                    next_point = i_points[ind + 1]
                    start_time_point = start_time + point[0] * length
                    end_time_point = start_time + next_point[0] * length
                    step_num = round((end_time_point - start_time_point) / step)
                    if step_num == 0:
                        continue
                    start_tone = point[1]
                    end_tone = next_point[1]
                    intns_step = (end_tone - start_tone) / step_num
                    for s in range(step_num):
                        intns_list.append((start_time_point + step * s, start_tone + intns_step * s))
            else:
                intns_list = [(i_points[0][0], i_points[0][1])]
            i_points_track.append(i_points)
            self.note_list[i].append(intns_list)

    def to_vsqx(self, addr='', start=True):  # phonemes to vsqx
        note_list = self.note_list[:]
        pre = 5000
        total_time = self.total_time + 100 + pre
        f0_level = 63
        pbs = 13

        # 牺牲性能，生成接口：（拼音，起始时间，时长，pit坐标序列(其实是独立的），响度坐标序列（独立））
        # 土法炼成xml
        text_insert = ''
        text_insert += '\r\t\t\t<cc><t>{}</t><v id="P">{}</v></cc>'.format(0, round(note_list[0][3][0][1]))
        # 这个for循环占用了最长的时间
        total = len(note_list)
        counter = 0
        to_print = [0, 0.25, 0.50, 0.75, 1.00]
        for note in note_list[:]:
            counter += 1
            for pit in note[3]:
                p = pit[1]
                if p < -8190:
                    p = -8190
                text_insert += '\r\t\t\t<cc><t>{}</t><v id="P">{}</v></cc>'.format(round(pit[0] + pre), round(p))
            for intns in note[4]:
                if 'ang' in note[0]:
                    intns = (intns[0], intns[1] * 0.6)
                text_insert += '\r\t\t\t<cc><t>{}</t><v id="D">{}</v></cc>'.format(round(intns[0] + pre),
                                                                                   round(intns[1] * 64))
            if note[0][0] == 'sil':  # 调整间隔note，但参数不取消
                note_list.remove(note)
            percent = round(counter/total, 2)
            if percent in to_print:
                print('关键循环完成度：', int(percent*100), '%')
                to_print.remove(percent)
        last_ph = ''
        for i, note in enumerate(note_list):
            try:
                next_py = note_list[i+1][0][0]
            except IndexError:
                next_py = ''
            py = note[0][0]
            er_ph = ''
            nasal_ph = ''
            if py == 'yo':
                py = 'you'
            if py == 'eng' or note[0] == 'n':
                py = 'en'
            vel = 64
            if py[0] in 'bdg':
                vel = 115
            if last_ph:
                if last_ph[-2:] == '@`':
                    vel = 110
            if py[:2] in ('sh', 'ch') or py[0] in ('q', 'x', 's', 't', 'k','p'):
                vel = 40
            if False and py[-1] == 'g' and note[2] < 200 and next_py[:1] not in ('aoeyw'):  #效果不好暂时取消
                nasal_ph = ' n'
            if note[0][1]:
                er_ph = ' @`'
            try:
                dict_ph = DCTION[py]
            except KeyError:
                dict_ph = 'unknown'
                er_ph = ''
            last_ph = dict_ph+er_ph
            text_insert += """
\t\t\t<note>
\t\t\t\t<t>{start}</t>
\t\t\t\t<dur>{dur}</dur>
\t\t\t\t<n>{f0}</n>
\t\t\t\t<v>{vel}</v>
\t\t\t\t<y><![CDATA[{py}]]></y>
\t\t\t\t<p><![CDATA[{ph}]]></p>
\t\t\t\t<nStyle>
\t\t\t\t\t<v id="accent">50</v>
\t\t\t\t\t<v id="bendDep">0</v>
\t\t\t\t\t<v id="bendLen">0</v>
\t\t\t\t\t<v id="decay">50</v>
\t\t\t\t\t<v id="fallPort">0</v>
\t\t\t\t\t<v id="opening">127</v>
\t\t\t\t\t<v id="risePort">0</v>
\t\t\t\t\t<v id="vibLen">0</v>
\t\t\t\t\t<v id="vibType">0</v>
\t\t\t\t</nStyle>
\t\t\t</note>""".format(start=note[1] + pre, dur=note[2], f0=f0_level, py=py, ph=dict_ph+er_ph+nasal_ph, vel=vel)

        address = addr
        text_head = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<vsq4 xmlns="http://www.yamaha.co.jp/vocaloid/schema/vsq4/"
      xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:schemaLocation="http://www.yamaha.co.jp/vocaloid/schema/vsq4/ vsq4.xsd">
\t<vender><![CDATA[Yamaha corporation]]></vender>
\t<version><![CDATA[4.0.0.3]]></version>
\t<vVoiceTable>
\t\t<vVoice>
\t\t\t<bs>4</bs>
\t\t\t<pc>0</pc>
\t\t\t<id><![CDATA[BK8H76TAEHXWSKDB]]></id>
\t\t\t<name><![CDATA[Luotianyi_CHN_Meng]]></name>
\t\t\t<vPrm>
\t\t\t\t<bre>0</bre>
\t\t\t\t<bri>0</bri>
\t\t\t\t<cle>0</cle>
\t\t\t\t<gen>0</gen>
\t\t\t\t<ope>0</ope>
\t\t\t</vPrm>
\t\t</vVoice>
\t</vVoiceTable>
\t<mixer>
\t\t<masterUnit>
\t\t\t<oDev>0</oDev>
\t\t\t<rLvl>0</rLvl>
\t\t\t<vol>0</vol>
\t\t</masterUnit>
\t\t<vsUnit>
\t\t\t<tNo>0</tNo>
\t\t\t<iGin>0</iGin>
\t\t\t<sLvl>-898</sLvl>
\t\t\t<sEnable>0</sEnable>
\t\t\t<m>0</m>
\t\t\t<s>0</s>
\t\t\t<pan>64</pan>
\t\t\t<vol>0</vol>
\t\t</vsUnit>
\t\t<monoUnit>
\t\t\t<iGin>0</iGin>
\t\t\t<sLvl>-898</sLvl>
\t\t\t<sEnable>0</sEnable>
\t\t\t<m>0</m>
\t\t\t<s>0</s>
\t\t\t<pan>64</pan>
\t\t\t<vol>0</vol>
\t\t</monoUnit>
\t\t<stUnit>
\t\t\t<iGin>0</iGin>
\t\t\t<m>0</m>
\t\t\t<s>0</s>
\t\t\t<vol>-129</vol>
\t\t</stUnit>
\t</mixer>
\t<masterTrack>
\t\t<seqName><![CDATA[Untitled2]]></seqName>
\t\t<comment><![CDATA[New VSQ File]]></comment>
\t\t<resolution>480</resolution>
\t\t<preMeasure>4</preMeasure>
\t\t<timeSig><m>0</m><nu>4</nu><de>4</de></timeSig>
\t\t<tempo><t>0</t><v>12000</v></tempo>
\t</masterTrack>
\t<vsTrack>
\t\t<tNo>0</tNo>
\t\t<name><![CDATA[Track]]></name>
\t\t<comment><![CDATA[Track]]></comment>
\t\t<vsPart>"""  # 不含到下一段的换行符
        text_length = """
\t\t\t<t>7680</t>
\t\t\t<playTime>{}</playTime>""".format(total_time)  # <t>起始时间，序列不能早于7680
        text_mid = """
\t\t\t<name><![CDATA[NewPart]]></name>
\t\t\t<comment><![CDATA[New Musical Part]]></comment>
\t\t\t<sPlug>
\t\t\t\t<id><![CDATA[ACA9C502-A04B-42b5-B2EB-5CEA36D16FCE]]></id>
\t\t\t\t<name><![CDATA[VOCALOID2 Compatible Style]]></name>
\t\t\t\t<version><![CDATA[3.0.0.1]]></version>
\t\t\t</sPlug>
\t\t\t<pStyle>
\t\t\t\t<v id="accent">50</v>
\t\t\t\t<v id="bendDep">8</v>
\t\t\t\t<v id="bendLen">0</v>
\t\t\t\t<v id="decay">50</v>
\t\t\t\t<v id="fallPort">0</v>
\t\t\t\t<v id="opening">127</v>
\t\t\t\t<v id="risePort">0</v>
\t\t\t</pStyle>
\t\t\t<singer>
\t\t\t\t<t>0</t>
\t\t\t\t<bs>4</bs>
\t\t\t\t<pc>0</pc>
\t\t\t</singer>"""
        text_pbs = '\r\t\t\t<cc><t>0</t><v id="S">{}</v></cc>'.format(pbs)
        text_last = """
\t\t\t<plane>0</plane>
\t\t</vsPart>
\t</vsTrack>
\t<monoTrack>
\t</monoTrack>
\t<stTrack>
\t</stTrack>
\t<aux>
\t\t<id><![CDATA[AUX_VST_HOST_CHUNK_INFO]]></id>
\t\t<content><![CDATA[VlNDSwAAAAADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=]]></content>
\t</aux>
</vsq4>"""
        text = text_head + text_length + text_mid + text_pbs + text_insert + text_last  # 清醒过来才发现我写了个什么玩意

        with open(address + 'output.vsqx', 'w', encoding='utf-8') as f:
            f.write(text)
        if start:
            os.startfile(os.path.abspath('.') + '\\output.vsqx')


if __name__ == '__main__':
    t = Talker(online=True)
    while True:
        i = input('洛天依说：')
        cmd = 'type'
        if i == '':
            cmd = 'read'
        t.load(cmd, i)
        t.to_vsqx(start=True)
        for mark in t.mark_list:
            print(mark)
