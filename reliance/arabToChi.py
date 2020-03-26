"""‘两’与‘二’的区分：二用在二十以内时
update 3.12 修正了数字过长无法找到对应数量词的bug
"""
num_str_start_symbol = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10', '.']
# more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
import re
def to_Chiness_num(num):
    num_dict = {'1':'一', '2':'二', '3':'三', '4':'四', '5':'五', '6':'六', '7':'七', '8':'八', '9':'九', '0':'零', '.': '点'}
    index_dict = {1:'', 2:'十', 3:'百', 4:'千', 5:'万', 6:'十', 7:'百', 8:'千', 9:'亿'}

    nums = list(num)
    digits = []
    if '.' in nums:
        i = nums.index('.')
        digits = nums[i:]
        digits = [num_dict[d] for d in digits]
        nums = nums[:i]

    if len(nums) > 9:  # 太长不转换
        digits = [num_dict[d] for d in nums] + digits
        nums = []

    nums_index = [x for x in range(1, len(nums)+1)][-1::-1]

    string = ''
    for index, item in enumerate(nums):
        string = "".join((string, num_dict[item], index_dict[nums_index[index]]))

    string = re.sub("零[十百千零]*", "零", string)
    string = re.sub("零万", "万", string)
    string = re.sub("亿万", "亿零", string)
    string = re.sub("零零", "零", string)
    string = re.sub("零\\b", "", string)
    string = re.sub('二千', '两千', string)
    string = re.sub('二万', '两万', string)
    string = re.sub('二亿', '两亿', string)

    string = string + "".join(digits)

    if string[:2] == '一十':
        string = string[1:]
    return string
def changeArabNumToChinese(oriStr):
    lenStr = len(oriStr)
    aProStr = ''
    if lenStr == 0:
        return aProStr

    hasNumStart = False  #默认 不含阿拉伯数字
    numberStr = ''        #默认 不含阿拉伯数字
    endWithDot = False
    for idx in range(lenStr): #遍历所有 字符
        if oriStr[idx] in num_str_start_symbol: #当前字符 是 阿拉伯数字或"."
            if not hasNumStart:
                hasNumStart = True
            endWithDot = oriStr[idx] == '.'

            numberStr += oriStr[idx] #将个位阿拉伯数字 累加到输出字符串
    
    
        else:   #当前字符 不是 阿拉伯数字或"."
            
            if hasNumStart:   #当前字符不是数字，但是【之前的字符 包含数字】，先把之前的数字进行转换，并累加
                if endWithDot:
                    numberStr = numberStr[:-1]

                numResult = str(to_Chiness_num(numberStr)) # 将之前的阿拉伯数字 转换为 汉字数字

                aProStr += numResult  #将汉字数字，累加到 输出字符串

                numberStr = ''  # 数字清空，为了记录下一次的 阿拉伯数字
                hasNumStart = False # 设置为false  恢复到初始值

            aProStr += oriStr[idx]  #将不是阿拉伯的 当前汉字字符 累加到 输出字符串

 
    if len(numberStr) > 0: # 处理以数字 结尾的情况 ，不能去掉。  '12套餐12'
        resultNum = to_Chiness_num(numberStr) # 将汉字数字 转换为 阿拉伯数字
        aProStr += str(resultNum)
 
    return aProStr
 
def convert(str):
    return changeArabNumToChinese(str)

"""
版权声明：本文件内的代码的基础是CSDN博主「HeartBeating_RUC」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_40467413/article/details/86015439
"""