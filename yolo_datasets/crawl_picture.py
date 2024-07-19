import re
# 进行http请求的第三方库
import requests
from urllib import error
# 解析HTML和XML文档的库
from bs4 import BeautifulSoup
import os

num = 0
numPi = 0
file = ''
List = []
 
def makefile(word):
    file = word + '文件'
    file_na = os.path.exists(file)
    if file_na == 1:
        print('该文件已存在，请重新输入')
        file = input('请建立一个存储图片的文件夹，输入文件夹名称即可: ')
        os.mkdir(file)
    else:
        os.mkdir(file)
    return file, os.path.abspath(file)

def Find(url, A):
    global List  
    t = 0          
    i = 1           
    s = 0           
    while t < 300000:
        Url = url + str(t)
        try:
            Result = A.get(Url, timeout=7, allow_redirects=False)
        except BaseException:
            t = t + 60
            continue
        else:
            result = Result.text
            pic_url = re.findall('"objURL":"(.*?)",', result, re.S)
            # 图片总数
            # 提取到的图片url数量加到s变量上
            s += len(pic_url)
            # 如果提取到的图片URL数量为 0，表示网页中没有图片，跳出循环
            if len(pic_url) == 0:
                break
            else:
                # 将提取到的图片 URL 添加到 List 列表中。
                List.append(pic_url)
                #  将时间戳 t 增加 60 秒，然后继续循环
                t = t + 60
    return s
 
 
# 记录相关数据
def recommend(url):
    Re = []
    try:
        # 向网页发送一个请求并返回响应
        html = requests.get(url, allow_redirects=False)
    except error.HTTPError as e:
        return
    else:
        html.encoding = 'utf-8'
        # html文件解析，解析响应的文件内容，html.text 是 HTML 文档的源代码，
        # 'html.parser' 是解析器，用于指定如何解析 HTML 文档
        bsObj = BeautifulSoup(html.text, 'html.parser')
        # 找到页面中id为topsRS的div元素
        div = bsObj.find('div', id='topRS')
        # 从该div元素中找到所有a的标签，并提取其中文本内容
        if div is not None:
            listA = div.findAll('a')
            for i in listA:
                if i is not None:
                    Re.append(i.get_text())
        return Re
 
 
# 下载图片
def dowmloadPicture(html, keyword, file_path):
    global num
    pic_url = re.findall('"objURL":"(.*?)",', html, re.S)
    print('找到:' + keyword + '的图片，开始下载....')
    # 遍历图片的url
    for each in pic_url:
        print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
        try:
            if each is not None:
                # 请问时间不能超过7s
                pic = requests.get(each, timeout=7)
            else:
                continue
        except BaseException:
            print('错误，当前图片无法下载')
            continue
        else:
            # 构建图片保存路径
            string = file_path + '//'+ str(num) + '.jpg'
            # 以二进制写入模式打开新建文件
            fp = open(string, 'wb')
            # 将下载的图片内容写入文件
            fp.write(pic.content)
            # 关闭文件
            fp.close()
            # 已经下载一张图片加1
            num += 1
        # 检查是否已经下载所有需要下载的图片
        if num >= numPi:
            return
 
def run():
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
 
    # 创建一个请求的会话
    A = requests.Session()
    # 设置头部信息
    A.headers = headers
    word = input("输入要搜索的关键词:")
    numPi = int(input('输入要下载的数量:'))
    # 拼接路径
    url = 'https://image.baidu.com/search/flip?ct=201326592&cl=2&st=-1&lm=-1&nc=1&ie=utf-8&tn=baiduimage&ipn=r&rps=1&pv=&fm=rs1&word=' + word

    total = Find(url, A)
    # 记录相关推荐图片
    Recommend = recommend(url)
    print('经过检测%s类图片共有%d张' % (word, total))
    file, file_path = makefile(word)

    t = 0
    tmp = url

    while t < numPi:
        try:
            url = tmp + str(t)
            result = requests.get(url, timeout=10)
            print(url)
        except error.HTTPError as e:
            print('网络错误，请调整网络后重试')
            t = t + 60
        else:
            dowmloadPicture(result.text, word, file_path)
            t = t + 60


run()

# if __name__ == '__main__':  # 主函数入口
    # headers = {
    #     'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
    #     'Connection': 'keep-alive',
    #     'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
    #     'Upgrade-Insecure-Requests': '1'
    # }
 
    # # 创建一个请求的会话
    # A = requests.Session()
    # # 设置头部信息
    # A.headers = headers
    # word = input("输入要搜索的关键词:")
    # numPi = int(input('输入要下载的数量:'))
    # # 拼接路径
    # url = 'https://image.baidu.com/search/flip?ct=201326592&cl=2&st=-1&lm=-1&nc=1&ie=utf-8&tn=baiduimage&ipn=r&rps=1&pv=&fm=rs1&word=' + word

    # total = Find(url, A)
    # # 记录相关推荐图片
    # Recommend = recommend(url)
    # print('经过检测%s类图片共有%d张' % (word, total))
    # file, file_path = makefile(word)

    # t = 0
    # tmp = url

    # while t < numPi:
    #     try:
    #         url = tmp + str(t)
    #         result = requests.get(url, timeout=10)
    #         print(url)
    #     except error.HTTPError as e:
    #         print('网络错误，请调整网络后重试')
    #         t = t + 60
    #     else:
    #         dowmloadPicture(result.text, word)
    #         t = t + 60