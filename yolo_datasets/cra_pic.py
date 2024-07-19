import re
import requests
import time

# 定义请求头部信息
headers = {
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Mobile Safari/537.36",
    # 此处省略了 Cookie 信息，因为通常包含敏感数据
}

# 存储图片地址的列表
detail_urls = []

# 爬取20页图片，每页获取一张
for i in range(1, 200, 20):
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1719971009214_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&dyTabStr=&ie=utf-8&ctd=1719971009214%5E00_2495X1288&sid=&word=%E7%81%AD%E7%81%AB%E5%99%A8%E7%AE%B1&pn={}'.format(i)
    #url = 'https://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=&st=-1&fm=result&fr=&sf=1&fmq=1592804203005_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1592804203008%5E00_1328X727&sid=&word=%E7%81%AD%E7%81%AB%E5%99%A8&pn={}'.format(i)
    
    response = requests.get(url, headers=headers, timeout=(5, 7))  # 设置请求超时时间3-7秒
    content = response.content.decode('utf-8')  # 使用utf-8进行解码
    detail_url = re.findall('"objURL":"(.*?)"', content, re.DOTALL)  # 匹配图片URL
    detail_urls.append(detail_url)

# 图片计数器
b = 0

# 遍历所有页面的图片URL列表
for page in detail_urls:
    for url in page:
        try:
            print('正在下载第{}张图片...'.format(b + 1))
            response = requests.get(url, headers=headers, timeout=5)  # 设置请求超时时间
            content = response.content

            # 根据文件扩展名保存图片
            if url.endswith('.jpg'):
                filename = '消防栓_500/{}.jpg'.format(b)
            elif url.endswith('.jpeg'):
                filename = '消防栓_500/{}.jpeg'.format(b)
            elif url.endswith('.png'):
                # 修正文件扩展名，应该是.png而不是.pon
                filename = '消防栓_500/{}.png'.format(b)
            else:
                print('未知文件类型，跳过')
                continue

            # 保存图片
            with open(filename, 'wb') as f:
                f.write(content)
            print('图片已保存：{}'.format(filename))

        except requests.exceptions.RequestException as e:
            print('下载图片失败：', e)
        finally:
            b += 1

print('图片下载完成。')

