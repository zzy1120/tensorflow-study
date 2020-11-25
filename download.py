# coding:utf-8

import os
import random
import urllib.request as request

# 下载图片脚本用到的库

path2 = "c:\\data"
# os.mkdir(path2 + "\\picture\\")
# 以上两行即在d盘tu目录下创建名称为test的文件夹

c = "http://i.imgur.com/05lUV1G.jpg"
# 图片地址

headers = {
    'authority': 'cl.bc53.xyz',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site': 'none',
    'sec-fetch-mode': 'navigate',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': '__cfduid=d9b8dda581516351a1d9d388362ac222c1603542964',
}
# 头信息，后面请求图片地址的时候需要带上，否则容易禁止访问

print("loading" + " " + c)

pic_name = random.randint(0, 100)  # 图片名称随机命令

r = request.Request(c, headers=headers)  # 请求图片地址，注意”r“

with open(path2 + "\\picture\\" + str(pic_name) + '.jpg', 'wb') as fd:
    # for chunk in r.iter_content():
    #     fd.write(chunk)
    fd.write(r.read())
# 下载脚本，实际就是把图片保存到D盘tu目录test文件夹pic_name文件中
