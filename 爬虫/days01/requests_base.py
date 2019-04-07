import requests

url = 'https://blog.csdn.net/the_harder_to_love'  # 我的博客主页
response = requests.get(url)  # get method
# response = requests.get(url, data='夜是故乡明')  # 加上参数data, post method
html = response.text  # 获取网页内容
print(html)
