'''
爬取豆瓣电影Top 250
'''

import requests
from pyquery import PyQuery as pq
import time

BASE_URL = 'https://movie.douban.com/top250?start='


def get_movies():
    movies = []
    for i in range(0, 250, 25):
        url = BASE_URL + str(i)
        html = requests.get(url).text
        doc = pq(html)
        lis = doc('ol.grid_view li').items()
        for li in lis:
            # 电影排名
            rank = li('div.pic em').text()

            # 电影名称
            info = li('div.info')
            name = ''.join([item.text() for item in info('div.hd a span').items()])
            name = ''.join(name.split())  # 去除所有空格

            cnts = list(li('p').items())[0].text().split('\n')

            # 导演，演员
            humans = [item for item in cnts[0].split('\xa0') if item]

            if len(humans) < 2:
                director = humans[0]
                starring = ''
                director = director.replace('导演: ', '')
            else:
                director, starring = humans
                director = director.replace('导演: ', '')
                starring = starring.replace('主演: ', '')

            # 上映年份，国家，类型
            # 移除&nbsp空格符
            year, country, types = (cnts[1].replace('\xa0', '').split('/'))[-1:-4:-1]

            # 评星
            star = li('div.star span.rating_num').text()

            # 摘要
            quote = li('p.quote').text()

            # 电影排名，电影名称，导演，主演，上映年份，国家，类型，评分，摘要
            movie = (rank, name, director, starring, year, country, types, star, quote)
            print(movie)
            movies.append(movie)
        with open('douban_movies_top250.txt', 'w', encoding='utf-8') as f:
            f.writelines(','.join(movie)+'\n' for movie in movies)
    return movies




def count_spend_time(func):
    start_time = time.time()
    func()
    end_time = time.time()
    time_dif = (end_time - start_time)
    second = time_dif%60
    minute = (time_dif//60)%60
    hour = (time_dif//60)//60
    print('spend ' + str(hour) + 'hours,' + str(minute) + 'minutes,' + str(second) + 'seconds')


if __name__ == '__main__':
    count_spend_time(get_movies)