import json
import re
import pandas as pd
import urllib.request as req
from bs4 import BeautifulSoup

pattern = re.compile("page=[0-d]+")


def write_json_file(file_name, obj):
    with open(file_name, "w", encoding="utf-8") as t:
        json.dump(obj, t, ensure_ascii=False, indent="\t")


def read_json_file(file_name):
    f = open(file_name, "r")
    js = json.loads(f.read())
    f.close()

    return js


def get_stock_info_url(code):
    return "http://finance.naver.com/item/sise_day.nhn?code={code}".format(code=code)


def get_stock_sum_price_url(code):
    return "http://finance.naver.com/item/main.nhn?code={code}".format(code=code)


def get_stock_info_frgn_url(code):
    return "http://finance.naver.com/item/frgn.nhn?code={code}".format(code=code)


def get_invest_deal_url(bizdate, sosok):
    return "http://finance.naver.com/sise/investorDealTrendDay.nhn?bizdate={bizdate}&sosok={sosok}".\
        format(bizdate=bizdate, sosok=sosok)


def get_indices_url(code):
    return "http://finance.naver.com/sise/sise_index_day.nhn?code={code}".format(code=code)


def get_company_report_fnguide_url(fr_dt, to_dt):
    return "http://comp.fnguide.com/SVO2/asp/SVD_Report_Summary_Data.asp?" \
           "fr_dt={fr_dt}&to_dt={to_dt}&stext=&check=all&sortOrd=5&sortAD=A&_=2".format(fr_dt=fr_dt, to_dt=to_dt)


def get_page_data(url, page):
    return pd.read_html("{url}&page={page}".format(url=url, page=page), header=0, encoding="euc-kr")


def get_company_fundamental_fnguide_url(code):
    return "http://asp01.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{code}&NewMenuID=11&cID=50&MenuYn=N".\
        format(code=code)


def get_page_first_data(url, page):
    return get_page_data(url, page)[0]


def get_soup(url):
    try:
        return BeautifulSoup(req.urlopen(url), "html.parser")
    except req.HTTPError as e:
        print(e.code)
        # req.urlcleanup()
        return -1


def get_page_size(url):
    last_page_tag_list = get_soup(url).select("td.pgRR > a")
    if len(last_page_tag_list) > 0:
        page_size = int(pattern.findall(last_page_tag_list[0].attrs["href"])[0].split("=")[1])
    else:
        page_size = 1

    return page_size


def get_data(url, sort_str):
    page_size = get_page_size(url)
    df = pd.DataFrame()

    for page in range(1, page_size + 1):
        page_df = get_page_first_data(url, page)
        df = df.append(page_df.dropna(), ignore_index=True)

    return df.sort_values(by=[sort_str], ascending=True)


def get_frgn_data(url, index_str):
    page_size = get_page_size(url)
    df = pd.DataFrame()
    for page in range(1, page_size + 1):
        print(page)
        page_df_list = get_page_data(url, page)
        if len(page_df_list) > 2:
            page_df = page_df_list[2]
            df = df.append(page_df.dropna(), ignore_index=True)

    return df.set_index(index_str)




