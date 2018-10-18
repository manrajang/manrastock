from bs4 import BeautifulSoup
from dbMgr import DbMgr
import urllib.request as req

codeUrl = "https://finance.naver.com/sise/sise_market_sum.nhn?sosok="
suffix0 = "&page="
suffix1 = "&col=listprice&order=desc"
kospi_page_len = 31
kosdaq_page_len = 26


class CodeMgr:
    @staticmethod
    def _insert_stock_codes(stock_type, page):
        stock_codes = BeautifulSoup(req.urlopen(codeUrl + str(stock_type) + suffix0 + str(page)), "html.parser").select("td > a.tltle")

        for stock_code in stock_codes:
            # print(stock_code.attrs["href"][-6:], stock_code.getText(), stock_type)
            DbMgr.insert_into_stock_codes_db(stock_code.attrs["href"][-6:], stock_code.getText(), "P" if stock_type == 0 else "Q")

    @staticmethod
    def collect_stock_codes():
        DbMgr.create_stock_codes_db()

        # KOSPI
        for i in range(1, kospi_page_len):
            print("KOSPI" + str(i))
            CodeMgr._insert_stock_codes(0, i)

        # KOSDAQ
        for i in range(1, kosdaq_page_len):
            print("KOSDAQ" + str(i))
            CodeMgr._insert_stock_codes(1, i)


if __name__ == "__main__":
    i = 0
    stock_codes = DbMgr.get_all_stock_code()

    for code, row in stock_codes.iterrows():
        print(i, row["code_name"], code)
        i += 1
