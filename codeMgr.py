from bs4 import BeautifulSoup
from dbMgr import DbMgr
import urllib.request as req

codeUrl = "http://finance.daum.net/quote/marketvalue.daum?stype="
suffix0 = "&page="
suffix1 = "&col=listprice&order=desc"
kospi_page_len = 51
kosdaq_page_len = 44


class CodeMgr:
    @staticmethod
    def _insert_stock_codes(stock_type, page):
        stock_codes = BeautifulSoup(req.urlopen(codeUrl + stock_type + suffix0 + str(page) + suffix1),
                                    "html.parser").select("td.txt > a")

        for stock_code in stock_codes:
            DbMgr.insert_into_stock_codes_db(stock_code.attrs["href"][-6:], stock_code.getText(), stock_type)

    @staticmethod
    def collect_stock_codes():
        DbMgr.create_stock_codes_db()

        # KOSPI
        for i in range(1, kospi_page_len):
            print("KOSPI" + str(i))
            CodeMgr._insert_stock_codes("P", i)

        # KOSDAQ
        for i in range(1, kosdaq_page_len):
            print("KOSDAQ" + str(i))
            CodeMgr._insert_stock_codes("Q", i)


if __name__ == "__main__":
    i = 0
    stock_codes = DbMgr.get_all_stock_code()

    for code, row in stock_codes.iterrows():
        print(i, row["code_name"], code)
        i += 1
