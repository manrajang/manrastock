from codeMgr import CodeMgr
from dbMgr import DbMgr
from stockMgr import StockMgr
import matplotlib.pyplot as plt


def check_name(s_name):
    return "테마" not in s_name and "선물" not in s_name and "TIGER" not in s_name and "KODEX" not in s_name and \
           "ARIRANG" not in s_name and "KINDEX" not in s_name and "KOSEF" not in s_name and \
           "KBSTAR" not in s_name and "ETN" not in s_name and "QV" not in s_name and "2우B" not in s_name and \
           "2" not in s_name


def test(stock_codes):
    j = 0

    for code, row in stock_codes.iterrows():
        name = row["code_name"]

        if check_name(name) and DbMgr.get_stock_count(code) > 300 and StockMgr.check_condition(code):
            print(j, name, code)

        j += 1


def keras_predict(stock_codes, col):
    j = 0

    for code, row in stock_codes.iterrows():
        name = row["code_name"]

        if check_name(name) and DbMgr.get_stock_count(code) > 300 and StockMgr.check_condition(code):
            print(j, name, code)
            StockMgr.keras_train(code, col)

        j += 1


def tf_predict(stock_codes):
    j = 0

    for code, row in stock_codes.iterrows():
        name = row["code_name"]

        if check_name(name) and DbMgr.get_stock_count(code) > 300 and StockMgr.check_condition(code):
            print(j, name, code)
            result = StockMgr.tf_train(code, ["stock_open", "stock_high", "stock_low", "stock_volume", "stock_agency",
                                              "stock_foreigner", "stock_close"])

            if result:
                with open("tf_stock.txt", "a") as file:
                    file.write("{}|".format(code))
        j += 1


def all_train(stock_codes):
    j = 0

    for code, row in stock_codes.iterrows():
        name = row["code_name"]

        if check_name(name) and DbMgr.get_stock_count(code) > 300:
            print(j, name, code)
            result = StockMgr.keras_train(code, "stock_close")

            if result:
                with open("keras_stock.txt", "a") as file:
                    file.write("{}|".format(code))

        j += 1


def print_menu():
    print("1. 주식 업데이트")
    print("2. 케라스 코스닥 주식 예측")
    print("3. 케라스 코스피 주식 예측")
    print("4. 텐서플로우 코스피 주식 예측")
    print("5. 텐서플로우 코스닥 주식 예측")
    print("6. 테스트")
    print("7. 종료")

    return int(input("메뉴선택: "))


if __name__ == "__main__":
    while 1:
        menu = print_menu()
        if menu == 1:
            i = 0
            CodeMgr.collect_stock_codes()
            stock_codes = DbMgr.get_all_stock_code()

            for code, row in stock_codes.iterrows():
                print(i, row["code_name"], code)
                i += 1

                if DbMgr.exist_stock_info_db(code)[0][0] == 1:
                    if DbMgr.get_stock_count(code) < 10:
                        print("생성")
                        StockMgr.collect_total_stock_info(code)
                    else:
                        StockMgr.collect_stock_info(code)
                else:
                    print("생성")
                    StockMgr.collect_total_stock_info(code)
        elif menu == 2:
            keras_predict(DbMgr.get_stock_code("P"), "stock_close")
        elif menu == 3:
            keras_predict(DbMgr.get_stock_code("Q"), "stock_close")
        elif menu == 4:
            tf_predict(DbMgr.get_stock_code("P"))
        elif menu == 5:
            tf_predict(DbMgr.get_stock_code("Q"))
        elif menu == 6:
            test(DbMgr.get_all_stock_code())
        elif menu == 8:
            print(DbMgr.get_stock_count("000660")[0][0])
        else:
            break
