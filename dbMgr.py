import sqlite3
import pandas as pd

dbPath = "stock.sqlite"
stockCodesTable = "stock_codes"
stockInfoTablePrefix = "stock"


class DbMgr:
    conn = sqlite3.connect(dbPath)
    cursor = conn.cursor()

    @staticmethod
    def check_condition_info(table, fr_dt, to_dt):
        df = pd.read_sql_query("SELECT stock_date, stock_close, stock_open, stock_high, stock_low, stock_volume, " +
                               "stock_agency, stock_foreigner, ma5, ma20, ma60 FROM " + stockInfoTablePrefix + table +
                               " WHERE stock_date BETWEEN '" + fr_dt + "' AND '" + to_dt +
                               "' ORDER BY stock_date", DbMgr.conn)

        for index, row in df.iterrows():
            df.ix[index, "stock_agency"] = str(row["stock_agency"]).replace(",", "")
            df.ix[index, "stock_foreigner"] = str(row["stock_foreigner"]).replace(",", "")

        df[["stock_agency", "stock_foreigner"]] = df[["stock_agency", "stock_foreigner"]].astype(int)
        df = df.set_index("stock_date")

        return df[-21:]

    @staticmethod
    def get_stock_count(table):
        return pd.read_sql_query("SELECT Count(*) FROM '" + stockInfoTablePrefix + table + "'", DbMgr.conn).values[0][0]

    @staticmethod
    def get_stock_data(table, columns):
        df = pd.read_sql_query("SELECT " + columns + " FROM " + stockInfoTablePrefix + table + " ORDER BY stock_date",
                               DbMgr.conn)

        if "stock_agency" in columns:
            for index, row in df.iterrows():
                df.ix[index, "stock_agency"] = str(row["stock_agency"]).replace(",", "")

            df[["stock_agency"]] = df[["stock_agency"]].astype(float)

        if "stock_foreigner" in columns:
            for index, row in df.iterrows():
                df.ix[index, "stock_foreigner"] = str(row["stock_foreigner"]).replace(",", "")

            df[["stock_foreigner"]] = df[["stock_foreigner"]].astype(float)

        if "ma5" in columns:
            df.drop(df[df["ma5"] == 0].index, inplace=True)
            df.index = range(len(df.index))
            df[["ma5"]] = df[["ma5"]].astype(float)

        if "ma20" in columns:
            df.drop(df[df["ma20"] == 0].index, inplace=True)
            df.index = range(len(df.index))
            df[["ma20"]] = df[["ma20"]].astype(float)

        if "stock_open" in columns:
            df.drop(df[df["stock_open"] == 0].index, inplace=True)
            df.index = range(len(df.index))

        if "stock_close" in columns:
            df[["stock_close"]] = df[["stock_close"]].astype(float)

        if "stock_volume" in columns:
            df.drop(df[df["stock_volume"] == 0].index, inplace=True)
            df.index = range(len(df.index))

        return df.values

    @staticmethod
    def get_current_stock_data(table, columns):
        df = pd.read_sql_query("SELECT " + columns + " FROM " + stockInfoTablePrefix + table + " ORDER BY stock_date",
                               DbMgr.conn)
        df[["stock_close"]] = df[["stock_close"]].astype(float)

        return df[-50:].values

    @staticmethod
    def create_stock_info_db(table):
        DbMgr.cursor.execute("DROP TABLE IF EXISTS " + stockInfoTablePrefix + table)
        DbMgr.cursor.execute("CREATE TABLE IF NOT EXISTS " + stockInfoTablePrefix + table +
                             "(stock_date TEXT PRIMARY KEY, stock_close Integer, stock_diff Integer, " +
                             "stock_open Integer, stock_high Integer, stock_low Integer, stock_volume Integer, " +
                             "stock_change TEXT, stock_agency Integer, stock_foreigner Integer, " +
                             "stock_supplement Integer, stock_supplement_rate TEXT, " +
                             "ma5 Integer, ma20 Integer, ma60 Integer, ma120 Integer)")

    @staticmethod
    def get_stock_limit_data(table, limit):
        df = pd.read_sql_query("SELECT * FROM " + stockInfoTablePrefix + table + " ORDER BY stock_date DESC limit " +
                               str(limit), DbMgr.conn)
        df.columns = ["날짜", "종가", "전일비", "시가", "고가", "저가", "거래량", "등락률", "기관", "외국인", "보유주수", "보유율",
                      "MA5", "MA20", "MA60", "MA120"]
        df = df.sort_values(by=["날짜"], ascending=True)
        df = df.set_index("날짜")

        return df

    @staticmethod
    def insert_into_stock_codes_db(code_num, code_name, code_type):
        DbMgr.cursor.execute("INSERT OR REPLACE INTO " + stockCodesTable + "(code_num, code_name, code_type) " +
                             "VALUES(?, ?, ?)", (code_num, code_name, code_type))
        DbMgr.conn.commit()

    @staticmethod
    def get_all_stock_code():
        df = pd.read_sql_query("SELECT code_num, code_name FROM " + stockCodesTable, DbMgr.conn)
        df = df.set_index("code_num")

        return df

    @staticmethod
    def get_stock_code(code_type="P"):
        df = pd.read_sql_query("SELECT code_num, code_name FROM " + stockCodesTable + " WHERE code_type='" +
                               code_type + "'", DbMgr.conn)
        df = df.set_index("code_num")

        return df

    @staticmethod
    def create_stock_codes_db():
        DbMgr.cursor.execute("DROP TABLE IF EXISTS " + stockCodesTable)
        DbMgr.cursor.execute("CREATE TABLE IF NOT EXISTS " + stockCodesTable +
                             "(code_num TEXT PRIMARY KEY, code_name TEXT, code_type TEXT)")

    @staticmethod
    def insert_into_stock_info_db(table, date, close, diff, s_open, high, low, volume, change, agency, foreigner,
                                  supplement, supplement_rate, ma5, ma20, ma60, ma120):
        DbMgr.cursor.execute("INSERT OR REPLACE INTO " + stockInfoTablePrefix + table +
                             "(stock_date, stock_close, stock_diff, stock_open, stock_high, stock_low, stock_volume, " +
                             "stock_change, stock_agency, stock_foreigner, stock_supplement, stock_supplement_rate, " +
                             "ma5, ma20, ma60, ma120) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (date, close, diff, s_open, high, low, volume, change, agency, foreigner, supplement,
                              supplement_rate, ma5, ma20, ma60, ma120))
        DbMgr.conn.commit()

    @staticmethod
    def exist_stock_info_db(table):
        DbMgr.cursor.execute("SELECT Count(*) FROM sqlite_master where type='table' and name='" + stockInfoTablePrefix +
                             table + "'")
        return DbMgr.cursor.fetchall()
