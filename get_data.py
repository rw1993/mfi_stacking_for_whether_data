import sqlite3
import numpy as np

con = sqlite3.connect("data/whether")

def add_cusor(func):
    def _(*args, **kw):
        cursor = con.cursor()
        r = func(*args, **kw, cursor=cursor)
        cursor.close()
        return r
    return _

@add_cusor
def get_cities(cursor):
    record_city = {}
    for city, records in cursor.execute("select city, count(*) from whether group by city"):
        if records not in record_city:
            record_city[records] = []
        record_city[records].append(city)
    records_cities = [(len(record_city[i]), i) for i in record_city]
    return record_city[max(records_cities)[1]]
cities = get_cities()

@add_cusor
def get_city_set(city, cursor):
    sql = "select * from whether where city==? order by yearmonth"
    return [row for row in cursor.execute(sql, (city,))]

@add_cusor
def get_whether_types(cursor):
    sql = "select distinct morning_whether from whether"
    r1 = [t for t, in cursor.execute(sql)]
    sql = "select distinct night_whether from whether"
    r2 = [t for t, in cursor.execute(sql)]
    return list(set(r1).union(set(r2)))
whether_types = get_whether_types()

@add_cusor
def get_wind_types(cursor):
    sql = "select distinct morning_wind from whether"
    r1 = [w for w, in cursor.execute(sql)]
    sql = "select distinct night_wind from whether"
    r2 = [w for w, in cursor.execute(sql)]
    return list(set(r1).union(set(r2)))
wind_types = get_wind_types()

@add_cusor
def get_years(cursor):
    sql = "select distinct yearmonth from whether"
    years = set()
    for ym, in cursor.execute(sql):
        year = ym[:4]
        years.add(year)
    return list(years)

years = get_years()

def row_to_feature(row):
    # one_hot for type
    def add_float(n):
        try:
            feature.append(float(n))
        except:
            print("invalid num", n, row[0], row[1])
            feature.append(10.0)
    feature = []
    add_float(row[4])
    add_float(row[5])
    city = row[0]
    feature = feature + [0.0 if c != city else 1.0 for c in cities]
    ymd = row[1]
    feature = feature + [0.0 if y != ymd[:4] else 1.0 for y in years]
    mw = row[2]
    feature = feature + [0.0 if w != mw else 1.0 for w in whether_types]
    nw = row[3]
    feature = feature + [0.0 if w != nw else 1.0 for w in whether_types]

    mw = row[6]
    feature = feature + [0.0 if w != mw else 1.0 for w in wind_types]
    mw = row[7]
    feature = feature + [0.0 if w != mw else 1.0 for w in wind_types]
    return np.array(feature)

def get_data():
    def get_city_data(city):
        data = get_city_set(city)
        data = list(map(row_to_feature, data))
        return np.array(data)
    total_data = list(map(get_city_data, cities))
    return total_data

@add_cusor
def get_temp_range(cursor):
    tmps = set()
    for tmp, in cursor.execute("select distinct morning_temperture from whether"):
        try:
            tmps.add(float(tmp))
        except:
            print(tmp)
    for tmp, in cursor.execute("select distinct night_temperture from whether"):
        try:
            tmps.add(float(tmp))
        except:
            print(tmp)
 
    return max(tmps), min(tmps)

if __name__ != "__main__":
    data = get_data()
    pass

def main():
    max_t, min_t = get_temp_range()
    import ipdb; ipdb.set_trace()
    

if __name__ == '__main__':
    main()
