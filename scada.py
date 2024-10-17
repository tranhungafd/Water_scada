from pydruid.client import PyDruid

# Kết nối tới Druid
druid = PyDruid('http://124.158.4.142:8888/', 'druid/v2/sql')
# Thực hiện truy vấn
query = 'SELECT * FROM my_datasource LIMIT 10'
results = druid.sql(query)

# Hiển thị kết quả
for row in results:
    print(row)