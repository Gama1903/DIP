s1 = 'h'
try:
    int(s1)
except Exception as e:  # 未捕获到异常，程序直接报错
    print(e)
# int(s1)