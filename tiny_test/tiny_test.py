def hoge(someAarg, someList=None):
    if someList is None:
        someList = []
    print(someList)
    someList.append(someAarg)

hoge('a')
hoge('b')
hoge('c')
