import pickle

dictToPickle = {'hello': 'world', 'kaga' : 'igen', 'teszt' : 'XDDD'}

pickle.dump(dictToPickle,
            open('pickledDict.pickle', 'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)