def neighbourhood(s, alpha = [chr(i) for i in range(ord('a'), ord('z')+1)], alpha_upper = [chr(i) for i in range(ord('A'), ord('Z')+1)], n = 1):
    def strn(alphabet, l):
        if l == 0:
            return [""]
        return [c + s for c in alphabet for s in strn(alphabet, l-1)]
    return [s[:i] + a + s[i+1:] for j in range(n) for (i, c) in enumerate(s) for a in strn((alpha_upper if s.isupper() else alpha), j+1) if c != a]

def ocrc(s, gensimmodel, n = 1):
    ns = neighbourhood(s, n = n)
    ss = {n: gensimmodel.similarity(s, n) for n in ns if gensimmodel.vocab.has_key(n) and gensimmodel.vocab[s].count > gensimmodel.vocab[n].count}
    ranrank = len(model.vocab)/(len(ns)+1)
    ransim = gensimmodel.similar_by_word(s, topn=ranrank)
    rant = ransim[-1][-1]
    return {w: v for (w, v) in ss.iteritems() if v > rant}

#1/2    = 1
#2/3    = 2
#3/4    = 3
#4/5    = 4
#20/21  = 20    


def ocr_correct(gensimmodel, n = 1):
    ws = list(sorted(gensimmodel.vocab, key = lambda x: gensimmodel.vocab[x].count, reverse=True)) #[:1000]
    done = {}
    for w in ws:
        if not done.has_key(w):
            done[w] = ocrc(w, gensimmodel, n = n)
            for (rw, _) in done[w].iteritems():
                if not done.has_key(rw):
                    done[rw] = w
            #print w, done[w]
    return done

def fd(xs):
    d = {}
    for x in xs:
        d[x] = d.get(x, 0) + 1
    return d

def df(x, y):
    i = min([i for i in range(len(x)) if x[i] != y[i]] + [len(x)-1])
    return (x[i], y[i:len(y)-len(x)+i+1])

def transitions(done):
    return grp2([(df(w, xw), (w, xw)) for (w, rws) in done.iteritems() if type(rws) == type({}) for xw in rws.iterkeys()])


import gensim

model = gensim.models.Word2Vec.load("tgw2v.pkl")
print "SOV", ocrc("SOV", model)
print "man", ocrc("man", model)
    
r = ocr_correct(model, n = 2)
