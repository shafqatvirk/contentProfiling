def neighbourhood(s, alpha = [chr(i) for i in range(ord('a'), ord('z')+1)], alpha_upper = [chr(i) for i in range(ord('A'), ord('Z')+1)], n = 1):
    def strn(alphabet, l):
        if l == 0:
            return [""]
        return [c + s for c in alphabet for s in strn(alphabet, l-1)]
    return [s[:i] + a + s[i+1:] for j in range(n) for (i, c) in enumerate(s) for a in strn((alpha_upper if s.isupper() else alpha), j+1) if c != a]

def ocrc(s, gensimmodel, n = 1):
    ns = neighbourhood(s, n = n)
    ss = {n: gensimmodel.similarity(s, n) for n in ns if gensimmodel.vocab.has_key(n) and gensimmodel.vocab[s].count > gensimmodel.vocab[n].count}
    ranrank = len(gensimmodel.vocab)/(len(ns)+1)
    ransim = gensimmodel.similar_by_word(s, topn=ranrank)
    rant = ransim[-1][-1]
    return {w: v for (w, v) in ss.iteritems() if v > rant}

def ocr_correct(gensimmodel, n = 1):
    done = {}
    for w in gensimmodel.vocab.iterkeys():
        if not done.has_key(w):
            done[w] = ocrc(w, gensimmodel, n = n)
            for (rw, _) in done[w].iteritems():
                if not done.has_key(rw):
                    done[rw] = w
    return done

def stopword_filter(rf, gensimmodel):
    rfo = {w: xws for (w, xws) in rf.iteritems() if type(xws) == type({}) and len(xws) > 0}
    erdf = [(gensimmodel.vocab[xw].count, gensimmodel.vocab[w].count + gensimmodel.vocab[xw].count) for (w, xws) in rfo.iteritems() for (xw, v) in xws.iteritems()]
    fr = sum([fxw for (fxw, fa) in erdf])/float(sum([fa for (fxw, fa) in erdf]))

    r = {w: {xw: v for (xw, v) in xws.iteritems() if ((gensimmodel.vocab[xw].count/float(gensimmodel.vocab[w].count+gensimmodel.vocab[xw].count))/v) < fr} for (w, xws) in rfo.iteritems()}
    return {k: v for (k, v) in r.iteritems() if v}


#Get a similarity measure, for instance via gensim see
#https://radimrehurek.com/gensim/models/word2vec.html
#Here we have a trained one stored in tgv2v50.pkl
import gensim
model = gensim.models.Word2Vec.load("tgw2v50.pkl")

#Now we can the OCR variants for any term
print "SOV", ocrc("SOV", model, n = 1)
print "man", ocrc("man", model, n = 1)
print "of", ocrc("of", model, n = 1)

#We can do it for all terms, avoid doing it twice when not necessary 
rf = ocr_correct(model, n = 1)

#This corrects for true minimal pairs, see the last step in the Poor man's
#OCR post-correction method
#Hammarstr\"om, Harald, Shafqat Virk & Markus Forsberg. 2017. Poor man's OCR post-correction: unsupervised recognition of variant spelling applied to a multilingual document collection. Proceedings of the Digital Access to Textual Cultural Heritage (DATeCH) conference.
r = stopword_filter(rf, model)


#And this transforms the output into a dictionary of erroneous terms to the proposed corrections
ocrerror_to_correction = {xw: w for (w, xws) in r.iteritems() for (xw, v) in xws.iteritems()}

