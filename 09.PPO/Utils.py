class buffer():
    def __init__(self):
        self.bs, self.ba, self.br, self.blogp, self.bd = [], [], [], [], []

    def get_atr(self):
        return self.bs, self.ba, self.br, self.blogp, self.bd

    def store_r_d(self, r, d):
        self.br.append(r)
        self.bd.append(d)

    def clear(self):
        del self.bs[:]
        del self.ba[:]
        del self.br[:]
        del self.blogp[:]
        del self.bd[:]
