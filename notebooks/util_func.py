from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

class SaveFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.features = outp
    def remove(self):
        self.handle.remove()
        
class FeatureSaver:
    def __init__(self, learner, tmp_path, data):
        # Inherit this class and change self.sf line
        self.learner = learner
        self.tmp_path = tmp_path
        self.m = learner.models.model
        self.sf = [SaveFeatures(m1) for m1 in [self.m[8], self.m[16]]]
        self.data = data
        self.first_step()
    
    def first_step(self):
        self.m.eval()
        x, y = next(iter(data.trn_dl))
        y = self.m(V(x))
        return
        
    def create_feat_empty_bcolz(self, nf, name):
        return bcolz.carray(np.zeros(nf, np.float32), chunklen=1, mode='w', rootdir=name)

    def get_activations_holder(self, tmpl, nf=None, force=False):
        #tmpl = f'_{self.models.name}_{self.data.sz}.bc'
        names = [os.path.join(self.tmp_path, p+tmpl) for p in ('x_act', 'x_act_val', 'x_act_test')]
        if os.path.exists(names[0]) and not force:
            activations = [bcolz.open(p) for p in names]
        else:
            activations = [self.create_feat_empty_bcolz(nf, n) for n in names]
        return activations

    def save_features_mdl(self, idx, tmpl):
        nf = self.sf[idx].features.shape
#         nf = len(self.sf[idx].features)
        actv = self.get_activations_holder(tmpl, nf, force=True)
        tr_act, val_act, test_act = actv
        if len(actv[0])!=len(self.data.trn_ds):
            self.predict_feat_to_bcolz(idx, self.data.fix_dl, tr_act)
        if len(actv[1])!=len(self.data.val_ds):
            self.predict_feat_to_bcolz(idx, self.data.val_dl, val_act)
        if self.data.test_dl and (len(actv[2])!=len(self.data.test_ds)):
            if self.data.test_dl: self.predict_feat_to_bcolz(idx, self.data.test_dl, test_act)
                
    def predict_feat_to_bcolz(self, idx, gen, arr):
        arr.trim(len(arr))
        lock=threading.Lock()
        self.m.eval()
        print('Starting now')
        for x, _ in tqdm(gen):
            y = self.m(V(x))
            o1 = to_np(self.sf[idx].features)
            with lock:
                arr.append(o1)
                arr.flush()