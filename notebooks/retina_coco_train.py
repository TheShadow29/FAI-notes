from fastai.conv_learner import *
from matplotlib import patches, patheffects
from collections import defaultdict
from pycocotools.cocoeval import COCOeval

coco_path = Path('/scratch/arka/Ark_git_files/coco/')
ann_path = coco_path / 'annotations'
train_path = coco_path / 'train2017'
val_path = coco_path / 'val2017'

IMAGES,ANNOTATIONS,CATEGORIES = ['images', 'annotations', 'categories']

FILE_NAME,ID,IMG_ID,CAT_ID,BBOX = 'file_name','id','image_id','category_id','bbox'

JPEGS = 'imgs'

cats = json.load(open('category.json', 'r'))
trn_fns = json.load(open('trn_fns.json', 'r'))
trn_ids = json.load(open('trn_ids.json', 'r'))
trn_ids = [str(o) for o in trn_ids]
trn_anno = json.load(open('trn_anno.json', 'r'))


def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1],a[2]-a[0]])

MBB_CSV = coco_path / 'csv_new' / 'train_bbx.csv'
JPEGS = 'imgs'
sz=224
bs=32
f_model = resnet34
aug_tfms = [RandomRotate(3, p=0.5, tfm_y=TfmType.COORD),
            RandomLighting(0.05, 0.05, tfm_y=TfmType.COORD),
            RandomFlip(tfm_y=TfmType.COORD)]
tfms = tfms_from_model(f_model, sz, aug_tfms=aug_tfms, crop_type=CropType.NO, tfm_y=TfmType.COORD)
md = ImageClassifierData.from_csv(coco_path, JPEGS, MBB_CSV, tfms=tfms, bs=bs, continuous=True, num_workers=4)

class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2):
        self.ds,self.y2 = ds,y2
        self.sz = ds.sz
    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x,y = self.ds[i]
        return (x, (y,self.y2[i]))

mc = [[cats[str(p[1])] for p in trn_anno[o]] for o in trn_ids]
id2cat = list(cats.values())
cat2id = {v:k for k,v in enumerate(id2cat)}
mcs = np.array([np.array([cat2id[p] for p in o]) for o in mc])

val_idxs = get_cv_idxs(len(trn_ids))
((val_mcs,trn_mcs),) = split_by_idx(val_idxs, mcs)

assert len(trn_ids) == len(trn_fns)
assert len(trn_ids) == len(trn_anno)

trn_ds2 = ConcatLblDataset(md.trn_ds, trn_mcs)
val_ds2 = ConcatLblDataset(md.val_ds, val_mcs)
md.trn_dl.dataset = trn_ds2
md.fix_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2
md.aug_dl.dataset = val_ds2


def hw2corners(ctr, hw): return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)


k = 9
anc_grids = [28,14,7,4,2] #Depends of the initial size 224.
anc_zooms = [1., 2**(1/3), 2**(2/3)]
# anc_zooms = [1]
anc_ratios = [(1.,1.), (1.,2), (2,1.)]
# anc_ratios = [(1.,1.)]
anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
anc_offsets = [1/(o*2) for o in anc_grids]
anc_x = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])
anc_y = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])
anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)
anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids])
grid_sizes = V(np.concatenate([np.array([ 1/ag       for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids]), requires_grad=False).unsqueeze(1)
anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()
anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])

n_clas = len(id2cat)+1
n_act = k*(4+n_clas)

res50 = resnet50(pretrained=True)

def pad_out(k):
    return (k-1)//2

class FPN_backbone(nn.Module):
    def __init__(self, inch_list):
        super().__init__()
        
#         self.backbone = backbone
        
        # expects c3, c4, c5 channel dims
        self.inch_list = inch_list
        self.feat_size = 256
        self.p7_gen = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, stride=2, kernel_size=3,
                               padding=1)
        self.p6_gen = nn.Conv2d(in_channels=self.inch_list[2], 
                            out_channels=self.feat_size, kernel_size=3, stride=2, padding=pad_out(3))
        self.p5_gen1 = nn.Conv2d(in_channels=self.inch_list[2], 
                                 out_channels=self.feat_size, kernel_size=1, padding=pad_out(1))
#         self.p5_gen2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_gen3 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size,
                                kernel_size=3, padding=pad_out(3))
        
        self.p4_gen1 = nn.Conv2d(in_channels=self.inch_list[1], out_channels=self.feat_size, kernel_size=1,
                                padding=pad_out(1))
#         self.p4_gen2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_gen3 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3, 
                                padding=pad_out(3))
        
        self.p3_gen1 = nn.Conv2d(in_channels=self.inch_list[0], out_channels=self.feat_size, kernel_size=1,
                                padding=pad_out(1))
        self.p3_gen2 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size, kernel_size=3,
                                padding=pad_out(3))
        
    def forward(self, inp):
        # expects inp to be output of c3, c4, c5
        c3 = inp[0]
        c4 = inp[1]
        c5 = inp[2]
        p51 = self.p5_gen1(c5)
        p5_out = self.p5_gen3(p51)
        
#         p5_up = self.p5_gen2(p51)
        p5_up = F.interpolate(p51, scale_factor=2)
        p41 = self.p4_gen1(c4) + p5_up
        p4_out = self.p4_gen3(p41)
        
#         p4_up = self.p4_gen2(p41)
        p4_up = F.interpolate(p41, scale_factor=2)
        p31 = self.p3_gen1(c3) + p4_up
        p3_out = self.p3_gen2(p31)
        
        p6_out = self.p6_gen(c5)
        
        p7_out = self.p7_gen(F.relu(p6_out))
        
        return [p3_out, p4_out, p5_out, p6_out, p7_out]
        
def flatten_conv(x,k):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)

def initialize_vals(mdl):
#     prior = 0.01
    for m in mdl.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class classf_model(nn.Module):
    def __init__(self, fs=256, na=9, nc=80):
        super().__init__()
        self.na = na
        self.nc = nc
        self.feat_size = fs
        self.cls_modl = nn.Sequential(*nn.ModuleList([nn.Conv2d(in_channels=self.feat_size,
                                                                           out_channels=self.feat_size,
                                                                           kernel_size=3, padding=1)]*4),
                                                  nn.Conv2d(in_channels=self.feat_size,
                                                            out_channels=self.na * self.nc,
                                                            kernel_size=3, padding=1))
        initialize_vals(self.cls_modl)
    def forward(self, inp):
#         import pdb; pdb.set_trace();
        out = self.cls_modl(inp)
        out2 = flatten_conv(out, self.na)
        return out2

class regress_model(nn.Module):
    def __init__(self, fs=256, na=9, nc=80):
        super().__init__()
        self.na = na
        self.nc = nc
        self.feat_size = fs
        self.reg_model = nn.Sequential(*nn.ModuleList([nn.Conv2d(in_channels=self.feat_size,
                                                                           out_channels=self.feat_size,
                                                                           kernel_size=3, padding=1)]*4),
                                                  nn.Conv2d(in_channels=self.feat_size,
                                                            out_channels=self.na * 4,
                                                            kernel_size=3, padding=1))
        initialize_vals(self.reg_model)
    def forward(self, inp):
        out = self.reg_model(inp)
        out2 = flatten_conv(out, self.na)
        return out2
    
    
class retina_net_model(nn.Module):
    def __init__(self, resnet_model, na=9, nc=81):
        super().__init__()
        self.res_backbone = resnet_model
        self.fpn_sizes = [self.res_backbone.layer2[-1].conv3.out_channels, 
                          self.res_backbone.layer3[-1].conv3.out_channels,
                          self.res_backbone.layer4[-1].conv3.out_channels]
        self.feat_size = 256
        self.num_anch = na
        self.num_class = nc
        self.fpn = FPN_backbone(self.fpn_sizes)
        self.cls_model = classf_model(self.feat_size, self.num_anch, self.num_class)
        self.reg_model = regress_model(self.feat_size, self.num_anch, self.num_class)
        
        
    def forward(self, inp):
        x = self.res_backbone.conv1(inp)
        x = self.res_backbone.bn1(x)
        x = self.res_backbone.relu(x)
        x = self.res_backbone.maxpool(x)
        x1 = self.res_backbone.layer1(x)
        x2 = self.res_backbone.layer2(x1)
        x3 = self.res_backbone.layer3(x2)
        x4 = self.res_backbone.layer4(x3)

        features = self.fpn([x2, x3, x4])
#         features = self.fpn([x4])
        out_cls = []
        out_bbx = []
        for p in features:
            out_cls.append(self.cls_model(p))
            out_bbx.append(self.reg_model(p))
        
        return [torch.cat(out_cls, dim=1),
                torch.cat(out_bbx, dim=1)]
    
def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes+1)
        t = V(t[:,:-1].contiguous())#.cpu()
        x = pred[:,:-1]
        w = self.get_weight(x,t)
        return F.binary_cross_entropy_with_logits(x, t, w, reduction='element_wise_mean')/self.num_classes
    
    def get_weight(self,x,t): return None
    
class FocalLoss(BCE_Loss):
    def get_weight(self,x,t):
        alpha,gamma = 0.25,1
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt).pow(gamma)

    
def intersect(box_a, box_b):
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def box_sz(b): return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union

def get_y(bbox,clas):
    bbox = bbox.view(-1,4)/sz
    bb_keep = ((bbox[:,2]-bbox[:,0])>0).nonzero()[:,0]
    return bbox[bb_keep],clas[bb_keep]


def actn_to_bb(actn, anchors):
#     import pdb; pdb.set_trace()

    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
    actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
    return hw2corners(actn_centers, actn_hw)

def map_to_ground_truth(overlaps, print_it=False):
    prior_overlap, prior_idx = overlaps.max(1)
    if print_it: print(prior_overlap)
#     pdb.set_trace()
    gt_overlap, gt_idx = overlaps.max(0)
    gt_overlap[prior_idx] = 1.99
    for i,o in enumerate(prior_idx): gt_idx[o] = i
    return gt_overlap,gt_idx

# loss_f = BCE_Loss(len(id2cat))
loss_f = FocalLoss(len(id2cat))
# loss_f = nn.CrossEntropyLoss()

def ssd_1_loss(b_c,b_bb,bbox,clas,print_it=False):
#     import pdb; pdb.set_trace()
    bbox,clas = get_y(bbox,clas)
    a_ic = actn_to_bb(b_bb, anchors)
    overlaps = jaccard(bbox.data, anchor_cnr.data)
    gt_overlap,gt_idx = map_to_ground_truth(overlaps,print_it)
    gt_clas = clas[gt_idx]
    pos = gt_overlap > 0.4
    pos_idx = torch.nonzero(pos)[:,0]
    gt_clas[1-pos] = len(id2cat)
    gt_bbox = bbox[gt_idx]
    loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
    clas_loss  = loss_f(b_c, gt_clas)
    return loc_loss, clas_loss

def ssd_loss(pred,targ,print_it=False):
    lcs,lls = 0.,0.
    for b_c,b_bb,bbox,clas in zip(*pred,*targ):
        loc_loss,clas_loss = ssd_1_loss(b_c,b_bb,bbox,clas,print_it)
        lls += loc_loss
        lcs += clas_loss
#     if print_it: 
#     print(f'loc: {lls.data[0]}, clas: {lcs.data[0]}')
    return lls+lcs

def coco_metrics(preds, targs):
    lcs,lls = 0.,0.
    for b_c,b_bb,bbox,clas in zip(*preds,*targs):
        loc_loss,clas_loss = ssd_1_loss(b_c,b_bb,bbox,clas,False)
        lls += loc_loss
        lcs += clas_loss
    return to_np(lls + lcs)

retina_model = retina_net_model(res50, k, n_clas)

learn = ConvLearner.from_model_data(retina_model, md)
learn.crit = ssd_loss
learn.opt_fn = optim.Adam
learn.metrics = [coco_metrics]
learn.fit(1e-3, 1, cycle_len=10, best_save_name='retina_first_try')