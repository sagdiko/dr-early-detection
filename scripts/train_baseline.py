"""Train a small baseline from a CSV and write metrics/checkpoint.

CSV columns: image_path,label,patient_id. This is intentionally explicit so
the research run cannot silently use a different split or leak patient IDs.
"""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
from dr_early_detection.data import FundusDataset
from dr_early_detection.models import build_model

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True); ap.add_argument('--val-csv', required=True)
    ap.add_argument('--out-dir', default='outputs/run'); ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--image-size', type=int, default=224); ap.add_argument('--seed', type=int, default=0)
    args=ap.parse_args(); seed_all(args.seed); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tfm=transforms.Compose([transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor()])
    train=FundusDataset(args.train_csv); val=FundusDataset(args.val_csv)
    train.transform=lambda image: {'image': tfm(image)}; val.transform=lambda image: {'image': tfm(image)}
    tr=DataLoader(train,batch_size=8,shuffle=True); va=DataLoader(val,batch_size=8)
    model=build_model('resnet50',num_classes=2,pretrained=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4); loss_fn=torch.nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        for b in tr:
            opt.zero_grad(); logits=model(b['image'].to(device)); loss=loss_fn(logits,b['label'].to(device)); loss.backward(); opt.step()
        model.eval(); ys=[]; ps=[]
        with torch.no_grad():
            for b in va:
                prob=model(b['image'].to(device)).softmax(1)[:,1].cpu().numpy(); ps.extend(prob); ys.extend(b['label'].numpy())
        pred=(np.array(ps)>=.5).astype(int); metrics={'accuracy':float(accuracy_score(ys,pred))}
        if len(set(ys))==2: metrics['auc']=float(roc_auc_score(ys,ps))
        print({'epoch':epoch+1,**metrics})
    torch.save(model.state_dict(),out/'resnet50_baseline.pt'); (out/'metrics.json').write_text(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
