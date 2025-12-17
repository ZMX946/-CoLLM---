import torch, torch.nn.functional as F
from datasets.cmapss import CMAPSSDataset
from models.small import SmallModel
from models.gpt2_ts import GPT2TimeSeries
from models.fuzzy import FuzzyDecisionAgent
from models.reflection import SelfReflection


def Qstar(p,y,a=5): return 1-torch.tanh(torch.abs(p-y)/a)

device = torch.device('cpu')
S = SmallModel().to( device); S.load_state_dict(torch.load('small.pt'))
L = GPT2TimeSeries().to( device); L.load_state_dict(torch.load('large.pt'))
Fz = FuzzyDecisionAgent(32,50).to( device)
Rf = SelfReflection(768,12).to( device)


opt = torch.optim.Adam(list(Fz.parameters())+list(Rf.parameters()),1e-3)
loader = torch.utils.data.DataLoader(CMAPSSDataset('../data/CMAPSS/train_FD001.txt'),128,True)
EPOCHS = 100
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x,y in loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad(): ys,ps=S(x); yl,pl=L(x)
        loss = F.mse_loss(Fz(ps),Qstar(ys,y)) + F.mse_loss(Rf(pl),Qstar(yl,y))
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item() * x.size(0)
    epoch_loss /= len(loader.dataset)
    print(f'Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f}')

torch.save(Fz.state_dict(),'fuzzy.pt'); torch.save(Rf.state_dict(),'reflect.pt')
