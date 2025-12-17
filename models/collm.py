import torch


class CoLLM:
    def __init__(self, S, L, F, R):
        self.S, self.L, self.F, self.R = S, L, F, R


    @torch.no_grad()
    def inference(self, x, tau1=0.6, tau2=0.05):
        ys, phi_s = self.S(x)
        Qs = self.F(phi_s)
        if Qs.mean() >= tau1:
            return ys
        yl, phi_l = self.L(x)
        Ql = self.R(phi_l)
        return yl if (Qs-Ql).mean() <= tau2 else 0.5*(ys+yl)
