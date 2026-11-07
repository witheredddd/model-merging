"""Read-only nonlinear step audit at a saved training checkpoint; no run mutation."""
import json,math,time
import torch
import train as t

def main():
    torch.set_num_threads(8);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(t.OUT/'resume.pt',map_location='cpu',weights_only=False)
    B=ck['B'].cuda();z=ck['z'].cuda();epoch=ck['epoch'];generator=ck['data_rng'];del ck
    enc,model,found,base=t.runtime.build_model(t.u,t.u.ClassificationHead.load(str(t.HEAD)))
    for p in model.parameters():p.requires_grad_(False)
    for _,p in found:p.requires_grad_(True)
    ed,dl,_,_=t.u.data(enc,42);ed.epoch=epoch+1;dl.generator.set_state(generator);model.train();rows=[]
    @torch.no_grad()
    def ce(theta,x,y):
        t.u.assign_flat(found,theta);value=0.
        for j in range(0,len(y),32):value+=float(torch.nn.functional.cross_entropy(model(x[j:j+32].cuda()),y[j:j+32].cuda(),reduction='sum'))/len(y)
        return value
    for i,(x,y,ix) in enumerate(dl):
        if i==3:break
        t.u.materialize(found,base,B,z);g,before=t.gradient(model,found,x,y,32)
        with torch.no_grad():
            h=B.T@g;r=g-B@h;r-=B@(B.T@r);rn=float(r.norm());zn=float(z.norm());v=z/zn
            step=epoch*len(dl)+i+1;factor=.01+.99*.5*(1+math.cos(math.pi*(step-len(dl))/(19*len(dl))))
            lr=.01*factor;angle=min(.05,.1*factor*rn*zn);newz=z-lr*h
            a=(math.cos(angle)-1)*(B@v)-math.sin(angle)*(r/rn)
            coordinate_theta=base+B@newz;basis_theta=base+B@z+a*(v@z);joint_theta=coordinate_theta+a*(v@newz)
            rows.append(dict(probe=i,before_ce=before,coordinate_only_ce=ce(coordinate_theta,x,y),basis_only_ce=ce(basis_theta,x,y),
                joint_ce=ce(joint_theta,x,y),linear_predicted_CE_change=-lr*float(h.square().sum())-angle*rn*zn,
                indices=ix.tolist()))
    result=dict(read_only=True,checkpoint_epoch=epoch,training_batches_only=True,checks=rows,time=time.time())
    t.write('nonlinear_step_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
