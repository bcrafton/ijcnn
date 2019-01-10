

addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('synapse_sweep3.sw0');
lssig(sig)

vd     = evalsig(sig, 'v_vd');
vg     = evalsig(sig, 'v_vg');
vc     = evalsig(sig, 'v_vc');
r      = evalsig(sig, 'r_rmem');
i      = evalsig(sig, 'i_rmem');

csvwrite('vd.csv',     vd);
csvwrite('vg.csv',     vg);
csvwrite('vc.csv',     vc);
csvwrite('r.csv',      r);
csvwrite('i.csv',      i);