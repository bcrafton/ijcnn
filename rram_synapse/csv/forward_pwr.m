
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('forward_pwr.sw0');
lssig(sig)

vg     = evalsig(sig, 'v_vg');
r      = evalsig(sig, 'r_rmem');
p      = evalsig(sig, 'p_vsvd');
pr     = evalsig(sig, 'p_rmem');
pn     = evalsig(sig, 'p_mnmos');

csvwrite('forward_vg.csv',     vg);
csvwrite('forward_r.csv',      r);
csvwrite('forward_p.csv',      p);
csvwrite('forward_pr.csv',     pr);
csvwrite('forward_pn.csv',     pn);
