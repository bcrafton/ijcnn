
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('forward_pwr.sw0');
lssig(sig)

vg     = evalsig(sig, 'v_vg');
r      = evalsig(sig, 'r_rmem');
p      = evalsig(sig, 'p_vsvd');
pr     = evalsig(sig, 'p_rmem');
pn     = evalsig(sig, 'p_mnmos');

csvwrite('forward_pwr_vg.csv',     vg);
csvwrite('forward_pwr_r.csv',      r);
csvwrite('forward_pwr_p.csv',      p);
csvwrite('forward_pwr_pr.csv',     pr);
csvwrite('forward_pwr_pn.csv',     pn);
