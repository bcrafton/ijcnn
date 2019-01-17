
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('backward_pwr.sw0');
lssig(sig)

vg     = evalsig(sig, 'v_vg');
r      = evalsig(sig, 'r_rmem');
p      = evalsig(sig, 'p_vsvc');
pr     = evalsig(sig, 'p_rmem');
pn     = evalsig(sig, 'p_mnmos');

csvwrite('backward_vg.csv',     vg);
csvwrite('backward_r.csv',      r);
csvwrite('backward_p.csv',      p);
csvwrite('backward_pr.csv',     pr);
csvwrite('backward_pn.csv',     pn);
