
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('backward_pwr.sw0');
lssig(sig)

vg     = evalsig(sig, 'v_vg');
r      = evalsig(sig, 'r_rmem');
p      = evalsig(sig, 'p_vsvc');
pr     = evalsig(sig, 'p_rmem');
pn     = evalsig(sig, 'p_mnmos');

csvwrite('backward_pwr_vg.csv',     vg);
csvwrite('backward_pwr_r.csv',      r);
csvwrite('backward_pwr_p.csv',      p);
csvwrite('backward_pwr_pr.csv',     pr);
csvwrite('backward_pwr_pn.csv',     pn);
