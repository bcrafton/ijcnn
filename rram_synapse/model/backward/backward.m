
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('synapse_sweep8.sw0');
% sig = loadsig('backward.sw0');
lssig(sig)

vd     = evalsig(sig, 'v_vd');
vg     = evalsig(sig, 'v_vg');
vc     = evalsig(sig, 'v_vc');
r      = evalsig(sig, 'r_rmem');
i      = evalsig(sig, 'i_rmem');

disp(size(vd))
disp(size(vg))
disp(size(vc))
disp(size(r))
disp(size(i))

csvwrite('backward_vd.csv',     vd);
csvwrite('backward_vg.csv',     vg);
csvwrite('backward_vc.csv',     vc);
csvwrite('backward_r.csv',      r);
csvwrite('backward_i.csv',      i);