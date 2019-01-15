
addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('forward.sw0');
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

csvwrite('forward_vd.csv',     vd);
csvwrite('forward_vg.csv',     vg);
csvwrite('forward_vc.csv',     vc);
csvwrite('forward_r.csv',      r);
csvwrite('forward_i.csv',      i);