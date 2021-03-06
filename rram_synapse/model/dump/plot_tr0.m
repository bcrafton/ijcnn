

addpath('../HspiceToolbox/');
colordef none;

sig = loadsig('forward.sw0');
lssig(sig);

vd     = evalsig(sig, 'v_vd');
vg     = evalsig(sig, 'v_vg');
vc     = evalsig(sig, 'v_vc');
vs     = evalsig(sig, 'v_vs');
r      = evalsig(sig, 'r_rmem');
i      = evalsig(sig, 'i_rmem');

l = length(vd)
t = linspace(0, 1, l);

subplot(5,1,1)
plot(t, vd)
subplot(5,1,2)
plot(t, vg)
subplot(5,1,3)
plot(t, vc)
subplot(5,1,4)
plot(t, vs)
subplot(5,1,5)
plot(t, i)
