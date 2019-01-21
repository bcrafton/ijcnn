cp spice/synapse_sweep7.sw0 forward/;
cp spice/forward_pwr.sw0    forward/;
cp spice/synapse_sweep8.sw0 backward/;
cp spice/backward_pwr.sw0   backward/;
cd forward;
matlab -r "run forward.m; quit;";
matlab -r "run forward_pwr.m; quit;";
python forward_model.py;
cd ..;
cd backward;
matlab -r "run backward.m; quit;";
matlab -r "run backward_pwr.m; quit;";
python backward_model.py;
cd ..;

