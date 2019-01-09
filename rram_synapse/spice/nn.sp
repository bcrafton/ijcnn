
***************************
.subckt neuron vdd vmem vspk

m20 vmem vlk   gnd gnd nmos w=45n l=30n

m19 vmem vca   gnd gnd nmos w=45n l=30n
m18 vca  vspk  v1  gnd nmos w=45n l=30n
m17 v1   v1    gnd gnd nmos w=45n l=30n

m16 v1   vadp  v2  vdd pmos w=45n l=30n
m15 v2   vo1   vdd vdd pmos w=45n l=30n

m13 vspk vo1   vdd vdd pmos w=45n l=30n
m14 vspk vo1   gnd gnd nmos w=45n l=30n

m12 vmem vo2   gnd gnd nmos w=45n l=30n

m11 v3   vrfr  gnd gnd nmos w=45n l=30n
m10 vo2  vo1   v3  gnd nmos w=45n l=30n
m9  vo2  vo1   v4  vdd pmos w=45n l=30n
m8  v4   v4    vdd vdd pmos w=45n l=30n

m7 vmem  v5    v6  vdd pmos w=45n l=30n
m6 v6    vo1   vdd vdd pmos w=45n l=30n

m5 vo1   vin   gnd gnd nmos w=45n l=30n
m4 vo1   vin   v5  vdd pmos w=45n l=30n
m3 v5    v5    vdd vdd pmos w=45n l=30n

m2 vdd   vmem  vin gnd nmos w=45n l=30n
m1 vin   vsf   gnd gnd nmos w=45n l=30n

***************************
cmem vmem gnd 500f
c1   vo2  gnd 100f
c2   vca  gnd 100f
***************************
* sources
vs1 vlk  gnd dc 0.2
* pmos
vs2 vadp gnd dc 0.9
* vsf = 0.65
vs3 vsf  gnd dc 0.25
* rfr = 300, 350, 450
vs4 vrfr gnd dc 0.2
***************************
.ends neuron
*****************************
.subckt inv vdd i o
mpmos1 o i vdd vdd pmos w=45n l=30n
mnmos1 o i gnd gnd nmos w=45n l=30n
.ends inv
*****************************
.subckt nor vdd a b o
mpmos1 x a vdd vdd pmos w=45n l=30n
mpmos2 o b x   vdd pmos w=45n l=30n
mnmos1 o a gnd gnd nmos w=45n l=30n
mnmos2 o b gnd gnd nmos w=45n l=30n
.ends nor
*****************************
.subckt nand vdd a b o
mpmos1 o a vdd vdd pmos w=45n l=30n
mpmos2 o b vdd vdd pmos w=45n l=30n
mnmos1 o a x   gnd nmos w=45n l=30n
mnmos2 x b gnd gnd nmos w=45n l=30n
.ends nand
*****************************
.subckt or vdd a b o
mpmos1 x a vdd vdd pmos w=45n l=30n
mpmos2 y b x   vdd pmos w=45n l=30n
mnmos1 y a gnd gnd nmos w=45n l=30n
mnmos2 y b gnd gnd nmos w=45n l=30n
mpmos3 o y vdd vdd pmos w=45n l=30n
mnmos3 o y gnd gnd nmos w=45n l=30n
.ends or
*****************************
.subckt memristor plus minus Ron=1meg Roff=100meg Rini=20meg
.param uv=0.00001f D=10n k='uv*Ron/D**2' a='(Rini-Ron)/(Roff-Rini)'
Rmem plus minus R='Roff + (Ron-Roff)/(1 + a * exp(-4 * k * V(Q)))'
Gx   gnd Q cur='i(Rmem)'
Cint Q gnd 1u
Raux Q gnd 100meg
.ends memristor
*****************************
.subckt synapse vdd pre post out

* pren
xinv1 vdd pre pren inv

* PREX
mpmos3 pre_out vb1  vdd     vdd pmos w=45n l=30n
mpmos4 gnd     pre  pre_out vdd pmos w=45n l=30n
cap1   vdd pre_out 5f

* POSTX
mpmos5 post_out vb2  vdd      vdd pmos w=45n l=30n
mpmos6 gnd      post post_out vdd pmos w=45n l=30n
cap2   vdd  post_out 5f

* prex postx
xnor1 vdd pre_out post_out inc nor

* PREX postx
xinv2 vdd pre_out  pre_outn inv
xnor2 vdd pre_outn post_out dec nor 

* prex postx or pre
xor1 vdd inc pren read or 

* control
mnmos23 vdd read vp  gnd nmos w=45n l=30n
mnmos24 vdd dec  vm  gnd nmos w=45n l=30n
mnmos25 vp  dec  gnd gnd nmos w=45n l=30n
mnmos26 vm  inc  gnd gnd nmos w=45n l=30n

xmemr vp vm memristor

* READ
mnmos21 vm pren gnd gnd nmos w=45n l=30n
* cap3 out gnd 50f

* sources
vs1 vb1  gnd dc 1.075
vs2 vb2  gnd dc 1.075

* Ex out gnd val='i(mnmos21)'
Ex out gnd val='i(xmemr.Rmem) * (v(pren)>0.5)'

.ends synapse
*****************************

* sources
vs1 vdd  gnd dc 1.1

* is1 vdd vin pulse( 0 5n 100u 0.1n 0.1n 25u 200u )

vs2 pre1 gnd pulse( 1.1 0 100u 0.1n 0.1n 25u 0.8m )
vs3 pre2 gnd pulse( 1.1 0 300u 0.1n 0.1n 25u 0.8m )
vs4 pre3 gnd pulse( 1.1 0 500u 0.1n 0.1n 25u 0.8m )
vs5 pre4 gnd pulse( 1.1 0 700u 0.1n 0.1n 25u 0.8m )

* vs6 post gnd pulse( 1.1 0 50m 0.1n 0.1n 0.01m 100m )
* vs6 post gnd dc 1.1

xsyn1 vdd pre1 post out1 synapse
xsyn2 vdd pre2 post out2 synapse
xsyn3 vdd pre3 post out3 synapse
xsyn4 vdd pre4 post out4 synapse

* is1 vdd vin PL(0 0 0 4m 1n 4.0001m) 

* for mnmos21
* Gx vdd vin val='( abs(v(out1) + v(out2) + v(out3) + v(out4)) / 1e2 )'
* for xmemr.Rmem
Gx vdd vin val='( abs(v(out1) + v(out2) + v(out3) + v(out4)) / 5 )'

xneuron1 vdd vin postn neuron
xinv1 vdd postn post inv

*****************************
.tran 1u 1
.option post=2 method=gear
.probe v(x*.*) i(x*.*)
*****************************

* PTM Low Power 45nm Metal Gate / High-K / Strained-Si
* nominal Vdd = 1.1V

.model  nmos  nmos  level = 54

+version = 4.0             binunit = 1               paramchk= 1               mobmod  = 0             
+capmod  = 2               igcmod  = 1               igbmod  = 1               geomod  = 1             
+diomod  = 1               rdsmod  = 0               rbodymod= 1               rgatemod= 1             
+permod  = 1               acnqsmod= 0               trnqsmod= 0             

+tnom    = 27              toxe    = 1.8e-009        toxp    = 1.5e-009        toxm    = 1.8e-009      
+dtox    = 3e-010          epsrox  = 3.9             wint    = 5e-009          lint    = 0             
+ll      = 0               wl      = 0               lln     = 1               wln     = 1             
+lw      = 0               ww      = 0               lwn     = 1               wwn     = 1             
+lwl     = 0               wwl     = 0               xpart   = 0               toxref  = 1.8e-009      

+vth0    = 0.62261         k1      = 0.4             k2      = 0               k3      = 0             
+k3b     = 0               w0      = 2.5e-006        dvt0    = 1               dvt1    = 2             
+dvt2    = 0               dvt0w   = 0               dvt1w   = 0               dvt2w   = 0             
+dsub    = 0.1             minv    = 0.05            voffl   = 0               dvtp0   = 1e-010        
+dvtp1   = 0.1             lpe0    = 0               lpeb    = 0               xj      = 1.4e-008      
+ngate   = 1e+023          ndep    = 3.24e+018       nsd     = 2e+020          phin    = 0             
+cdsc    = 0               cdscb   = 0               cdscd   = 0               cit     = 0             
+voff    = -0.13           nfactor = 1.6             eta0    = 0.0125          etab    = 0             
+vfb     = -0.55           u0      = 0.049           ua      = 6e-010          ub      = 1.2e-018      
+uc      = 0               vsat    = 130000          a0      = 1               ags     = 0             
+a1      = 0               a2      = 1               b0      = 0               b1      = 0             
+keta    = 0.04            dwg     = 0               dwb     = 0               pclm    = 0.02          
+pdiblc1 = 0.001           pdiblc2 = 0.001           pdiblcb = -0.005          drout   = 0.5           
+pvag    = 1e-020          delta   = 0.01            pscbe1  = 8.14e+008       pscbe2  = 1e-007        
+fprout  = 0.2             pdits   = 0.08            pditsd  = 0.23            pditsl  = 2300000       
+rsh     = 5               rdsw    = 210             rsw     = 80              rdw     = 80            
+rdswmin = 0               rdwmin  = 0               rswmin  = 0               prwg    = 0             
+prwb    = 0               wr      = 1               alpha0  = 0.074           alpha1  = 0.005         
+beta0   = 30              agidl   = 0.0002          bgidl   = 2.1e+009        cgidl   = 0.0002        
+egidl   = 0.8             aigbacc = 0.012           bigbacc = 0.0028          cigbacc = 0.002         
+nigbacc = 1               aigbinv = 0.014           bigbinv = 0.004           cigbinv = 0.004         
+eigbinv = 1.1             nigbinv = 3               aigc    = 0.015211        bigc    = 0.0027432     
+cigc    = 0.002           aigsd   = 0.015211        bigsd   = 0.0027432       cigsd   = 0.002         
+nigc    = 1               poxedge = 1               pigcd   = 1               ntox    = 1             
+xrcrg1  = 12              xrcrg2  = 5             

+cgso    = 1.1e-010        cgdo    = 1.1e-010        cgbo    = 2.56e-011       cgdl    = 2.653e-010    
+cgsl    = 2.653e-010      ckappas = 0.03            ckappad = 0.03            acde    = 1             
+moin    = 15              noff    = 0.9             voffcv  = 0.02          

+kt1     = -0.11           kt1l    = 0               kt2     = 0.022           ute     = -1.5          
+ua1     = 4.31e-009       ub1     = 7.61e-018       uc1     = -5.6e-011       prt     = 0             
+at      = 33000         

+fnoimod = 1               tnoimod = 0             

+jss     = 0.0001          jsws    = 1e-011          jswgs   = 1e-010          njs     = 1             
+ijthsfwd= 0.01            ijthsrev= 0.001           bvs     = 10              xjbvs   = 1             
+jsd     = 0.0001          jswd    = 1e-011          jswgd   = 1e-010          njd     = 1             
+ijthdfwd= 0.01            ijthdrev= 0.001           bvd     = 10              xjbvd   = 1             
+pbs     = 1               cjs     = 0.0005          mjs     = 0.5             pbsws   = 1             
+cjsws   = 5e-010          mjsws   = 0.33            pbswgs  = 1               cjswgs  = 3e-010        
+mjswgs  = 0.33            pbd     = 1               cjd     = 0.0005          mjd     = 0.5           
+pbswd   = 1               cjswd   = 5e-010          mjswd   = 0.33            pbswgd  = 1             
+cjswgd  = 5e-010          mjswgd  = 0.33            tpb     = 0.005           tcj     = 0.001         
+tpbsw   = 0.005           tcjsw   = 0.001           tpbswg  = 0.005           tcjswg  = 0.001         
+xtis    = 3               xtid    = 3             

+dmcg    = 0               dmci    = 0               dmdg    = 0               dmcgt   = 0             
+dwj     = 0               xgw     = 0               xgl     = 0             

+rshg    = 0.4             gbmin   = 1e-010          rbpb    = 5               rbpd    = 15            
+rbps    = 15              rbdb    = 15              rbsb    = 15              ngcon   = 1             


.model  pmos  pmos  level = 54


+version = 4.0             binunit = 1               paramchk= 1               mobmod  = 0             
+capmod  = 2               igcmod  = 1               igbmod  = 1               geomod  = 1             
+diomod  = 1               rdsmod  = 0               rbodymod= 1               rgatemod= 1             
+permod  = 1               acnqsmod= 0               trnqsmod= 0             

+tnom    = 27              toxe    = 1.82e-009       toxp    = 1.5e-009        toxm    = 1.82e-009     
+dtox    = 3.2e-010        epsrox  = 3.9             wint    = 5e-009          lint    = 0             
+ll      = 0               wl      = 0               lln     = 1               wln     = 1             
+lw      = 0               ww      = 0               lwn     = 1               wwn     = 1             
+lwl     = 0               wwl     = 0               xpart   = 0               toxref  = 1.82e-009     

+vth0    = -0.587          k1      = 0.4             k2      = -0.01           k3      = 0             
+k3b     = 0               w0      = 2.5e-006        dvt0    = 1               dvt1    = 2             
+dvt2    = -0.032          dvt0w   = 0               dvt1w   = 0               dvt2w   = 0             
+dsub    = 0.1             minv    = 0.05            voffl   = 0               dvtp0   = 1e-011        
+dvtp1   = 0.05            lpe0    = 0               lpeb    = 0               xj      = 1.4e-008      
+ngate   = 1e+023          ndep    = 2.44e+018       nsd     = 2e+020          phin    = 0             
+cdsc    = 0               cdscb   = 0               cdscd   = 0               cit     = 0             
+voff    = -0.126          nfactor = 1.8             eta0    = 0.0125          etab    = 0             
+vfb     = 0.55            u0      = 0.021           ua      = 2e-009          ub      = 5e-019        
+uc      = 0               vsat    = 90000           a0      = 1               ags     = 1e-020        
+a1      = 0               a2      = 1               b0      = 0               b1      = 0             
+keta    = -0.047          dwg     = 0               dwb     = 0               pclm    = 0.12          
+pdiblc1 = 0.001           pdiblc2 = 0.001           pdiblcb = 3.4e-008        drout   = 0.56          
+pvag    = 1e-020          delta   = 0.01            pscbe1  = 8.14e+008       pscbe2  = 9.58e-007     
+fprout  = 0.2             pdits   = 0.08            pditsd  = 0.23            pditsl  = 2300000       
+rsh     = 5               rdsw    = 250             rsw     = 75              rdw     = 75            
+rdswmin = 0               rdwmin  = 0               rswmin  = 0               prwg    = 0             
+prwb    = 0               wr      = 1               alpha0  = 0.074           alpha1  = 0.005         
+beta0   = 30              agidl   = 0.0002          bgidl   = 2.1e+009        cgidl   = 0.0002        
+egidl   = 0.8             aigbacc = 0.012           bigbacc = 0.0028          cigbacc = 0.002         
+nigbacc = 1               aigbinv = 0.014           bigbinv = 0.004           cigbinv = 0.004         
+eigbinv = 1.1             nigbinv = 3               aigc    = 0.0097          bigc    = 0.00125       
+cigc    = 0.0008          aigsd   = 0.0097          bigsd   = 0.00125         cigsd   = 0.0008        
+nigc    = 1               poxedge = 1               pigcd   = 1               ntox    = 1             
+xrcrg1  = 12              xrcrg2  = 5             

+cgso    = 1.1e-010        cgdo    = 1.1e-010        cgbo    = 2.56e-011       cgdl    = 2.653e-010    
+cgsl    = 2.653e-010      ckappas = 0.03            ckappad = 0.03            acde    = 1             
+moin    = 15              noff    = 0.9             voffcv  = 0.02          

+kt1     = -0.11           kt1l    = 0               kt2     = 0.022           ute     = -1.5          
+ua1     = 4.31e-009       ub1     = 7.61e-018       uc1     = -5.6e-011       prt     = 0             
+at      = 33000         

+fnoimod = 1               tnoimod = 0             

+jss     = 0.0001          jsws    = 1e-011          jswgs   = 1e-010          njs     = 1             
+ijthsfwd= 0.01            ijthsrev= 0.001           bvs     = 10              xjbvs   = 1             
+jsd     = 0.0001          jswd    = 1e-011          jswgd   = 1e-010          njd     = 1             
+ijthdfwd= 0.01            ijthdrev= 0.001           bvd     = 10              xjbvd   = 1             
+pbs     = 1               cjs     = 0.0005          mjs     = 0.5             pbsws   = 1             
+cjsws   = 5e-010          mjsws   = 0.33            pbswgs  = 1               cjswgs  = 3e-010        
+mjswgs  = 0.33            pbd     = 1               cjd     = 0.0005          mjd     = 0.5           
+pbswd   = 1               cjswd   = 5e-010          mjswd   = 0.33            pbswgd  = 1             
+cjswgd  = 5e-010          mjswgd  = 0.33            tpb     = 0.005           tcj     = 0.001         
+tpbsw   = 0.005           tcjsw   = 0.001           tpbswg  = 0.005           tcjswg  = 0.001         
+xtis    = 3               xtid    = 3             

+dmcg    = 0               dmci    = 0               dmdg    = 0               dmcgt   = 0             
+dwj     = 0               xgw     = 0               xgl     = 0             

+rshg    = 0.4             gbmin   = 1e-010          rbpb    = 5               rbpd    = 15            
+rbps    = 15              rbdb    = 15              rbsb    = 15              ngcon   = 1             

.end
