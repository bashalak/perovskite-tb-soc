set terminal postscript eps color enhanced font 'Helvetica,20' linewidth 1
set output "ssgw-band.eps"
set size square
set key off

x1=0
x2=2.7882500
y1=-1.2
y2=3.8
haev = 27.2113
bohr = 0.529177
vbt = -0.33
cbb = 1.24
kshift = 1.2104420
mvxx = 0.209
mvyy = 0.234
mvzz = 0.207
mcxx = 0.236
mcyy = 0.199
mczz = 0.245
set ytics 1.0
set ylabel offset 0.5
set ylabel "Energy (eV)"
set xr [x1:x2]
set yr [y1:y2]
# set arrow from  x1,y1 to  x1,y2 nohead lw 0.5 front
set arrow from 0.3556,y1 to 0.3556,y2 nohead lw 0.5 front
set arrow from 0.6039,y1 to 0.6039,y2 nohead lw 0.5 front 
set arrow from 0.9437,y1 to 0.9437,y2 nohead lw 0.5 front 
set arrow from 1.2129,y1 to 1.2129,y2 nohead lw 0.5 front 
set arrow from 1.5836,y1 to 1.5836,y2 nohead lw 0.5 front 
set arrow from 1.9449,y1 to 1.9449,y2 nohead lw 0.5 front 
set arrow from 2.2008,y1 to 2.2008,y2 nohead lw 0.5 front 
set arrow from 2.5420,y1 to 2.5420,y2 nohead lw 0.5 front 
# set arrow from x2,y1 to x2,y2 nohead lw 0.5 front 
unset xtics

set xtics ("{/Symbol G}" x1, "X" 0.3556, "S" 0.6039, "Y" 0.9437, "{/Symbol G}" 1.2129, "Z" 1.5836, "U" 1.9449, "R" 2.2008, "T" 2.5420, "Z" x2)

plot "ss-gw.band.dat" u 1:($2-vbt)  w l lt 1 lc "blue" lw 3
