poso = 100;
velo = 0;
maxTime = 10000;
deltaT = .01;
time = 1:1:maxTime/deltaT;
%Real time is maxTime*deltaT

%Accel slowly changes
accel = zeros(maxTime,1);
accel(1) = 0;
for i = 2:1:maxTime/deltaT
    accel(i) = accel(i-1) + (rand() - .5)/10;
end
plot(time, accel)

vel = zeros(maxTime,1);
vel(1) = velo;
pos = zeros(maxTime,1);
pos(1) = poso;

for i = 2:1:maxTime/deltaT
    vel(i) = vel(i-1) + accel(i)*deltaT;
    pos(i) = pos(i-1) + vel(i)*deltaT;
end

hold on
plot(time,vel)
plot(time,pos)