%% load data

%uiopen('C:\Users\hkedw\Documents\GitHub\IVPS\sample\results\results.csv',1)
%save("results.mat");

load("results.mat");

%% load data
row = size(results.x,1);
odd = 1:2:row;
even = 2:2:row;

x = results.x(odd);
y = results.y(odd);
ex = results.ex(odd);
ey = results.ey(odd);
stdex = results.ex(even);
stdey = results.ey(even);

erx = abs(ex-x);
ery = abs(ey-y);

%% show error 1
cmap = colormap;
nerx = erx/max(erx);
nery = ery/max(ery);

figure(1);
clf
for i=1:row/2
    plot(x(i),y(i),'o','Color',cmap(ceil(nerx(i)*64),:))
    hold on
end
grid on
hold off
xlim([-2 1.5])
ylim([-1.5 1.5])
title('xerror');
c = colorbar

figure(2);
clf
for i=1:row/2
    plot(x(i),y(i),'o','Color',cmap(ceil(nery(i)*64),:))
    hold on
end
grid on
hold off
xlim([-2 1.5])
ylim([-1.5 1.5])
title('yerror');
colorbar

%%
vx = ex -x;
vy = ey -y;

figure(3)
for i=1:row/2
    quiver(x(i),y(i),vx(i)*10,vy(i)*10)
    hold on
end
