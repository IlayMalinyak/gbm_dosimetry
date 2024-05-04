close all
clear

% User input
imgs_folder = "C:\Users\guyhe\Documents\University Stuff\DART\Numerical Solutions stuff\TG43 calculations\GBM configuration\";

Gamma_Ra0_uCi = 3; % source activity in micro-Curie
Lseed = 10; % source length [mm]
Tmax_days = 30; % treatment duration [days]

Ncones = 2;
Nseeds_per_cone = 7;
% d_cones_vec = 10; % distance between cone bases (along z, mm)
d_cones_vec = 3 : 1 : 10;

alpha_deg = 19; % angle of the sources w.r.t the needle axis, in degrees
alpha = alpha_deg*pi/180; 
Rin = 2.385; % radius of inner circle (where the seed tails are located)

dose_target = 20; % the 'cold spot' is defined as below this threshold.
% dose_levels = [1:9 10:10:50]; % for contour plot
dose_levels = dose_target;
zview = 0; %d_cones-1*Lseed*cos(alpha); % view plane z (mm)
zview_vec_len = 1;

pad = 0; % This values will be added, after the 'dose_target' distance, to
% the radius and length of the cylinder where we look at the cold spot. 
% For example, if 'pad=1' and the radius to the dose_target is 3 mm from the center,
% the cylinder radius for the cold spot calculation will be 4 mm. The same
% for z.

% Load DART2D solution

% load DART2D_sol_LRn030_LPb060_LBi0006_PleakPb050_Time30d_l10_R035_16-May-2023
% load DART2D_sol_LRn030_LPb020_LBi002_PleakPb080_Time30d_l10_R0185_20-Jun-2023
load DART2D_sol_LRn030_LPb060_LBi0006_PleakPb050_Time30d_l10_R0185_31-Jul-2023

f_plt_heatmap = 0;
f_plt_cont = 1;
f_plot_min_dose_vs_z = 0;
f_plot_cold_spot_ratio = 1;
f_export_images = 0;

Gamma_Ra0 = DART2D_sol.Gamma_Ra_0/3.7e4; % convert to uCi
LRn = DART2D_sol.L_Rn;
LPb = DART2D_sol.L_Pb;
dose_DART2D = DART2D_sol.TotalDose_2D; % Total alpha dose from DART2D
r_DART2D = DART2D_sol.r; % mm
z_DART2D = DART2D_sol.z; % mm
[R_DART2D,Z_DART2D] = meshgrid(r_DART2D,z_DART2D);
dose_DART2D(R_DART2D < DART2D_sol.R0 & abs(Z_DART2D) < 0.5*DART2D_sol.l) = max(max(dose_DART2D)); % Filling the inside of the source with the maximum dose. This will prevent interpolation shannanigans later.
% Prepare r,theta dose lookup table
rmax = min(max(r_DART2D),max(z_DART2D)); % max radius for the r,theta lookup table (LUT)
r_LUT = linspace(0.01,rmax,200);
theta_LUT = linspace(0,180,400);
[R_LUT,THETA_LUT] = meshgrid(r_LUT,theta_LUT);
X = R_LUT.*sin(THETA_LUT*pi/180);
Z = R_LUT.*cos(THETA_LUT*pi/180);
dose_LUT = interp2(R_DART2D,Z_DART2D,dose_DART2D,X,Z);

% text_string = "L_{Rn} = " + num2str(DART2D_sol.L_Rn) + "mm \n L_{Pb} = " + num2str(DART2D_sol.L_Pb) + " mm \n P_{leak}^{Pb} = " + num2str(DART2D_sol.P_leak_pb);
text_string = "L_{Rn} = " + num2str(DART2D_sol.L_Rn) + "mm" + newline + "L_{Pb} = " + num2str(DART2D_sol.L_Pb) + " mm" + newline + "P_{leak}^{Pb} = " + num2str(DART2D_sol.P_leak_pb);

% This is readable but very slow. I checked the short version above gives
% identical results
%{
for i = 1:length(r_LUT)
    for j = 1:length(theta_LUT)
        xij = r_LUT(i)*sin(theta_LUT(j)*pi/180);
        zij = r_LUT(i)*cos(theta_LUT(j)*pi/180);
        dose_LUT(i,j) = interp2(R_DART2D,Z_DART2D,dose_DART2D,xij,zij);
    end
end
%}

%{
figure(1)
clf
h = pcolor(R_LUT,THETA_LUT,dose_LUT);
colormap('jet')
colorbar
set(gca,'ColorScale','log')
set(h,'EdgeColor','none')

figure(2)
clf
h = pcolor(X,Z,dose_LUT);
colorbar
colormap('jet')
set(gca,'ColorScale','log')
set(h,'EdgeColor','none')
axis equal

figure(3)
clf
[cs,h1] = contour(X,Z,dose_LUT,[1 5 10 20 30 50 100]);
axis equal
colormap('jet')
colorbar
caxis([0 50])
xlabel('X [mm]')
ylabel('Y [mm]')
%clabel(cs,h1,'manual')

error('hi')
%}



% z0 = 0;
s = 1;
z0_mat = zeros(length(d_cones_vec), zview_vec_len);
min_dose_mat = zeros(length(d_cones_vec), zview_vec_len);
cold_spot_ratio_vec = zeros(length(d_cones_vec), 1);

for d_cones = d_cones_vec
Rout = Rin + Lseed*sin(alpha);
seed_xyz = [];
Nseeds = Ncones*Nseeds_per_cone;

z0 = -0.5 * d_cones;
for nc = 1:Ncones
    for ns = 1:Nseeds_per_cone
        if mod(nc,2) == 1 % odd cones
            phi = (ns-1)*2*pi/Nseeds_per_cone;
        end
        if mod(nc,2) == 0 % even cones
            phi = (ns-0.5)*2*pi/Nseeds_per_cone;
        end
        x1 = Rin*cos(phi);
        y1 = Rin*sin(phi);
        z1 = z0 + (nc-1)*d_cones;
        x2 = Rout*cos(phi);
        y2 = Rout*sin(phi);
        z2 = z1 - Lseed*cos(alpha);
        seed_xyz = [seed_xyz; x1 y1 z1 x2 y2 z2];
    end
end

% XY view
W = 8;
% view plane coordinates
dx=0.1; % mm
dy=0.1; % mm


i=1;
zview_vec = linspace(z0, -z0, zview_vec_len) ;

for zview = zview_vec
    [doseXY, xmat, ymat, ~, yvec] = calc_dose_XY(W, dx, dy, zview, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels, f_plt_heatmap, f_plt_cont);
    min_dose_mat(s, i) = find_min_dose_XY(doseXY, xmat, ymat, Rin, Rout);
    i = i+1;
end

z0_mat(s, :) = zview_vec;

% XZ view
W = 8;
% view plane coordinates
dx=0.05; % mm
Wz = 25;
dz=0.05; % mm
y0 = 0;
disp(d_cones);
% [doseXZ, ~, ~, ~, ~, ~] = calc_dose_XZ(W, dx, Wz, dz, y0, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels,...
%     f_plt_heatmap, f_plt_cont, d_cones, f_plot_cold_spot_ratio, dose_target, pad);


% XYZ calc
dx = 0.1;
dz = 0.1;
y0 = 0; % This is the plane for the XZ visualization

[doseXYZ, xmat, zmat, xvec, zvec, cold_spot_ratio] = calc_dose_XYZ(W, dx, Wz, dz, y0, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels,...
    f_plt_heatmap, f_plt_cont, d_cones, f_plot_cold_spot_ratio, dose_target, pad, imgs_folder, LRn, LPb, text_string, Rin, alpha_deg, f_export_images);
cold_spot_ratio_vec(s) = cold_spot_ratio;
s = s+1;
end


if f_plot_min_dose_vs_z == 1
    figure('DefaultAxesFontSize',12, 'DefaultLineLineWidth', 1)
    plot(z0_mat', min_dose_mat');
    hold on
    plot(z0_mat(end, :)', dose_target*ones(length(z0_mat)), 'k', 'LineStyle', '--');
    xlabel('z (mm)');
    ylabel('Min dose (Gy)');
   
    if Rin == 2.385
        img_name = "min_dose_vs_z_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_Rin2p385";
        title("LRn = " + LRn + " mm, LPb = " + LPb + " mm" + " Rin = 2.385 mm");
    else
        if Rin == 2
            img_name = "min_dose_vs_z_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_Rin2";
            title("LRn = " + LRn + " mm, LPb = " + LPb + " mm" + " Rin = 2 mm");
        else
            img_name = "min_dose_vs_z_LRn0" + 100*LRn + "_LPb0" + 100*LPb;
            title("LRn = " + LRn + " mm, LPb = " + LPb + " mm");
        end
    end
    
    saveas(gcf,imgs_folder + img_name + '.fig');
    print(gcf, imgs_folder + img_name, '-dpng', '-r600');
end

if f_plot_cold_spot_ratio == 1
    figure('DefaultAxesFontSize',12, 'DefaultLineLineWidth', 1)
    plot(d_cones_vec', cold_spot_ratio_vec);
%     hold on
%     plot(z0_mat(end, :)', dose_target*ones(length(z0_mat)), 'k', 'LineStyle', '--');
    xlabel('d (mm)');
    ylabel('Cold spot ratio');

    
    if Rin == 2.385
        img_name = "cold_spot_ratio_vs_d_cyl_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_Rin2p385";
        title("LRn = " + LRn + " mm, LPb = " + LPb + " mm, pad = " + pad + " mm" + " Rin = 2.385 mm");
    else
        if Rin == 2
            img_name = "cold_spot_ratio_vs_d_cyl_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_Rin2";
            title("LRn = " + LRn + " mm, LPb = " + LPb + " mm, pad = " + pad + " mm" + " Rin = 2 mm");
        else
            img_name = "cold_spot_ratio_vs_d_cyl_LRn0" + 100*LRn + "_LPb0" + 100*LPb;
            title("LRn = " + LRn + " mm, LPb = " + LPb + " mm, pad = " + pad + " mm");
        end
    end
    saveas(gcf,imgs_folder + img_name + '.fig');
    print(gcf, imgs_folder + img_name, '-dpng', '-r600');
end




%{
figure(1)
clf
hold on
axis equal
for n = 1:(Ncones*Nseeds_per_cone)
    x1 = seed_xyz(n,1);
    y1 = seed_xyz(n,2);
    z1 = seed_xyz(n,3);
    x2 = seed_xyz(n,4);
    y2 = seed_xyz(n,5);
    z2 = seed_xyz(n,6);   
    plot3([x1 x2],[y1 y2],[z1 z2],'-b')
end
%}





function [doseXY, xmat, ymat, xvec, yvec] = calc_dose_XY(W, dx, dy, zview, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels, f_plt_heatmap, f_plt_cont)

xmin = -W;
xmax = W;
xvec = xmin:dx:xmax;

ymin = -W;
ymax = W;
yvec = ymin:dy:ymax;

[xmat,ymat] = meshgrid(xvec,yvec);
zmat = zview*ones(size(xmat));

doseXY = zeros(size(xmat));

for n = 1:Nseeds

      % seed end coordinate
      x1 = seed_xyz(n,1);
      y1 = seed_xyz(n,2);
      z1 = seed_xyz(n,3);
      x2 = seed_xyz(n,4);
      y2 = seed_xyz(n,5);
      z2 = seed_xyz(n,6);

      % seed center coordinates 
      xm = (x1+x2)/2;
      ym = (y1+y2)/2;
      zm = (z1+z2)/2;

      % unit vector along seed axis
      n12x = (x2-x1)/Lseed;
      n12y = (y2-y1)/Lseed;
      n12z = (z2-z1)/Lseed;
      
      % unit vector from seed center to point of interest + radial distance from seed center
      r = sqrt( (xmat-xm).^2 + (ymat-ym).^2 + (zmat-zm).^2 );     
      nm0x = (xmat-xm)./(r+eps);
      nm0y = (ymat-ym)./(r+eps);
      nm0z = (zmat-zm)./(r+eps);  
        
      % angle between unit vectors
      cos_theta = n12x*nm0x + n12y*nm0y + n12z*nm0z; % scalar product
      theta = 180/pi * acos(cos_theta);

      i = find(r<rmax);
      doseXY(i) = doseXY(i) + interp2(R_LUT,THETA_LUT,dose_LUT,r(i),theta(i));
end

% if f_plt_heatmap
%     figure
%     clf
%     h = pcolor(xmat,ymat,doseXY);
%     set(h,'edgecolor','none')
%     set(gca,'ColorScale','log')
%     axis equal
%     colormap('jet')
%     colorbar
%     %caxis([0 50])
%     xlabel('X [mm]')
%     ylabel('Y [mm]')
%     axis equal
%     title("XT view, Z = " + num2str(zview) + "mm");
% end

% if f_plt_cont
%     figure
%     clf
%     [cs,h1] = contour(xmat,ymat,doseXY,dose_levels);
%     colormap('jet')
%     colorbar
%     caxis([0 50])
%     xlabel('X [mm]')
%     ylabel('Y [mm]')
%     clabel(cs,h1,'manual','fontsize',6)
% end
end

function [doseXZ, xmat, zmat, xvec, zvec, cold_spot_ratio] = calc_dose_XZ(W, dx, Wz, dz, y0, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels,...
    f_plt_heatmap, f_plt_cont, d_cones, f_plot_cold_spot_ratio, dose_target, pad)

xmin = -W;
xmax = W;
xvec = xmin:dx:xmax;

zmin = -Wz;
zmax = Wz;
zvec = zmin:dz:zmax;

[xmat,zmat] = meshgrid(xvec,zvec);
ymat = y0*ones(size(xmat));

doseXZ = zeros(size(xmat));

for n = 1:Nseeds

      % seed end coordinate
      x1 = seed_xyz(n,1);
      y1 = seed_xyz(n,2);
      z1 = seed_xyz(n,3);
      x2 = seed_xyz(n,4);
      y2 = seed_xyz(n,5);
      z2 = seed_xyz(n,6);

      % seed center coordinates 
      xm = (x1+x2)/2;
      ym = (y1+y2)/2;
      zm = (z1+z2)/2;

      % unit vector along seed axis
      n12x = (x2-x1)/Lseed;
      n12y = (y2-y1)/Lseed;
      n12z = (z2-z1)/Lseed;
      
      % unit vector from seed center to point of interest + radial distance from seed center
      r = sqrt( (xmat-xm).^2 + (ymat-ym).^2 + (zmat-zm).^2 );     
      nm0x = (xmat-xm)./(r+eps);
      nm0y = (ymat-ym)./(r+eps);
      nm0z = (zmat-zm)./(r+eps);  
        
      % angle between unit vectors
      cos_theta = n12x*nm0x + n12y*nm0y + n12z*nm0z; % scalar product
%       theta = 180/pi * acos(cos_theta);
      theta = 180/pi * real(acos(cos_theta));
        
      i = find(r<rmax);
%       disp(n);
      doseXZ(i) = doseXZ(i) + 0.5*interp2(R_LUT,THETA_LUT,dose_LUT,r(i),theta(i));
end

if f_plot_cold_spot_ratio == 1
    [cold_spot_ratio, z_min, z_max, x_max] = calc_cold_spot_ratio(doseXZ, xmat, zmat, dose_target, pad);
end

if f_plt_heatmap
    figure
    clf
    h = pcolor(xmat,zmat,doseXZ);
    set(h,'edgecolor','none')
    set(gca,'ColorScale','log')
    axis equal
    colormap('jet')
    colorbar
    %caxis([0 50])
    xlabel('X [mm]')
    ylabel('Z [mm]')
    axis equal
    title("XZ view, y = " + num2str(y0) + "mm");
end

if f_plt_cont && floor(d_cones)==d_cones
    figure
    clf
%     [cs,h1] = contour(xmat,zmat,doseXZ,dose_levels);
    contour(xmat,zmat,doseXZ,[dose_levels, dose_levels],'k');
    axis equal
    axis([-8 8 -20 10])
    colormap('jet')
%     colorbar
%     caxis([0 50])
    rectangle('Position',[-x_max, z_min, 2*x_max, z_max-z_min], 'LineStyle', '--');
    xlabel('X [mm]')
    ylabel('Z [mm]')
%     clabel(cs,h1,'manual','fontsize',6)
    title("XZ view, y = " + num2str(y0) + ", d = " + d_cones + " mm");
    
end
end

function [doseXYZ, xmat, zmat, xvec, zvec, cold_spot_ratio] = calc_dose_XYZ(W, dx, Wz, dz, y0, Nseeds, seed_xyz, Lseed, R_LUT, THETA_LUT, dose_LUT, rmax, dose_levels,...
    f_plt_heatmap, f_plt_cont, d_cones, f_plot_cold_spot_ratio, dose_target, pad, imgs_folder, LRn, LPb, text_string, Rin, alpha_deg, f_export_images)

xmin = -W;
xmax = W;
xvec = xmin:dx:xmax;

zmin = -Wz;
zmax = Wz;
zvec = zmin:dz:zmax;

ymin = -W;
ymax = W;
yvec = ymin:dx:ymax;

[xmat,ymat, zmat] = meshgrid(xvec,yvec, zvec);
% ymat = y0*ones(size(xmat));

doseXYZ = zeros(size(xmat));

for n = 1:Nseeds

      % seed end coordinate
      x1 = seed_xyz(n,1);
      y1 = seed_xyz(n,2);
      z1 = seed_xyz(n,3);
      x2 = seed_xyz(n,4);
      y2 = seed_xyz(n,5);
      z2 = seed_xyz(n,6);

      % seed center coordinates 
      xm = (x1+x2)/2;
      ym = (y1+y2)/2;
      zm = (z1+z2)/2;

      % unit vector along seed axis
      n12x = (x2-x1)/Lseed;
      n12y = (y2-y1)/Lseed;
      n12z = (z2-z1)/Lseed;
      
      % unit vector from seed center to point of interest + radial distance from seed center
      r = sqrt( (xmat-xm).^2 + (ymat-ym).^2 + (zmat-zm).^2 );     
      nm0x = (xmat-xm)./(r+eps);
      nm0y = (ymat-ym)./(r+eps);
      nm0z = (zmat-zm)./(r+eps);  
        
      % angle between unit vectors
      cos_theta = n12x*nm0x + n12y*nm0y + n12z*nm0z; % scalar product
%       theta = 180/pi * acos(cos_theta);
      theta = 180/pi * real(acos(cos_theta));
        
      i = find(r<rmax);
%       disp(n);
      doseXYZ(i) = doseXYZ(i) + 0.5*interp2(R_LUT,THETA_LUT,dose_LUT,r(i),theta(i));
      
      %%% For visualizing the sources in 3D. It is advised to do this in
      %%% debug mode and only plot a few of these as they are heavy.
%       figure
%       x_tmp = xmat(doseXYZ > 200);
%       y_tmp = ymat(doseXYZ > 200);
%       z_tmp = zmat(doseXYZ > 200);
%       dose_tmp = doseXYZ(doseXYZ > 200);
%       scatter3(x_tmp,y_tmp,z_tmp,5*ones(numel(x_tmp), 1),dose_tmp);
end

if f_plot_cold_spot_ratio == 1
    [cold_spot_ratio, z_min, z_max, x_max, ~] = calc_cold_spot_ratio(doseXYZ, xmat, ymat, zmat, dose_target, pad);
end

if f_plt_heatmap
    figure
    clf
    h = pcolor(xmat,zmat,doseXYZ);
    set(h,'edgecolor','none')
    set(gca,'ColorScale','log')
    axis equal
    colormap('jet')
    colorbar
    %caxis([0 50])
    xlabel('X [mm]')
    ylabel('Z [mm]')
    axis equal
    title("XZ view, y = " + num2str(y0) + "mm");
end

if f_plt_cont && floor(d_cones)==d_cones
%     idx = ymat == y0;
    [dimx, ~, ~] = size(doseXYZ);
%     x_slab = reshape(xmat(idx), [dimz, dimx]);
%     z_slab = reshape(zmat(idx), [dimz, dimx]);
%     doseXZ = reshape(doseXYZ(idx), size(x_slab));
    
    idx = round(dimx/2);
    x_slab = squeeze(xmat(idx, :, :));
    z_slab = squeeze(zmat(idx, :, :));
    doseXZ = squeeze(doseXYZ(idx, :, :));
    figure
    clf
%     [cs,h1] = contour(xmat,zmat,doseXZ,dose_levels);
    contour(x_slab,z_slab,doseXZ,[dose_levels, dose_levels],'k');
    axis equal
    axis([-8 8 -20 10])
    colormap('jet')
%     colorbar
%     caxis([0 50])
    rectangle('Position',[-x_max, z_min, 2*x_max, z_max-z_min], 'LineStyle', '--');
    xlabel('X [mm]')
    ylabel('Z [mm]')
%     clabel(cs,h1,'manual','fontsize',6)
    
    if Rin == 2.385
        img_name = "contour_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_d" + d_cones + "_ang" + alpha_deg + "_Rin2p385";
        title("XZ view, y = " + num2str(y0) + ", d = " + d_cones + " mm, " + "\alpha = " + alpha_deg + "\circ, Rin = 2.385 mm");
    else
        if Rin == 2
            img_name = "contour_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_d" + d_cones + "_ang" + alpha_deg + "_Rin2";
            title("XZ view, y = " + num2str(y0) + ", d = " + d_cones + " mm, " + "\alpha = " + alpha_deg + "\circ, Rin = 2 mm");
        else
            img_name = "contour_LRn0" + 100*LRn + "_LPb0" + 100*LPb + "_d" + d_cones + "_ang" + alpha_deg;
            title("XZ view, y = " + num2str(y0) + ", d = " + d_cones + " mm, " + "\alpha = " + alpha_deg + "\circ");
        end
    end
    
    text(-7, -16, text_string, 'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    if f_export_images == 1
        saveas(gcf,imgs_folder + img_name + '.fig');
        print(gcf, imgs_folder + img_name, '-dpng', '-r600');
    end
end
end

function [min_dose] = find_min_dose_XY(doseXY, xmat, ymat, ra_min, ra_max)
R = sqrt(xmat.^2 + ymat.^2);
% min_dose = min(doseXY(R >= ra_min & R <= ra_max));
min_dose = doseXY(round(length(xmat)/2), round(length(xmat)/2));
end

function [cold_spot_ratio, z_min_cyl, z_max_cyl, x_max_cyl, y_max_cyl] = calc_cold_spot_ratio(doseXYZ, xmat, ymat, zmat, dose_target, pad)
x_max = max(xmat(doseXYZ >= dose_target));
y_max = max(ymat(doseXYZ >= dose_target));
z_max = max(zmat(doseXYZ >= dose_target));
z_min = min(zmat(doseXYZ >= dose_target));

x_max_cyl = x_max + pad;
y_max_cyl = y_max + pad;
r_max_cyl = x_max_cyl;
z_max_cyl = z_max + pad;
z_min_cyl = z_min - pad;

cyl_inds = sqrt(xmat.^2 + ymat.^2) <= r_max_cyl & zmat >= z_min_cyl & zmat <= z_max_cyl;

% disp("Cold spot sum: " + sum(sum(doseXZ(cyl_inds) <= dose_target)));
% disp("Cyl inds sum: " + sum(cyl_inds(:)));
cold_spot_ratio = sum(sum(sum(doseXYZ(cyl_inds) <= dose_target))) / sum(cyl_inds(:));

end