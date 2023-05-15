% Load the hyperspectral and RGB images separately
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_N50_c102_resnext101_4/final_prediction.mat');

% Extract the spectral signature of a pixel or region of interest from each image
pixel_spectrum_hyperspectral = squeeze(pavia_23_pred(10,45,:));

load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_E10_M12/final_prediction.mat', 'pavia_23_pred');
pixel_spectrum_RGB = squeeze(pavia_23_pred(10,45,:));

% Plot the spectral signatures of the two images on the same graph
plot(pixel_spectrum_hyperspectral([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 'b', 'LineWidth', 2);
hold on;
plot(pixel_spectrum_RGB([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 'r', 'LineWidth', 2);
legend('Hyperspectral native CNN', 'RGB native CNN');
xlabel('Spectral band');
ylabel('Reflectance/Radiance');
title('Spectral signature for bands 10, 23, and 43');
