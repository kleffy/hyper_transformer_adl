% Load the hyperspectral and RGB images separately
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_N50_c102_resnext101_4/final_prediction.mat', 'pavia_23_pred');
% Extract the spectral signature of two consecutive pixels from each image
pixel_spectrum_hyperspectral = squeeze(pavia_23_pred(35:36,66,:));

load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_E10_M12/final_prediction.mat', 'pavia_23_pred');
pixel_spectrum_RGB = squeeze(pavia_23_pred(35:36,66,:));

% Plot the spectral signatures of the two images on the same graph
plot(pixel_spectrum_hyperspectral', 'b', 'LineWidth', 2);
hold on;
plot(pixel_spectrum_RGB', 'r', 'LineWidth', 2);
legend('Hyperspectral native CNN', 'RGB native CNN');
xlabel('Spectral band');
ylabel('Reflectance/Radiance');
