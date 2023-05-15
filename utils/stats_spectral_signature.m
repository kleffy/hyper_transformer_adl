% Load the hyperspectral and RGB images separately
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_N50_c102_resnext101_4/final_prediction.mat', 'pavia_23_pred');

% Extract a random ROI of 25x25 pixels in both images
[row, col, bands] = size(pavia_23_pred);
rand_row = randi([1, row-24], 1);
rand_col = randi([1, col-24], 1);
roi_hyperspectral = pavia_23_pred(rand_row:rand_row+24, rand_col:rand_col+24, :);
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_E10_M12/final_prediction.mat', 'pavia_23_pred');
roi_RGB = pavia_23_pred(rand_row:rand_row+24, rand_col:rand_col+24, :);

% Reshape the ROI to a 2D matrix (pixels x bands) for analysis
spectral_data_hyperspectral = reshape(roi_hyperspectral, [], bands);
spectral_data_RGB = reshape(roi_RGB, [], bands);

% Compare the distribution of spectral values using a boxplot
figure;
boxplot(spectral_data_hyperspectral([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 'colors', 'b');
hold on;
boxplot(spectral_data_RGB([5, 10, 15, 20, 25, 30, 35, 40, 45, 50]), 'colors', 'r');
legend('Hyperspectral native CNN', 'RGB native CNN');
xlabel('Spectral band');
ylabel('Reflectance/Radiance');