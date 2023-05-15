% Load the mat file
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/enmap_dataset/FT_N50_c224_resnext3D_7/final_prediction.mat');
whos
% Identify the variable name containing the hyperspectral image data
%ENMAP01-____L1C-DT0000003373_20220913T013912Z_008_V010111_20230
%pavia_23_pred
var_name = 'ENMAP01-____L1C-DT0000003373_20220913T013912Z_008_V010111_20230';

% Access the hyperspectral image data using the identified variable name
spectral_data = eval(var_name);

% Extract the spectral signature of a pixel at row 10, column 15
pixel_spectrum = squeeze(spectral_data(10,15,:));

% Plot the spectral signature
plot(pixel_spectrum);
