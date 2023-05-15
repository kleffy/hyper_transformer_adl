% Load the mat file
load('/vol/research/RobotFarming/Projects/hyper_transformer/Experiments/HSIT/pavia_dataset/FT_E10_M12/final_prediction.mat');
whos
% Identify the variable name containing the hyperspectral image data
var_name = 'pavia_23_pred';

% Access the hyperspectral image data using the identified variable name
spectral_data = eval(var_name);

% Extract the spectral signature of a pixel at row 10, column 15
pixel_spectrum = squeeze(spectral_data(10,15,:));

% Plot the spectral signature
plot(pixel_spectrum);
