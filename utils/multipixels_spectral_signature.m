% Load the hyperspectral image data
load('/path/to/hyperspectral/image.mat', 'pavia_23_pred');

% Extract the spectral signature of 3 different pixels
pixel_spectrum_1 = squeeze(pavia_23_pred(10,15,:));
pixel_spectrum_2 = squeeze(pavia_23_pred(20,30,:));
pixel_spectrum_3 = squeeze(pavia_23_pred(50,70,:));

% Plot the spectral signatures of the 3 pixels on the same graph
plot(pixel_spectrum_1, 'b', 'LineWidth', 2);
hold on;
plot(pixel_spectrum_2, 'g', 'LineWidth', 2);
plot(pixel_spectrum_3, 'r', 'LineWidth', 2);
legend('Pixel 1', 'Pixel 2', 'Pixel 3');
xlabel('Spectral band');
ylabel('Reflectance/Radiance');
title('Spectral Signatures of 3 Pixels');