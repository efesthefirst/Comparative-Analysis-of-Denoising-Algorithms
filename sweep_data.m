clear all; clc;
addpath(genpath('UNIQUE-Unsupervised-Image-Quality-Estimation-master'));
addpath(genpath('SUMMER-master'));
addpath(genpath('MS-UNIQUE-master'));
addpath(genpath('CSV-master'));


noisy_dir1 = '/home/olives/Desktop/ECE6258_Project/Datasets/12_grayscale_underexposure/';
noisy_dir2 = '/home/olives/Desktop/ECE6258_Project/Datasets/13_grayscale_overexposure/';
noisy_dir3 = '/home/olives/Desktop/ECE6258_Project/Datasets/14_grayscale_blur/';
noisy_dir4 = '/home/olives/Desktop/ECE6258_Project/Datasets/15_grayscale_contrast/';
noisy_dir5 = '/home/olives/Desktop/ECE6258_Project/Datasets/16_grayscale_dirtylens1/';
noisy_dir6 = '/home/olives/Desktop/ECE6258_Project/Datasets/17_grayscale_dirtylens2/';
noisy_dir7 = '/home/olives/Desktop/ECE6258_Project/Datasets/18_grayscale_saltpepper/';

denoised_dir1 = '/home/olives/Desktop/ECE6258_Project_Code/12_grayscale_underexposure_denoised_images/';
denoised_dir2 = '/home/olives/Desktop/ECE6258_Project_Code/13_grayscale_overexposure_denoised_images/';
denoised_dir3 = '/home/olives/Desktop/ECE6258_Project_Code/14_grayscale_blur_denoised_images/';
denoised_dir4 = '/home/olives/Desktop/ECE6258_Project_Code/15_grayscale_contrast_denoised_images/';
denoised_dir5 = '/home/olives/Desktop/ECE6258_Project_Code/16_grayscale_dirtylens1_denoised_images/';
denoised_dir6 = '/home/olives/Desktop/ECE6258_Project_Code/17_grayscale_dirtylens2_denoised_images/';
denoised_dir7 = '/home/olives/Desktop/ECE6258_Project_Code/18_grayscale_saltpepper_denoised_images/';

noisy_dirs = {noisy_dir1,noisy_dir2,noisy_dir3,noisy_dir4,noisy_dir5,noisy_dir6,noisy_dir7};
denoised_dirs = {denoised_dir1,denoised_dir2,denoised_dir3,denoised_dir4,denoised_dir5,denoised_dir6,denoised_dir7};
ref_dir = '/home/olives/Desktop/ECE6258_Project/Datasets/10_grayscale_no_challenge/';
samples_dir = '/home/olives/Desktop/ECE6258_Project/min_samples_ubuntu/';
file_lst = dir(samples_dir);
output_file = '/home/olives/Desktop/ECE6258_Project/cure_or_metrics_deneme.txt';

for i=4:length(file_lst)
    noisy_dir = noisy_dirs{i-2};
    denoised_dir = denoised_dirs{i-2};
    file = file_lst(i);
    images = readlines(strcat(samples_dir,file.name),"EmptyLineRule","skip");
    for j=1:length(images)
        image = images(j);
        image = convertStringsToChars(image);
        metrics1 = calculate_metrics(image,strcat(noisy_dir,'Level_1/'),ref_dir,denoised_dir,output_file);
        metrics2 = calculate_metrics(strcat(image(1:end-5),'2.jpg'),strcat(noisy_dir,'Level_2/'),ref_dir,denoised_dir,output_file);
        metrics3 = calculate_metrics(strcat(image(1:end-5),'3.jpg'),strcat(noisy_dir,'Level_3/'),ref_dir,denoised_dir,output_file);
        metrics4 = calculate_metrics(strcat(image(1:end-5),'4.jpg'),strcat(noisy_dir,'Level_4/'),ref_dir,denoised_dir,output_file);
        metrics5 = calculate_metrics(strcat(image(1:end-5),'5.jpg'),strcat(noisy_dir,'Level_5/'),ref_dir,denoised_dir,output_file);
    end
end

