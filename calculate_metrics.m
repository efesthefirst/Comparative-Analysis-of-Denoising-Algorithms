function metrics = calculate_metrics(noisy,noisy_dir,ref_dir,denoised_dir,output_file)
    display(noisy);
    ref = strcat(noisy(1:end-8),'10_0.jpg');
    denoised = strcat(noisy(end-17:end-4),'_d.jpg');
    img_noisy = imread(strcat(noisy_dir,noisy));
    img_ref = imread(strcat(ref_dir,ref));
    img_denoised = imread(strcat(denoised_dir,denoised));

    psnr_noisy = psnr(img_noisy,img_ref);
    psnr_denoised = psnr(img_denoised,img_ref);

    ssim_noisy = ssim(img_noisy,img_ref);
    ssim_denoised = ssim(img_denoised,img_ref);
    
    cwssim_noisy = cwssim_index(img_noisy, img_ref,6,16,0,0);
    cwssim_denoised = cwssim_index(img_denoised, img_ref,6,16,0,0);

    cat_img_ref = cat(3,img_ref,img_ref,img_ref);
    cat_img_noisy = cat(3,img_noisy,img_noisy,img_noisy);
    cat_img_denoised = cat(3,img_denoised,img_denoised,img_denoised);

    unique_noisy = mslUNIQUE(cat_img_ref,cat_img_noisy);
    unique_denoised = mslUNIQUE(cat_img_ref,cat_img_denoised);

    ms_unique_noisy = mslMSUNIQUE(cat_img_ref,cat_img_noisy);
    ms_unique_denoised = mslMSUNIQUE(cat_img_ref,cat_img_denoised);

    csv_noisy = csv(cat_img_ref,cat_img_noisy);
    csv_denoised = csv(cat_img_ref,cat_img_denoised);

    summer_noisy = SUMMER(cat_img_ref,cat_img_noisy);
    summer_denoised = SUMMER(cat_img_ref,cat_img_denoised);
   
    metrics = [psnr_noisy,psnr_denoised,ssim_noisy,ssim_denoised,cwssim_noisy,cwssim_denoised,unique_noisy,unique_denoised,ms_unique_noisy,ms_unique_denoised,csv_noisy,csv_denoised,summer_noisy,summer_denoised];
    % Check if the output file exists; if not, write column names
    if ~exist(output_file, 'file')
        column_names = {'image', 'psnr_noisy', 'psnr_denoised', 'ssim_noisy', 'ssim_denoised', ...
            'cwssim_noisy', 'cwssim_denoised', 'unique_noisy', 'unique_denoised', 'ms_unique_noisy', ...
            'ms_unique_denoised', 'csv_noisy', 'csv_denoised', 'summer_noisy', 'summer_denoised'};
        fid = fopen(output_file, 'w');
        fprintf(fid, '%s ', column_names{:});
        fprintf(fid, '\n');
        fclose(fid);
    end

    % Save metrics to a text file
    fid = fopen(output_file, 'a');  % Open file in append mode
    fprintf(fid, '%s ', noisy);  % Write the noisy input name
    fprintf(fid, '%f ', metrics);  % Write the metrics
    fprintf(fid, '\n');  % Start a new line for the next set of metrics
    fclose(fid);  % Close the file
end


