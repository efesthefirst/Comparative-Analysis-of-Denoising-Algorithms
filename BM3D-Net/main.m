  load('\neuralweights_sig50.mat');
  w = {};
  w{1} = single(w_1);
  w{2} = single(w_2);
  w{3} = single(w_3);
  w{4} = single(w_4);
  w{5} = single(w_5);
  clear w_1; clear w_2; clear w_3; clear w_4; clear w_5;
  model = {};
% width of the Gaussian window for weighting output pixels
model.weightsSig = 2;
% the denoising stride. Smaller is better, but is computationally 
% more expensive.
model.step = 4;
%% Add your images
s=dir(\*.png");
%%
%%where to write your denoised images
folder=
for i = 1:numel(s)

     path=fullfile(s(i).folder,s(i).name);
       im_noisy=(imread(path));
     b1=fNeural(im_noisy, 50, model,w);
     imwrite(mat2gray(b1),fullfile(folder,s(i).name));

end
