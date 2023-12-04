  load('C:\Users\Mohammed\Downloads\BM3D-master1\BM3D-master\neuralweights_sig50.mat');
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

%s=dir('E:\Dataset_BUSI\Dataset_BUSI_with_GT\*\*).png');
s=dir("C:\Users\Mohammed\OneDrive - Georgia Institute of Technology\Set12_noisy\Set12_noisy\*.png");
%%
for i = 1:numel(s)

     path=fullfile(s(i).folder,s(i).name);
       im_noisy=(imread(path));
     b1=fNeural(im_noisy, 50, model,w);
     imwrite(mat2gray(b1),fullfile(string(s(i).folder).replace("Set12_noisy\Set12_noisy","Set12_noisy\Set12_d"),s(i).name));

end