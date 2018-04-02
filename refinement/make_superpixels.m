clear all;
addpath('./utils/');
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight 
alpha=0.99;% control the balance of two items in manifold ranking cost function
spnumber=200;% superpixel number
imgRoot='../../Dataset/THUR/Images/';% test image path
saldir='../../Result/THUR/SaliencyMap/DeepMap/';% the output path of the saliency map
initsaldir='../../Result/THUR/DeepMap/V1/MAT/';% the input path of the deep saliency map
supdir='../../Result/THUR/SuperPixels/';% the superpixel label file path
gtdir = '../../Dataset/THUR/Groundtruth/';
mkdir(supdir);
mkdir(saldir);
imnames=dir([imgRoot '*' 'jpg']);
for ii=1:length(imnames)
    disp(ii);
    imname=[imgRoot imnames(ii).name]; 
    [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame 
    [m,n,k] = size(input_im);

%%----------------------generate superpixels--------------------%%
    imname=[imname(1:end-4) '.bmp'];% the slic software support only the '.bmp' image
    comm=['SLICSuperpixelSegmentation' ' ' imname ' ' int2str(20) ' ' int2str(spnumber) ' ' supdir];
    system(comm);    
    spname=[supdir imnames(ii).name(1:end-4)  '.dat'];
    superpixels=ReadDAT([m,n],spname); % superpixel label matrix
    spnum=max(superpixels(:));% the actual superpixel number
end
    