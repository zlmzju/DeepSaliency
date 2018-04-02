% Demo for paper "Saliency Detection via Graph-Based Manifold Ranking" 
% by Chuan Yang, Lihe Zhang, Huchuan Lu, Ming-Hsuan Yang, and Xiang Ruan
% To appear in Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2013), Portland, June, 2013.
clear all;
addpath('./utils/');
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight 
alpha=0.99;% control the balance of two items in manifold ranking cost function
spnumber=200;% superpixel number
imgRoot='../../Dataset/MSRA1000/Images/';% test image path
saldir='../../Result/MSRA1000/SaliencyMap/boundarymap/';% the output path of the saliency map
%initsaldir='../../Result/DUT-OMRON/DeepMap/V1/MAT/';% the input path of the deep saliency map
supdir='../../Result/MSRA1000/SuperPixels/';% the superpixel label file path
gtdir = '../../Dataset/MSRA1000/Groundtruth/';
mkdir(supdir);
mkdir(saldir);
imnames=dir([imgRoot '*' 'jpg']);
supnames = dir([supdir '*' 'dat']);
%initmapnames = dir([initsaldir '*' 'mat']);
for ii=1:length(imnames)
    disp(ii);
    imname=[imgRoot imnames(ii).name]; 
    supname=[supdir supnames(ii).name];
    %initmapname=[initsaldir initmapnames(ii).name];
    [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame 
    [m,n,k] = size(input_im);

%%----------------------read superpixels--------------------%%
    superpixels=ReadDAT([m,n],supname); % superpixel label matrix
    spnum=max(superpixels(:));% the actual superpixel number
   
%%----------------------for each superpixel--------------------%%    
    % for each node (superpixels)
    input_vals=reshape(input_im, m*n, k);
    rgb_vals=zeros(spnum,1,3);
    inds=cell(spnum,1);
    for i=1:spnum 
        inds{i}=find(superpixels==i);
        rgb_vals(i,1,:)=mean(input_vals(inds{i},:),1);
    end
    lab_vals = colorspace('Lab<-', rgb_vals); 
    seg_vals=reshape(lab_vals,spnum,3);% feature for each superpixel
 % get edges
    adjloop=AdjcProcloop(superpixels,spnum);
    edges=[];
    for i=1:spnum
        indext=[];
        ind=find(adjloop(i,:)==1);
        %neighbor's neighbor
        for j=1:length(ind)
            indj=find(adjloop(ind(j),:)==1);
            indext=[indext,indj];
        end
        indext=[indext,ind];
        indext=indext((indext>i));
        indext=unique(indext);
        if(~isempty(indext))
            ed=ones(length(indext),2);
            ed(:,2)=i*ed(:,2);
            ed(:,1)=indext;
            edges=[edges;ed];
        end
    end

% compute affinity matrix
    weights = makeweights(edges,seg_vals,theta);
    W = adjacency(edges,weights,spnum);   
    %W = exp(W*5)-1;
%% main algorithm
    y=zeros(spnum,1);
%background
  
    background=[unique(superpixels(1,1:n))';unique(superpixels(1:m,1));...
        unique(superpixels(1:m,n));unique(superpixels(m,1:n))'];
    y(background)=-1;
    
    labeled=(y~=0);
    P=pdist2(seg_vals,seg_vals);
    delta=1;
    K = exp(-P./(2*delta*delta));
    alpha=laprls(K,W,y);
    out=K*alpha;
    
%--------------show result------------------------
    tmapstage3=zeros(m,n);
    for i=1:spnum
        tmapstage3(inds{i})=out(i);    
    end
    tmapstage3=(tmapstage3-min(tmapstage3(:)))/(max(tmapstage3(:))-min(tmapstage3(:)));

    mapstage3=zeros(w(1),w(2));
    mapstage3(w(3):w(4),w(5):w(6))=tmapstage3;
    mapstage3=uint8(mapstage3*255);
    outname=[saldir imnames(ii).name(1:end-4)  '.png'];
    imwrite(mapstage3,outname);
    %figure;
    imshow(mapstage3);
    drawnow;
end
