clear;close all;
addpath('./utils/');
% t1 = cputime;
% ***
% t2 = cputime;
% exetime = t2 - t1;
%%------------------------set parameters---------------------%%
theta=10; % control the edge weight
alpha=0.99;% control the balance of two items in manifold ranking cost function
spnumber=200;% superpixel number
datasets={'THUR'};%'DUT-OMRON','ECSSD','MSRA1000','PASCAL-S','SED2','SOD','THUR','THUS'}; %8 dataset
ROOTDIR='Z:/project/Saliency/ICCV_EXP/';
imageName='Butterfly150.jpg';
%ROOTDIR='../../';
dataset_num=size(datasets,2);
for idx_data=1:dataset_num
    DATASET=datasets{idx_data};
    imgRoot=[ROOTDIR,'Dataset/',DATASET,'/Images/'];% test image path
    saldir=[ROOTDIR,'Result/',DATASET,'/SaliencyMap/Ours/'];% the output path of the saliency map
    initsaldir=[ROOTDIR,'Result/',DATASET,'/DeepMap/V7/MAT/'];% the input path of the deep saliency map
    supdir=[ROOTDIR,'Result/',DATASET,'/SuperPixels/'];% the superpixel label file path
    gtdir = [ROOTDIR,'Dataset/',DATASET,'/Groundtruth/'];
%     mkdir(supdir);
    mkdir(saldir);

    fprintf(['\nStart ',DATASET,' ...\n']);
    tic;
    for ii=1:1
        imname=[imgRoot imageName];
        fprintf([imname,'\n']);
        supname=[supdir imageName(1:end-4) '.dat'];
        initmapname=[initsaldir imageName(1:end-4) '.mat'];
%         gtname = [gtdir imageName(1:end-4) '.bmp'];
        [input_im,w]=removeframe(imname);% run a pre-processing to remove the image frame
        [m,n,k] = size(input_im);
        %%----------------------read superpixels--------------------%%
        superpixels=ReadDAT([m,n],supname); % superpixel label matrix
        spnum=max(superpixels(:));% the actual superpixel number


        %%----------------------read deep map--------------------%%
        loadmap=load(initmapname);
        map = loadmap.deepMap;
        map = imresize(map,[m n]);
        map=(map-min(map(:)))/(max(map(:))-min(map(:)));


        %%----------------------design the graph model--------------------------%%
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

        P=pdist2(seg_vals,seg_vals);
        norm_p=normalize(P);
        K = exp(-theta*norm_p);

        mapstage_all = MapReranking(K,W,map,superpixels,inds, w, m, n,spnum);            

        mapstage_o=(mapstage_all);
        mapstage_o=(mapstage_o-min(mapstage_o(:)))/(max(mapstage_o(:))-min(mapstage_o(:)));
        mapstage_o=uint8(mapstage_o*255);
        toc;
        imshow(mapstage_o);
        fprintf('.');
        if mod(ii,100)==99
            fprintf('\n');
        end
    end
    fprintf(['\n',DATASET,' ok!\n']);
end