function mapstage_o = MapReranking(K,W,map,superpixels,inds, w, m, n,spnum)
DeepPriorSpmap=zeros(spnum,1);
for i=1:spnum
    tmp = map(inds{i});
    DeepPriorSpmap(i) =  mean(tmp);
end
DeepPriorSpmap=(DeepPriorSpmap-min(DeepPriorSpmap(:)))/(max(DeepPriorSpmap(:))-min(DeepPriorSpmap(:)));

y=zeros(spnum,1);
background=[unique(superpixels(1,1:n))';unique(superpixels(1:m,1));...
    unique(superpixels(1:m,n));unique(superpixels(m,1:n))'];
% BoundarySpnum = length(background);
y(background)= -1;


alpha=laprls(K,W,y,1);
out=K*alpha;
out=(out-min(out(:)))/(max(out(:))-min(out(:)));

beta = 0.2;
y = (out.^beta) .* (DeepPriorSpmap.^(1-beta));
%y = out .* DeepPriorSpmap;
%y = DeepPriorSpmap;
y=(y-min(y(:)))/(max(y(:))-min(y(:)));
% Py = y;
y = 2*y - 1;

alpha=laprls(K,W,y,0);
out=K*alpha;


% assign the saliency value to each pixel
tmapstage_o=zeros(m,n);
for i=1:spnum
    tmapstage_o(inds{i})=out(i);
end
tmapstage_o=(tmapstage_o-min(tmapstage_o(:)))/(max(tmapstage_o(:))-min(tmapstage_o(:)));

mapstage_o=zeros(w(1),w(2));
mapstage_o(w(3):w(4),w(5):w(6))=tmapstage_o;
%mapstage_o=uint8(mapstage_o*255);