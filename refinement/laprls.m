function alpha=laprls(K,W,y,flag)
% % generating a random dataset
% flag=1 is semi-supervised learning, flag=0 is regression
% y=Y;
% y(30:50)=0;
% P=pdist2(X,X);
% delta=1;
% K = exp(-P./(2*delta*delta));
%LapRLS algorithm
gamma_A=1e-6;
gamma_I=1;

D = sum(W,2);
% D_half=diag(sqrt(1./D));
% S=eye(size(K))-D_half*K*D_half;
S=diag(D)-W;
n=length(y);
labeled= true(n,1);
if flag
    labeled=(y~=0);
end
l=double(nnz(labeled));
if ~flag
    assert(l==n);
end
JK=zeros(size(K));
JK(labeled,:)=K(labeled,:);
alpha=(gamma_A*l*eye(n)+JK+(gamma_I*l/(n*n))*S*K)\y;

% out=sign(K*alpha);
% er=100*nnz(out~=Y)/length(y);
% fprintf('Error rate=%.1f\n\n',er);
end