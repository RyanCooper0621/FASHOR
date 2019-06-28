% 3-mode simulated dataset testing
% TODO: parameters need to be selected through cv
clc,clear;
addpath('tensor_toolbox/');
% global setups
N = 100;
p = [40 40 40];
M = 3;
R = 1;
repeat = 10;
totalTime = [0 0 0];
W_MSE = [0 0 0];

%================ FASHOR L1 ==========================
% parameters setup
fprintf('===== FASHOR L1 =====\n')
lambda = 0.05;
epsilon = 1;
iter = 1000;
% time cost
for it = 1:repeat
    % generate dataset
    threshold = 0.5;
    X = tenrand([N p]);
    W = tenrand(p);
    Y = ttt(X,W,2:4,1:3);
    err = 0.1*tenrand([N 1]);
    err = tensor(err.data, N);
    Y = Y + err;
    tic
    [Err, estimated_W] = FASHOR_l1(X, Y, R, lambda, epsilon, round(iter/(R*length(p))), 1e-4);
    t = toc;
    totalTime(1) = totalTime(1) + t;
end
%fprintf('Elapsed time is %.4f sec\n',totalTime/repeat)
% MSE
% TODO: new function to compute MSE
r1 = vec2Tensor(estimated_W(1,:),p);
%r2 = vec2Tensor(estimated_W(2,:),p);
%r3 = vec2Tensor(estimated_W(3,:),p);
%estimated_W = r1+r2+r3;
estimated_W = r1;
error = W-estimated_W;
%fprintf('MSE is %.6f\n',norm(error) / prod(p))
W_MSE(1) = W_MSE(1) + norm(tensor(error)) / norm(tensor(W));



%================ Remurs ==========================
% parameters setup
addpath('RemursCode/Code/')
fprintf('===== Remurs =====\n')
alpha=1e-3;
beta=1e-3;
epsilon=1e-4;
iter=1000;
% time cost
for it = 1:repeat
    % generate dataset
    X = tenrand([p N]);
    W = tenrand(p);
    Y = ttt(X,W,1:3,1:3);
    err = 0.1*tenrand([N 1]);
    err = tensor(err.data, N);
    Y = Y + err;
    Y_mean = mean(Y.data);
    for i =1:N
        if Y(i)>Y_mean
            Y(i)=1;
        else
            Y(i)=-1;
        end
    end
    Y=reshape(Y.data,[N 1]);
    X=reshape(X.data,[p N]);
    tic
    [tW, errList]=Remurs(X, Y, alpha, beta, epsilon, iter);
    t = toc;
    totalTime(2) = totalTime(2) + t;
end
%fprintf('Elapsed time is %.4f sec\n',totalTime/repeat)
error = reshape(W.data,p)-tW;
%fprintf('MSE is %.6f\n',norm(tensor(error)) / prod(p))
W_MSE(2) = W_MSE(2) + norm(tensor(error)) / norm(tensor(W));


%================ SURF ==========================
% parameters setup
addpath('SURF_code/')
fprintf('===== SURF =====\n')
epsilon = 0.1;
xi = epsilon*0.005;
alpha = 1;
estimated_W = zeros(R,prod(p));
% time cost
for it = 1:repeat
    % generate dataset
    Xt = zeros([N prod(p)]);
    X = tensor(zeros([p N]));
    for i = 1:N
        Xt(i,:)=rand([1 prod(p)]);
        X(:,:,:,i) = tensor(Xt(i,:),p);
    end
    W = tenrand(p);
    Wvec = zeros(1,prod(p));
    for i = 1:p(1)
        for j = 1:p(2)
            for k =1:p(3)
                index = (i-1)*p(1)+(j-1)*p(2)+k;
                Wvec(1,index) = W(i,j,k);
            end
        end
    end
    Y = ttt(X,W,1:3,1:3);
    err = 0.1*tenrand([N 1]);
    err = tensor(err.data, N);
    Y = Y + err;
    X = X.data;
    Y = Y.data;
    tic
    for r =1:R
        [W_r, residual] = MyTrain(X, Xt,Y, epsilon, xi, alpha, 1e-4);
        Y = residual;
        estimated_W(r,:) = W_r;
    end
    t = toc;
    totalTime(3) = totalTime(3) + t;
end
%fprintf('Elapsed time is %.4f sec\n',totalTime/repeat)
% TODO: new function to compute MSE
estimated_Wvec = zeros(1,prod(p)); % for compute MSE
for r = 1:R
    estimated_Wvec = estimated_Wvec + estimated_W(r,:);
end
error = Wvec-estimated_Wvec;
%fprintf('MSE is %.6f\n',norm(error) / prod(p))
W_MSE(3) = W_MSE(3) + norm(tensor(error)) / norm(tensor(W));

totalTime = totalTime/repeat;
W_MSE = W_MSE/repeat;

