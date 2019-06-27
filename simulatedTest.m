% 3-mode simulated dataset testing

addpath('tensor_toolbox/');
N = 100;
R = 3;
p = [10 10 10];
threshold = 0.5;

X = tenrand([N p]);
W = tenrand(p);
Y = ttt(X,W,2:4,1:3);
err = 0.1*tenrand([N 1]);
err = tensor(err.data, N);
Y = Y + err;

lambda = 0.1;
epsilon = 1;

tic
[Err, estimated_W] = FASHOR_l1(X, Y, R, lambda, epsilon, 10, 1e-5);
t = toc;
disp('Elapsed time :')
t

% MSE
r1 = vec2Tensor(estimated_W(1,:),p);
r2 = vec2Tensor(estimated_W(2,:),p);
r3 = vec2Tensor(estimated_W(3,:),p);
estimated_W = r1+r2+r3;
error = W-estimated_W;
norm(error) / prod(p)
