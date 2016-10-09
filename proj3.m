clear all;

%% load data

filename = 'train-images.idx3-ubyte';
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

train_images = fread(fp, inf, 'unsigned char');
train_images = reshape(train_images, numCols, numRows, numImages);
% images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
train_images = reshape(train_images, size(train_images, 1) * size(train_images, 2), size(train_images, 3));
% Convert to double and rescale to [0,1]
train_images = double(train_images) / 255;



filename = 'train-labels.idx1-ubyte';

%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

train_labels = fread(fp, inf, 'unsigned char');

assert(size(train_labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);


filename = 't10k-images.idx3-ubyte';
%loadMNISTImages returns a 28x28x[number of MNIST test_images] matrix containing
%the raw MNIST test_images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

test_images = fread(fp, inf, 'unsigned char');
test_images = reshape(test_images, numCols, numRows, numImages);
% test_images = permute(test_images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
test_images = reshape(test_images, size(test_images, 1) * size(test_images, 2), size(test_images, 3));
% Convert to double and rescale to [0,1]
test_images = double(test_images) / 255;



filename= 't10k-labels.idx1-ubyte';

%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the test_labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

test_labels = fread(fp, inf, 'unsigned char');

assert(size(test_labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

%% logistic regression

N = 60000;
D = 784;
K = 10;
eta = 0.01;
blr = ones(1,K);
Wlr = zeros(D,K);
optWlr = zeros(D,K);
t_train_k = zeros(N,K);
error_prev = 0;
eta_prev=0;

for n = 1:N
    t_train_k(n,train_labels(n)+1)=1;
end

%% iterating multiple times
maxIter = 30;

for iter = 1 : maxIter
    %% Training
    
    for tau = 1: N
        a = zeros(1,K);
        y = zeros(1,K);
        err = zeros(D,K);
        for k= 1:K
            a(k) = (Wlr(:,k).'* train_images(:,tau)) + blr(k);
        end
        a_max = max(a);
        a = a./a_max;
        den_y = sum(exp(a));
        for k = 1:K
            y(k) = exp(a(k))/(den_y);
        end
        
        [val idx] = max(y);
        for p=1:10
            if(p==idx)
                y(p) = 0.999999;
            else
                y(p) = 0.000001;
            end
        end
        err = ((y - t_train_k(tau,:)).' * train_images(:,tau).').';
        error = -sum(t_train_k(tau,:).* log(y));
        Wlr = Wlr - (eta .* err);
        eta_prev = eta;
        
        if abs(error) < abs(error_prev)  && eta > 0.000001
            eta = eta *0.999;
        end
        error_prev = error;
    end
    eta = eta_prev;
    
    %% Testing
    
    N2=10000;
    test_labels_k = zeros(N2,K);
    right_count=0;
    wrong_count=0;
    
    for i = 1: N2
        a = zeros(1,K);
        y = zeros(1,K);
        err = zeros(D,K);
        for k= 1:K
            a(k) = (Wlr(:,k)'* test_images(:,i)) + blr(k);
        end
        a_max = max(a);
        a = a./a_max;
        den_y = sum(exp(a));
        for k = 1:K
            y(k) = exp(a(k))./(den_y);
        end
        
        [val idx]=max(y);
        if((idx-1)==test_labels(i,1))
            right_count=right_count+1;
        else
            wrong_count=wrong_count+1;
        end
    end
    fprintf('iter#   : %d ',iter);
    fprintf('\teta   : %d ',eta);
    fprintf('\tRight : %d ',right_count);
    fprintf('\tWrong : %d ',wrong_count);
    fprintf('\tMCR   : %f \n',wrong_count*0.01);
end

%% neural network


N = 60000;
D = 784;
K = 10;
J_arr = [300];
%J_arr = [5 10 15 17 20 50 100 200 300 400 500 784];
%eta_arr = [0.1 0.01 0.001 0.0001 0.00001 0.000001 ];
eta_arr = [0.01];

h = 'sigmoid';
%% Functions

sigmoid = @(a) 1.0./(1.0 + exp(-a));

%%
J = J_arr(1);
Wnn1 = randn(D,J);
Wnn2 = randn(J,K);
bnn1 =  0.005*ones(1,J);
bnn2 =  0.005*ones(1,K);

maxIter=40;
for(iter=1:maxIter)
    
    for q = 1: size(J_arr , 2)
        J = J_arr(q);
        
        %% Training
        
        for qr = 1: size(eta_arr , 2)
            eta = eta_arr(qr);
            
            for tau = 1: N
                aj = zeros(1,J);
                ak = zeros(1,K);
                y = zeros(1,K);
                
                for j= 1:J
                    aj(j) = (Wnn1(:,j)'* train_images(:,tau)) + bnn1(j);
                end
                z = sigmoid(aj);
                for k = 1:K
                    ak(k) = (Wnn2(:,k)'* z') + bnn2(k);
                end
                den_y = sum(exp(ak));
                for k = 1:K
                    y(k) = exp(ak(k))./(den_y);
                end
                del_k = y - t_train_k(tau,:);
                for j=1:J
                    for k=1:10
                        err1(k) = Wnn2(j,k).*del_k(k);
                    end
                    del_j(j) = (sigmoid(z(j)).*(1 - sigmoid(z(j)))).*(sum(err1)) ;
                end
                
                % updating w
                for j=1:J
                    Wnn1(:,j) = Wnn1(:,j) - eta.*(del_j(j).*train_images(:,tau));
                end
                
                for k=1:K
                    Wnn2(:,k) = Wnn2(:,k) - (eta.*(del_k(k).*z))';
                end
                
            end
            
            %% Testing
            
            N1 = 10000;
            right_count=0;
            wrong_count=0;
            
            for tau = 1:N1
                aj = zeros(1,J);
                ak = zeros(1,K);
                z = zeros(1,J);
                y = zeros(1,K);
                
                for j= 1:J
                    aj(j) = (Wnn1(:,j)'* test_images(:,tau)) + bnn1(j);
                    z(j) = sigmoid(aj(j));
                end
                
                for k = 1:K
                    ak(k) = (Wnn2(:,k)'* z') + bnn2(k);
                end
                den_y = sum(exp(ak));
                for k = 1:K
                    y(k) = exp(ak(k))./(den_y);
                end
                
                [val idx]=max(y);
                if(idx==test_labels(tau,1)+1)
                    right_count=right_count+1;
                else
                    wrong_count=wrong_count+1;
                end
            end
            fprintf('iter : %d ',iter);
            fprintf('\teta : %d ',eta);
            fprintf('\tJ : %d ',J);
            fprintf('\t Right : %d ',right_count);
            fprintf('\t Wrong : %d \n',wrong_count);
        end
    end
end