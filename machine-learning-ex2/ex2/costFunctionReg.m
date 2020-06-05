function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
e=zeros(m,1);
a=X*theta;
for i=1:m
    e(i)=(1/(1+exp(-a(i))));
end
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% identical to costFunction.m
h = sigmoid(X * theta);

% cost term - identical to costFunction.m
cost = (1 / m) * sum((-y .* log(h)) - ((1 - y) .* log(1 - (h))));
theta2=theta.*theta;

% regularize term - skipping theta(0)
% "end" keyword here refers to last element in matrix so we can subset the matrix
regCost = (lambda / (2 * m)) *(sum(theta2)-theta2(1));

% identical to costFunction.m
for i=1:length(theta)
    grad(i)=(1/m)*sum((e-y).*X(:,i));
end

% regularize gradient term - skipping theta(0)
regGrad = (lambda / m) .* theta;
regGrad(1) = 0;

% Regularization just consists of summing the original cost
% with the regularized term.
J = cost + regCost;
grad = grad + regGrad;

% =============================================================

end