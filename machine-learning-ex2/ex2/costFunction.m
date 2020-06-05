function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

t=zeros(m,1);

for i=1:length(theta)
    grad(i)=(1/m)*sum((e-y).*X(:,i));
end

for i=1:m
    %disp(i)
    t(i)=(-1/m)*((y(i)*log(e(i)))+(1-y(i))*(log(1-e(i))));
end
J=sum(t);





% =============================================================

end
