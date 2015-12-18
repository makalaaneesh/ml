function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1); % s(j+1) by sj + 1

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% theta1 = 25 by 401
%  x/a1 = 5000 by 400
%see the videos for terminology
X = [ones(m,1), X]; % adding the bias element as 1


a1 = X;
z2 = Theta1 * a1';
a2 = sigmoid(z2);
% a2 is now 25 by 5000
% theta2 is 10 by 26
a2 = [ones(1,m);a2];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

% a3 is size 10 by 5000 (one column containing classifier results for each of the 5000 examples)


[value, index] = max(a3,[], 1); % dim 1 is column; 2 or row
p = index';








% =========================================================================


end
