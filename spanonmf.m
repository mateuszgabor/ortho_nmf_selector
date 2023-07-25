function [W, H, err] = spanonmf(M, k, varargin)
% [W, H] = SPANONMF(M, K) returns an approximate factorization of the
% nonnegative m x n matrix M into an m x k nonnegative matrix W with
% orthogonal columns and a k x n nonnegative matrix H.
% 
% SPANONMF(...,'approximationrank',r) uses a rank-r approximation of
% the input to solve the nn PCA problem. Default value is r=5.
% SPANONMF(...,'numsamples',T) determines the number of samples drawn
% (iterations performed) by the algorithm. Default is T=1e5.
% SPANONMF(...,'verbose',true) allows the algorithm to print progress
% messages and display a progress bar (if available). 
% SPANONMF(...,'verbose',false) suppresses messages and progress bar.

% rng('default')

[m, n] = size(M);

defaultApproxRank = 5;
defaultNumSamples = 1e5;

% Auxiliary functions for checking parameter values:
ispositiveint = @(x) (isnumeric(x) && numel(x) == 1 && ...
                      mod(x, 1) == 0 && x > 0);
isnonnegativematrix = @(x) (isnumeric(x) && ~any(x(:) < 0));

% Register input parameters:
parser = inputParser;
addRequired(parser, 'M', isnonnegativematrix);
addRequired(parser, 'k', ispositiveint);
addParameter(parser, 'approximationrank', defaultApproxRank,...
             ispositiveint);
addParameter(parser, 'numsamples', defaultNumSamples, ispositiveint);

% Parse input arguments:
parse(parser, M, k, varargin{:});
approxrank = parser.Results.approximationrank;
numsamples = parser.Results.numsamples;

maxallowedrank = min([m, n]);
if approxrank > maxallowedrank
   warning('approxrank > min(m, n); reducing to %d', maxallowedrank);
   approxrank = maxallowedrank;
end


% Run spannnpca to compute W:
[W] = spannnpcamulti(M', k, ...
                     'approximationrank', approxrank, ...
                     'numsamples', numsamples);

% Now that W is determined, we can also re-optimize the columns for the
% given supports:
for col = 1:k
   support = find(W(:, col));
   [values, ~, ~] = svds(M(support, :), 1);
   W(support, col) = abs(values);
end

% Compute H given W:
H = W' * M;

err = norm(M-W*H, 'fro');


