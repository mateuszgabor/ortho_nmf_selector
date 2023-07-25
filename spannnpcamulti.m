function [W] = spannnpcamulti(M, k, varargin)
% [W, H] = SPANNNPCAMULTI(M, K) returns a N x K nonnegative matrix W with
% orthonormal columns that contain the extracted (approximate) nonnegative
% components of the m x n (raw) data matrix M.
%
% SPANNNPCAMULTI(...,'approximationrank',r) uses a rank-r approximation of
% the input to solve the nn PCA problem. Default value is r=5.
% SPANNNPCAMULTI(...,'numsamples',T) determines the number of samples drawn
% (iterations performed) by the algorithm. Default is T=1e5.
% SPANNNPCAMULTI(...,'verbose',true) allows the algorithm to print progress
% messages and display a progress bar (if available). 
% SPANNNPCAMULTI(...,'verbose',false) suppresses messages and progress bar.

%rng('default')

[m, n] = size(M);

defaultApproximationRank = 5;
defaultNumSamples = 1e5;
defaultVerbose = true;

% Auxiliary functions for checking parameter values:
ispositiveint = @(x) (isnumeric(x) && numel(x) == 1 && ...
                      mod(x, 1) == 0 && x > 0);

% Register input parameters:
parser = inputParser;
addRequired(parser, 'M');
addRequired(parser, 'k', ispositiveint);
addParameter(parser, 'approximationrank', defaultApproximationRank, ...
             ispositiveint);
addParameter(parser, 'numsamples', defaultNumSamples, ispositiveint);
addParameter(parser, 'verbose', defaultVerbose, @islogical);

% Parse input arguments:
parse(parser, M, k, varargin{:});
approxrank = parser.Results.approximationrank;
numsamples = parser.Results.numsamples;
verbose = parser.Results.verbose;

maxallowedrank = min([m, n]);
if approxrank > maxallowedrank
   warning('approxrank > min(m, n); reducing to %d', maxallowedrank);
   approxrank = maxallowedrank;
end

% Check if the textprogressbar is available in Matlab's path:
tpbavailable = (exist('textprogressbar', 'file') == 2);

% Compute low-rank approximation:
if verbose
    tsvd = tic;
    fprintf('[spannnpca:] Computing rank-%d approximation...', approxrank);
end
[~, S, V] = svds(M, approxrank, 'L');
VS = V*S;
if verbose
    fprintf('Done. (%d seconds)\n', round(toc(tsvd)));
end

signpatterns = de2bi(0:2^k-1, k);
signpatterns(signpatterns==0) = -1;

iter = 0;
optval = -Inf;
fun_obj_opt = -Inf;
optZ = zeros([approxrank,k]);

if verbose && tpbavailable
   tpbupd = textprogressbar(numsamples, ...
                            'updatestep', 100, ...
                            'startmsg', '[spannnpca:] Sampling...'); 
end
% Run algorithm (main sampling loop):
while(true)
    
    iter = iter + 1;
    if iter > numsamples
        break;
    end
    
    if verbose && tpbavailable
        tpbupd(iter);
    end
    % Generate a n x k matrix (columns in the span of V):
    Z = sphereCartesianSample(approxrank, k, optZ, iter);
    A = VS * Z;

    [W, localobj] = solveLocalMaximization(A, signpatterns);
 
  %  fun_obj(iter) = sum(sum(A.*full(W),1),2);
    W0_norm = zeros(k,1);
    for i = 1:k
        W0_norm(i) = length(find(W(:,i)));
    end
    if (localobj > optval) & rank(full(W)) > 1 & prod(W0_norm > .1*size(W,1)) & ( max(abs(diff(W0_norm))) > mean(W0_norm))
        optval = localobj;
        optW = W;
        optZ = Z;
        iter
    end
% 
%      if (fun_obj(iter) > fun_obj_opt) & rank(full(W)) > 1
%         fun_obj_opt = fun_obj(iter);
%         optW = W;
%         iter
%     end
    
end

W = optW;
end

%% Auxiliary functions

function [W, localobj] = solveLocalMaximization(A, patterns)
% Solves the `local` maximization problem on a n x k matrix A.

    localobj = 0;
    [m, k] = size(A);
    
    B = A;
 %   for p = 1:size(patterns, 1)
 %       B = bsxfun(@times, A, patterns(p, :));
 %       B(B<0) = 0;
    
        [selEntryVals, colAssignment] = max(B, [], 2);
        
        % Compute the objective that will be achieved by the best W for
        % this particular sign patter:
        obj = sum(selEntryVals.^2);

        selEntryVals_pos = zeros(size(B,1),1);
        colAssignment_pos = zeros(size(B,1),1);
       
        inx_pos = selEntryVals > 0;
        selEntryVals_pos(inx_pos) = selEntryVals(inx_pos);
        colAssignment_pos(inx_pos) = colAssignment(inx_pos);
        
        if obj > localobj
            % If obj is the best encountered so far, update localobj:
            localobj = obj;
            
            % Also, update W:
            for i = 1:k
                ithcolentries_pos = (colAssignment_pos==i);
                ithcolnorm = sqrt(sum(selEntryVals_pos(ithcolentries_pos).^2));
                if ithcolnorm == 0
                    % If column did not get any entries, skip: 
                    continue; 
                end
                selEntryVals_pos(ithcolentries_pos) = ...
                    selEntryVals_pos(ithcolentries_pos) / ithcolnorm;
            end
            W = sparse(1:m, colAssignment, selEntryVals_pos, m, k);
        end
 %   end
end

function [ C ] = sphereCartesianSample(dim, pow, Zm,iter)
    
 %   C = randl(dim, pow);
 %   C = C * diag(sum(C.^2, 1).^-0.5); % original
 %   C = C * diag(1./sum(abs(C), 1));

    alpha = max(.1,1 - tanh(iter));
    Cm = randl(dim, pow);
    C = Zm + alpha*Cm;
   % C = C * diag(1./sum(abs(C), 1));
    C = C * diag(sum(C.^2, 1).^-0.5);

end

function bi = de2bi(n, bitnum)
% Converts a vector n of integers to a bit stream.
% The columns of this matrix will store the 8-bit binary representations of
% the integers contained in vector n. Most Significant bit is at the top of
% each column and Least Significant bit is at the bottom

bi = zeros(bitnum, length(n));

for k=bitnum:-1:1,
    bit=mod(n,2);
    bi(k,:)= bit';
    n = n-bit;
    n = n/2;
end

bi = transpose(bi); %Stacks the columns of the matrix to create a vector
end
