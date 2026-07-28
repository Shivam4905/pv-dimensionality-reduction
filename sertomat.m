

function [Data, Target,Persis ] = sertomat(Input,Dim_data,pred_hor)

% Radial Basis Function Network for Regression.
%
%   Input:   time series; 
%   Dim_data:   Dimension of data;
%   pred_hor:   forecasting horizon;
%   Data:   output matrix data;
%   Target:   target to be forecasted;


%   Created by Hassen Bouzgou 01/2017.



N=length(Input);
n=Dim_data+pred_hor;

p=Dim_data+pred_hor-1;

m=N+1-n;
for j=1:m
    for(i=1:n)
    y(j,i) = Input((i+j-1));
    end
end


Data=y(:,1:Dim_data);
Target=y(:,n);
Persis=y(:, n-1);