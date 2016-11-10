function [weight] = gauss_weight(win_size,sigma)
% This function creates gaussian weights with Variance Sigma to a window of
% size win_size
% Input: win_size: 2x1 vector, sigma: positive scalar
% Output: The corresponding weight matrix
     par_1 = (win_size-1)/2;     
     [x,y] = meshgrid(-par_1(2):par_1(2),-par_1(1):par_1(1));
     arg   = -(x.*x + y.*y)/(2*sigma^2);

     weight = exp(arg);
     weight(weight<eps*max(weight(:))) = 0;

     sumh = sum(weight(:));
     if sumh ~= 0,
     weight = weight/sumh;
     end;

end