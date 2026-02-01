
tic
% Fit gaussians to RFs
for i = 1:size(STA,1)

    % extract RF
    curr_RF = reshape(STA(i,:), [1360,768]);

    % fit 2D gaussian
    fit = fit_rf_gaussian_2d(curr_RF);

    % Calculate SNR
    SNR(i) = abs(fit.p(1)); % define SNR as |A| (amplitude) of fit gaussian
    threshold = 0.1;
    SNR_labels = SNR > threshold;

    if rem(i,10)==0
        disp(['cell ' num2str(i) '/' num2str(size(STA,1)) ' complete'])
    end

end

% Plot RFs and measures
% for i = 1:size(STA,1)
%     curr_RF = reshape(STA(i,:), [1360,768]);
%     imagesc(curr_RF)
%     title(['neuron #' num2str(i) ', SNR = ' num2str(SNR(i)) ', label = '...
%         num2str(labels(i)) ', SNR label = ' num2str(SNR_labels(i))])
%     pause
% end

toc

% compare original labels and SNR-based labels
figure
scatter([1:size(STA,1)]-0.1,labels,'ro')
hold on
scatter([1:size(STA,1)]+0.1,SNR_labels,'bo')
axis([0 size(STA,1)+1 -0.1 1.1])


%% gaussian function
function g = gauss2d_rot(p, X)

A     = p(1);
x0    = p(2);
y0    = p(3);
sx    = p(4);
sy    = p(5);
theta = p(6);
B     = p(7);

Xg = X{1};
Yg = X{2};

xp =  (Xg - x0) * cos(theta) + (Yg - y0) * sin(theta);
yp = -(Xg - x0) * sin(theta) + (Yg - y0) * cos(theta);

g = A * exp(-0.5 * (xp.^2 ./ sx.^2 + yp.^2 ./ sy.^2)) + B;
end

%% fitting function
function fit = fit_rf_gaussian_2d(rf)
% input: rf - [ny x nx] receptive field map
%
% output: fit - fields:
%   p        fitted parameters
%   rf_hat   fitted RF
%   resnorm  sum of squared residuals
%   exitflag lsqcurvefit exit code

rf = double(rf);
[ny, nx] = size(rf);

[Xg, Yg] = meshgrid(1:nx, 1:ny);

[B0, idxMin] = min(rf(:));
[A0, idxMax] = max(rf(:));

if abs(A0) >= abs(B0)
    amp0 = A0 - median(rf(:));
    idx0 = idxMax;
else
    amp0 = B0 - median(rf(:));
    idx0 = idxMin;
end

[y0, x0] = ind2sub(size(rf), idx0);

sx0 = nx / 6;
sy0 = ny / 6;
theta0 = 0;
B0 = median(rf(:));

p0 = [amp0, x0, y0, sx0, sy0, theta0, B0];

% bounds
lb = [-Inf,   1,   1,  0.5,  0.5, -pi/2, -Inf];
ub = [ Inf,  nx,  ny,  nx,   ny,   pi/2,  Inf];

opts = optimoptions('lsqcurvefit', ...
    'Display','off', ...
    'MaxFunctionEvaluations', 1e4);

[p, resnorm, ~, exitflag] = lsqcurvefit( ...
    @(p,X) gauss2d_rot(p,X), ...
    p0, {Xg,Yg}, rf, lb, ub, opts);

rf_hat = gauss2d_rot(p, {Xg,Yg});

fit = struct();
fit.p = p;
fit.rf_hat = rf_hat;
fit.resnorm = resnorm;
fit.exitflag = exitflag;
end
