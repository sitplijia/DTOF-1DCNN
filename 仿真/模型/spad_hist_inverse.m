function [lambda_hat] = spad_hist_inverse( ...
    H_obs, num_cycles, dead_bins, min_available)

% ============================================================
% SPAD histogram physical inversion
%
% 特别处理：
% 一旦出现第一个不可信bin，从该bin开始，
% 所有后续bin的反演结果均设置为NaN。
%
% 模型：
% 1. 单SPAD
% 2. 非瘫痪型死时间
% 3. 每周期开始时SPAD可用
% ============================================================

    if nargin < 4 || isempty(min_available)
        min_available = 0.02;
    end

    validateattributes(H_obs, {'numeric'}, ...
        {'vector','real','finite','nonnegative'});

    validateattributes(num_cycles, {'numeric'}, ...
        {'scalar','real','finite','positive'});

    validateattributes(dead_bins, {'numeric'}, ...
        {'scalar','integer','nonnegative'});

    validateattributes(min_available, {'numeric'}, ...
        {'scalar','real','finite','>',0,'<=',1});

    original_size = size(H_obs);
    H = double(H_obs(:)');

    N = double(num_cycles);
    M = numel(H);

    if any(H > N)
        error('Single-SPAD bin counts cannot exceed num_cycles.');
    end

    % --------------------------------------------------------
    % 1. Observed probability
    % --------------------------------------------------------
    q = H / N;

    % --------------------------------------------------------
    % 2. Initialize
    % --------------------------------------------------------
    available = zeros(1, M);
    p_cond = nan(1, M);
    lambda = nan(1, M);

    valid = false(1, M);
    saturated = false(1, M);
    H_forward = nan(1, M);

    dead_sum = 0;

    p_max = 1 - 1e-6;

    % First unreliable bin
    first_invalid_bin = NaN;

    % Once false, inversion is permanently disabled
    inversion_enabled = true;

    % --------------------------------------------------------
    % 3. Physical inversion
    % --------------------------------------------------------
    for i = 1:M

        % Update dead-time window
        j_remove = i - dead_bins - 1;

        if j_remove >= 1
            dead_sum = dead_sum - q(j_remove);
        end

        A = 1 - dead_sum;
        available(i) = max(0, min(1, A));

        % Observed events still affect subsequent availability
        if dead_bins > 0
            dead_sum = dead_sum + q(i);
        end

        % Stop all subsequent inversions
        if ~inversion_enabled
            continue;
        end

        % Condition 1: insufficient availability
        if A < min_available

            inversion_enabled = false;
            first_invalid_bin = i;

            continue;
        end

        % Conditional detection probability
        p = q(i) / A;
        p_cond(i) = p;

        % Condition 2: saturation or inconsistent probability
        if ~isfinite(p) || p < 0 || p >= p_max

            saturated(i) = true;
            inversion_enabled = false;
            first_invalid_bin = i;

            continue;
        end

        % Physical inversion
        lambda(i) = -log1p(-p);

        valid(i) = true;

        % Forward consistency
        q_pred = A * (1 - exp(-lambda(i)));
        H_forward(i) = N * q_pred;

    end

    % --------------------------------------------------------
    % 4. Restore original shape
    % --------------------------------------------------------
    lambda_hat = reshape(lambda, original_size);

end
